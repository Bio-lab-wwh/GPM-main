"""Training, validation, and inference utilities for GPM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.optim.lr_scheduler import StepLR

from GPM.model import GPMModel, build_gpm_model
from GPM.utils import (
    GPMBatch,
    GPMInferenceResult,
    GPMSample,
    PackedTrainingMicroBatch,
    collate_gpm_samples,
    iter_packed_training_micro_batches,
    iter_prefetched_gpm_batches,
    iter_sample_batches,
    tensor_to_float32_numpy,
)
from Tools.Loss import (
    GPMLoss,
    GPMLossOutput,
    GPMLossTargets,
    merge_loss_weights,
)

logger = logging.getLogger(__name__)
GPM_ARCHITECTURE_VERSION = "GPM 2.1"
GPM_PRIMARY_INFERENCE_BRANCH = (
    "continuous_background_graph_first_structured_intervention_response"
)


@dataclass
class GPMTrainConfig:
    """Training configuration for ``GPMRunner``."""

    epochs: int
    lr: float
    lr_decay: float
    weight_decay: float
    device: str
    batch_size: int
    pert_group_size: int
    seed: int
    patience: int
    gene_embedding_dim: int
    cell_context_dim: int
    enhanced_node_dim: int
    edge_chunk_size: int
    technology_embedding_dim: int
    technology_names: Sequence[str]
    grad_clip_norm: float
    loss_blur: float
    loss_sinkhorn_iters: int
    print_every_n_groups: int
    min_groups: int = 10_000
    biogrid_path: Optional[Union[str, Path]] = None
    gene_embedding_path: Optional[Union[str, Path]] = None
    loss_weights: Optional[dict[str, float]] = None
    min_free_memory_gib: float = 4.0


@dataclass
class GPMStepMetrics:
    """Metrics for a training or validation step."""

    loss_total: float
    loss_sinkhorn: float
    loss_mse_expression: float
    n_groups: int = 0
    n_samples: int = 0
    n_response_samples: int = 0
    response_rms: float = 0.0
    n_gradient_samples: int = 0
    grad_context_encoder: float = 0.0
    grad_technology: float = 0.0
    grad_node_encoder: float = 0.0
    grad_graph_encoder: float = 0.0
    grad_intervention_generator: float = 0.0
    grad_propagator: float = 0.0
    grad_decoder: float = 0.0


@dataclass
class _GPMStepDeviceMetrics:
    """Step metrics stored on the compute device until reporting."""
    loss_values: torch.Tensor  # total, Sinkhorn, MSE
    response_values: torch.Tensor
    gradient_values: torch.Tensor
    n_samples: int
    n_response_samples: int
    n_gradient_samples: int


def unwrap_compiled_model(model: nn.Module) -> GPMModel:
    """Return the GPM model underlying an optional compilation wrapper."""
    if not isinstance(model, GPMModel):
        raise TypeError(f"Expected a GPMModel instance, got {type(model)}.")
    return model


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    req = str(device).strip().lower()
    if req.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if req.startswith("cuda"):
        logger.warning("CUDA is unavailable. Using the CPU.")
    return torch.device("cpu")


def _format_loss_metrics(
    m: GPMStepMetrics,
) -> str:
    """Format only optimization losses for concise console logs."""
    return (
        f"total={m.loss_total:.6f} | "
        f"mse_expression={m.loss_mse_expression:.6f} | "
        f"sinkhorn={m.loss_sinkhorn:.6f}"
    )


def _device_metrics_to_host(metrics: _GPMStepDeviceMetrics) -> GPMStepMetrics:
    """Copy device metrics to the host for reporting."""
    losses = metrics.loss_values.cpu().tolist()
    response = metrics.response_values.cpu().tolist()
    gradients = metrics.gradient_values.cpu().tolist()
    return GPMStepMetrics(
        loss_total=float(losses[0]),
        loss_sinkhorn=float(losses[1]),
        loss_mse_expression=float(losses[2]),
        n_samples=metrics.n_samples,
        n_response_samples=metrics.n_response_samples,
        response_rms=float(response[0]),
        n_gradient_samples=metrics.n_gradient_samples,
        grad_context_encoder=float(gradients[0]),
        grad_technology=float(gradients[1]),
        grad_node_encoder=float(gradients[2]),
        grad_graph_encoder=float(gradients[3]),
        grad_intervention_generator=float(gradients[4]),
        grad_propagator=float(gradients[5]),
        grad_decoder=float(gradients[6]),
    )


def _module_gradient_norm(module: nn.Module) -> torch.Tensor:
    """Compute the L2 norm across all parameter gradients on-device."""
    parameters = tuple(module.parameters())
    if not parameters:
        return torch.zeros((), dtype=torch.float32)
    total = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total.add_(parameter.grad.detach().float().square().sum())
    return total.sqrt()


class GPMRunner:
    """Run individual training epochs, make predictions, and manage checkpoints."""

    def __init__(
        self,
        model: GPMModel,
        *,
        device: Union[str, torch.device] = "cuda",
        loss_blur: float = 0.02,
        loss_sinkhorn_iters: int = 50,
        loss_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.loss_blur = float(loss_blur)
        self.loss_sinkhorn_iters = int(loss_sinkhorn_iters)
        self.loss_weights = merge_loss_weights(loss_weights)
        self.criterion = GPMLoss(
            blur=self.loss_blur,
            sinkhorn_iters=self.loss_sinkhorn_iters,
            weights=self.loss_weights,
        )

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._lr_scheduler: Optional[StepLR] = None
        self._grad_clip: float = 0.0
        self._ctrl_mean: Optional[np.ndarray] = None
        self._grad_scaler = amp.GradScaler(
            enabled=self.device.type == "cuda",
        )
        self._accel = amp.autocast(
            device_type=self.device.type,
            enabled=self.device.type == "cuda",
            dtype=torch.float16,
        )

    @classmethod
    def from_config(
        cls,
        config: GPMTrainConfig,
        *,
        gene_names: Sequence[str],
        **kwargs: Any,
    ) -> GPMRunner:
        """Build a runner and model from a training configuration."""
        model = build_gpm_model(
            gene_names=gene_names,
            gene_embedding_dim=config.gene_embedding_dim,
            cell_context_dim=config.cell_context_dim,
            enhanced_node_dim=config.enhanced_node_dim,
            edge_chunk_size=config.edge_chunk_size,
            technology_names=config.technology_names,
            technology_embedding_dim=config.technology_embedding_dim,
            biogrid_path=config.biogrid_path,
            gene_embedding_path=config.gene_embedding_path,
        )

        loss_weights = merge_loss_weights(config.loss_weights)
        runner = cls(
            model,
            device=config.device,
            loss_blur=config.loss_blur,
            loss_sinkhorn_iters=config.loss_sinkhorn_iters,
            loss_weights=loss_weights,
            **kwargs,
        )
        runner.build_optimizer(config)
        runner._grad_clip = float(config.grad_clip_norm)
        return runner

    def set_ctrl_mean(self, ctrl_mean: Union[np.ndarray, Sequence[float]]) -> None:
        """Set the reference control mean used to report predicted effects."""
        self._ctrl_mean = np.asarray(ctrl_mean, dtype=np.float32).ravel()

    def build_optimizer(self, config: GPMTrainConfig) -> torch.optim.Optimizer:
        """Build AdamW with optional exponential learning rate decay."""
        self._optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        decay = float(config.lr_decay)
        if decay != 1.0:
            self._lr_scheduler = StepLR(self._optimizer, step_size=1, gamma=decay)
        else:
            self._lr_scheduler = None
        return self._optimizer

    def current_lr(self) -> Optional[float]:
        """Return the current learning rate, or ``None`` before optimizer setup."""
        if self._optimizer is None:
            return None
        return float(self._optimizer.param_groups[0]["lr"])

    def step_lr_scheduler(self) -> Optional[float]:
        """Advance the epoch scheduler and return the updated learning rate."""
        if self._lr_scheduler is None:
            return self.current_lr()
        self._lr_scheduler.step()
        return self.current_lr()


    def _forward_batch(self, batch: GPMBatch) -> tuple[Any, GPMLossOutput]:
        """Run the model and compute group-mean MSE and Sinkhorn losses."""
        model = unwrap_compiled_model(self.model)

        if batch.any_control:
            raise ValueError(
                "Training batches must contain perturbed samples with paired control inputs."
            )

        out = model(
            batch.expression,
            batch.pert_tech_ids,
            batch.target_gene_indices,
            control_context_mean=batch.control_context_mean,
            control_context_variance=batch.control_context_variance,
            is_control=batch.is_control,
            target_groups=batch.target_groups,
        )
        loss_out = self.criterion(
            out.pred_perturbed_expression,
            GPMLossTargets(
                target_expression=batch.target_expression,
                pert_group_slices=batch.pert_group_slices or None,
            ),
        )
        return out, loss_out

    def _forward_batch_infer(self, batch: GPMBatch) -> Any:
        """Run the model without computing a training loss."""
        model = unwrap_compiled_model(self.model)
        return model(
            batch.expression,
            batch.pert_tech_ids,
            batch.target_gene_indices,
            control_context_mean=batch.control_context_mean,
            control_context_variance=batch.control_context_variance,
            is_control=batch.is_control,
            target_groups=batch.target_groups,
        )

    def _train_step_micro_batch(
        self,
        packed: PackedTrainingMicroBatch,
        batch: Optional[GPMBatch] = None,
        *,
        measure_gradient_norms: bool = False,
    ) -> _GPMStepDeviceMetrics:
        """Optimize one micro-batch containing one or more perturbation groups."""
        if self._optimizer is None:
            raise RuntimeError("Call build_optimizer before training.")

        if batch is None:
            model = unwrap_compiled_model(self.model)
            batch = collate_gpm_samples(
                packed.samples,
                tech_encoder=model.technology_embedding,
                device=self.device,
                pert_group_slices=packed.pert_group_slices,
            )

        self._optimizer.zero_grad(set_to_none=True)
        with self._accel:
            batch_out, batch_loss = self._forward_batch(batch)
        self._grad_scaler.scale(batch_loss.loss_total).backward()
        loss_values = torch.stack((
            batch_loss.loss_total,
            batch_loss.loss_sinkhorn,
            batch_loss.loss_mse_expression,
        )).detach().float()
        self._grad_scaler.unscale_(self._optimizer)
        model = unwrap_compiled_model(self.model)
        if measure_gradient_norms:
            gradient_values = torch.stack(tuple(
                _module_gradient_norm(module)
                for module in (
                    model.cell_context_encoder,
                    model.technology_embedding,
                    model.node_encoder,
                    model.graph_encoder,
                    model.intervention_generator,
                    model.propagator,
                    model.expression_decoder,
                )
            )).detach().float()
        else:
            gradient_values = torch.zeros(
                7, device=self.device, dtype=torch.float32
            )
        if self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)

        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

        n_samples = int(batch.expression.shape[0])
        return _GPMStepDeviceMetrics(
            loss_values=loss_values,
            response_values=batch_out.response_rms.detach().float().reshape(1),
            gradient_values=gradient_values,
            n_samples=n_samples,
            n_response_samples=n_samples,
            n_gradient_samples=n_samples if measure_gradient_norms else 0,
        )

    @torch.no_grad()
    def _eval_step_micro_batch(
        self,
        packed: PackedTrainingMicroBatch,
        batch: Optional[GPMBatch] = None,
    ) -> _GPMStepDeviceMetrics:
        """Evaluate one micro-batch without updating parameters."""
        if batch is None:
            model = unwrap_compiled_model(self.model)
            batch = collate_gpm_samples(
                packed.samples,
                tech_encoder=model.technology_embedding,
                device=self.device,
                pert_group_slices=packed.pert_group_slices,
            )

        with self._accel:
            batch_out, batch_loss = self._forward_batch(batch)
        loss_values = torch.stack((
            batch_loss.loss_total,
            batch_loss.loss_sinkhorn,
            batch_loss.loss_mse_expression,
        )).detach().float()
        n_samples = int(batch.expression.shape[0])
        return _GPMStepDeviceMetrics(
            loss_values=loss_values,
            response_values=batch_out.response_rms.detach().float().reshape(1),
            gradient_values=torch.zeros(7, device=self.device, dtype=torch.float32),
            n_samples=n_samples,
            n_response_samples=n_samples,
            n_gradient_samples=0,
        )

    def _run_epoch_on_samples(
        self,
        samples: Sequence[GPMSample],
        *,
        train: bool,
        group_size: int,
        micro_batch_size: Optional[int] = None,
        seed: int,
        print_every: int = 1,
        epoch: int = 0,
        split_name: str = "train",
    ) -> GPMStepMetrics:
        """Run one epoch over packed perturbation groups."""
        if train:
            self.model.train()
        else:
            self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        micro_bs = max(1, int(micro_batch_size or group_size))
        packed_batches = list(
            iter_packed_training_micro_batches(
                samples,
                group_size=group_size,
                micro_batch_size=micro_bs,
                shuffle=train,
                seed=seed + epoch,
            )
        )
        if not packed_batches:
            logger.warning(
                "[%s epoch %d] No valid micro-batches (group size: %d, micro-batch size: %d).",
                split_name,
                epoch,
                group_size,
                micro_bs,
            )
            return GPMStepMetrics(0.0, 0.0, 0.0, n_samples=0)

        n_batches = len(packed_batches)
        loss_sums = torch.zeros(3, device=self.device, dtype=torch.float32)
        response_sums = torch.zeros(1, device=self.device, dtype=torch.float32)
        gradient_sums = torch.zeros(7, device=self.device, dtype=torch.float32)
        total_samples = 0
        total_groups = 0
        total_response_samples = 0
        total_gradient_samples = 0
        model = unwrap_compiled_model(self.model)
        prefetched_batches = iter_prefetched_gpm_batches(
            packed_batches,
            tech_encoder=model.technology_embedding,
            device=self.device,
        )
        for bi, (packed, batch) in enumerate(prefetched_batches):
            should_print = print_every > 0 and (bi + 1) % print_every == 0
            try:
                if train:
                    device_metrics = self._train_step_micro_batch(
                        packed,
                        batch,
                        measure_gradient_norms=should_print,
                    )
                else:
                    device_metrics = self._eval_step_micro_batch(packed, batch)
                n_samples = max(1, device_metrics.n_samples)
                if train:
                    total_groups += len(packed.pert_group_slices)
                loss_sums.add_(device_metrics.loss_values, alpha=float(n_samples))
                total_samples += n_samples
                if device_metrics.n_response_samples > 0:
                    response_sums.add_(
                        device_metrics.response_values,
                        alpha=float(device_metrics.n_response_samples),
                    )
                    total_response_samples += device_metrics.n_response_samples
                if device_metrics.n_gradient_samples > 0:
                    gradient_sums.add_(
                        device_metrics.gradient_values,
                        alpha=float(device_metrics.n_gradient_samples),
                    )
                    total_gradient_samples += device_metrics.n_gradient_samples
            except Exception as exc:
                pk = packed.samples[0].resolved_panel_key() if packed.samples else "?"
                cond = packed.samples[0].condition if packed.samples else "?"
                raise RuntimeError(
                    f"[{split_name} epoch {epoch}] Micro-batch failed "
                    f"panel={pk!r} condition={cond!r} "
                    f"size={len(packed.samples)} slices={packed.pert_group_slices}: {exc}"
                ) from exc

            if should_print:
                m = _device_metrics_to_host(device_metrics)
                print(
                    f"  [{split_name} epoch {epoch} batch {bi + 1}/{n_batches}] "
                    f"n={len(packed.samples)} slices={len(packed.pert_group_slices)} "
                    f"{_format_loss_metrics(m)}"
                )

        agg = _device_metrics_to_host(_GPMStepDeviceMetrics(
            loss_values=loss_sums / max(1, total_samples),
            response_values=response_sums / max(1, total_response_samples),
            gradient_values=gradient_sums / max(1, total_gradient_samples),
            n_samples=total_samples,
            n_response_samples=total_response_samples,
            n_gradient_samples=total_gradient_samples,
        ))
        agg.n_groups = total_groups
        print(
            f"[{split_name} epoch {epoch} summary: {n_batches} micro-batches, "
            f"micro_batch_size={micro_bs}, completed_groups={total_groups}] "
            f"{_format_loss_metrics(agg)}"
        )
        return agg

    @torch.no_grad()
    def condition_mean_effect_pcc(
        self,
        samples: Sequence[GPMSample],
        *,
        control_means_by_panel: Mapping[str, np.ndarray],
        batch_size: int,
        desc: str = "validation perturbation effect",
    ) -> float:
        """Average control-referenced effect correlations across dataset-condition pairs."""
        if not samples:
            raise ValueError("Pearson correlation of perturbation effects requires a nonempty validation set.")
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        model = unwrap_compiled_model(self.model)
        sample_batches = list(
            iter_sample_batches(samples, max(1, int(batch_size)), shuffle=False, seed=0)
        )
        packed_batches = [
            PackedTrainingMicroBatch(samples=tuple(group), pert_group_slices=())
            for group in sample_batches
        ]
        sums: dict[tuple[str, str], list[Any]] = {}
        for packed, batch in iter_prefetched_gpm_batches(
            packed_batches,
            tech_encoder=model.technology_embedding,
            device=self.device,
        ):
            with self._accel:
                output = self._forward_batch_infer(batch)
            pred = output.pred_perturbed_expression.detach().float().cpu()
            truth = batch.target_expression.detach().float().cpu()
            for row, sample in enumerate(packed.samples):
                key = (sample.resolved_panel_key(), str(sample.condition))
                if key not in sums:
                    sums[key] = [
                        pred[row].double().clone(),
                        truth[row].double().clone(),
                        1,
                    ]
                else:
                    sums[key][0].add_(pred[row].double())
                    sums[key][1].add_(truth[row].double())
                    sums[key][2] += 1
            del batch, output, pred, truth

        correlations: list[torch.Tensor] = []
        skipped_conditions = 0
        for (panel_key, _condition), (pred_sum, truth_sum, count) in sums.items():
            if panel_key not in control_means_by_panel:
                raise KeyError(f"No saved control mean for validation dataset {panel_key!r}.")
            pred_mean = pred_sum / int(count)
            truth_mean = truth_sum / int(count)
            control_mean = torch.as_tensor(
                np.asarray(control_means_by_panel[panel_key], dtype=np.float64),
                dtype=torch.float64,
            )
            if control_mean.shape != pred_mean.shape:
                raise ValueError(
                    f"{panel_key}: Control mean shape {tuple(control_mean.shape)} does not match model output shape "
                    f"{tuple(pred_mean.shape)}."
                )

            pred_effect = pred_mean - control_mean
            truth_effect = truth_mean - control_mean
            pred_centered = pred_effect - pred_effect.mean()
            truth_centered = truth_effect - truth_effect.mean()
            denominator = torch.linalg.vector_norm(pred_centered) * torch.linalg.vector_norm(
                truth_centered
            )
            if (
                not torch.isfinite(pred_centered).all()
                or not torch.isfinite(truth_centered).all()
                or not torch.isfinite(denominator)
                or float(denominator) <= 1e-12
            ):
                skipped_conditions += 1
                continue
            correlations.append(
                torch.dot(pred_centered, truth_centered) / denominator
            )

        if not correlations:
            raise RuntimeError(
                "Pearson correlation is undefined for all validation conditions because the effect vectors have zero variance or contain non-finite values."
            )
        score = float(torch.stack(correlations).mean())
        print(
            f"[{desc}] mean_effect_pcc={score:.6f} "
            f"valid_conditions={len(correlations)}/{len(sums)} "
            f"skipped_constant_or_nonfinite={skipped_conditions}",
            flush=True,
        )
        return score

    @torch.no_grad()
    def predict(
        self,
        samples: Sequence[GPMSample],
        *,
        batch_size: int = 16,
        desc: str = "predict",
    ) -> list[GPMInferenceResult]:
        """Predict expression for each sample in panel-specific batches.

        Call ``set_ctrl_mean`` first to report ``pred_delta`` relative to the dataset
        control mean.
        """
        self.model.eval()
        if self._ctrl_mean is None:
            raise RuntimeError("Call set_ctrl_mean before inference.")

        ctrl = self._ctrl_mean
        results: list[GPMInferenceResult] = []
        model = unwrap_compiled_model(self.model)
        tech_enc = model.technology_embedding

        sample_batches = list(
            iter_sample_batches(samples, max(1, int(batch_size)), shuffle=False, seed=0)
        )
        packed_batches = [
            PackedTrainingMicroBatch(samples=tuple(group), pert_group_slices=())
            for group in sample_batches
        ]
        prefetched_batches = iter_prefetched_gpm_batches(
            packed_batches,
            tech_encoder=tech_enc,
            device=self.device,
        )

        for bi, (packed, batch) in enumerate(prefetched_batches):
            group = packed.samples
            try:
                with self._accel:
                    out = self._forward_batch_infer(batch)

                selected_expression = out.pred_perturbed_expression
                pred_e = tensor_to_float32_numpy(selected_expression)
                for j in range(pred_e.shape[0]):
                    pe = pred_e[j]
                    # Use the same dataset control mean for the reported predictions and observations.
                    pd = (pe - ctrl).astype(np.float32)
                    results.append(
                        GPMInferenceResult(
                            pred_expression=pe,
                            pred_delta=pd,
                            sample_id=group[j].sample_id,
                        )
                    )
                del batch, out, selected_expression
            except Exception as exc:
                pk = group[0].resolved_panel_key() if group else "?"
                raise RuntimeError(
                    f"[{desc} batch {bi + 1}/{len(packed_batches)}] panel={pk!r} inference failed: {exc}"
                ) from exc

        if len(results) != len(samples):
            raise RuntimeError(
                f"[{desc}] Produced {len(results)} predictions for {len(samples)} input samples."
            )
        return results


    def collate_batch(self, samples: Sequence[GPMSample]) -> GPMBatch:
        """Collate samples into a ``GPMBatch``."""
        model = unwrap_compiled_model(self.model)
        return collate_gpm_samples(
            samples,
            tech_encoder=model.technology_embedding,
            device=self.device,
        )

    def save_checkpoint(
        self,
        path: Union[str, Path],
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Save model and optimizer state."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model = unwrap_compiled_model(self.model)
        payload: dict[str, Any] = {
            "architecture": GPM_ARCHITECTURE_VERSION,
            "gene_names": model.gene_names,
            "gene_embedding_source": model.gene_embedding_source,
            "technology_names": model.technology_names,
            "technology_embedding_dim": model.technology_embedding.embedding_dim,
            "model": model.state_dict(),
            "loss_weights": dict(self.loss_weights),
            "primary_inference_branch": GPM_PRIMARY_INFERENCE_BRANCH,
            "training_objective": "group_mean_expression_mse_plus_sinkhorn",
            "execution_config": {
                "receiver_chunk_size": model.propagator.receiver_chunk_size,
                "graph_layers": model.graph_encoder.num_layers,
            },
            "mechanism": {
                "initial_support": "perturbation_targets_only",
                "graph_propagation_path": (
                    "one_step_target_to_all_receivers_conditioned_by_"
                    "single_layer_hybrid_graph_encoder"
                ),
                "priors": (
                    "BIOGRID physical_genetic_sparse_one_hop_plus_"
                    "global_linear_attention"
                ),
                "cell_context": (
                    "per_sample_global_context_from_cell_line_mean_variance_"
                    "and_paired_control_offset"
                ),
                "enhanced_node_state": (
                    "H0_g=GeneIdentityQuery(gene_statistics_token,"
                    "global_cell_context_token)"
                ),
                "cell_context_batch_dependent": True,
                "normalization": (
                    "upstream_normalize_total_log1p_only; "
                    "no_model_expression_standardization_or_robust_scale"
                ),
                "heldout_cell_controls_used_for_optimization": False,
                "gene_identity": "trainable_biogrid_contrastive_prior_or_random_fallback",
                "gene_identity_source": model.gene_embedding_source,
                "initial_gene_state": (
                    "pretrained_gene_identity_querying_two_continuous_background_tokens"
                ),
                "technology_conditioning": {
                    "vocabulary_source": "current_experiment_dataset_labels",
                    "embedding_dim": model.technology_embedding.embedding_dim,
                    "generator": "graph_enhanced_target_node_and_technology_structural_equation",
                    "parameterization": "single_sparse_intervention_vector_I",
                },
                "response_monitoring": [
                    "response_rms",
                ],
                "intervention_dim": int(model.intervention_generator.node_dim),
                "hybrid_graph_encoder": {
                    "method": "typed_one_hop_relation_tokens_then_global_linear_attention",
                    "global_attention": "single_head_elu_kernel_linear_attention",
                    "local_aggregation": "typed_row_normalized_one_hop_mean",
                    "relation_fusion": "per_gene_two_token_cross_attention",
                    "relation_tokens": ["physical", "genetic"],
                    "relation_token_scope": "same_gene_only",
                    "gate_parameterization": "sigmoid_scalar_per_stage",
                    "gate_initialization": 0.1,
                    "residual_normalization": "layer_norm_after_relation_fusion_and_global_attention",
                    "num_directed_edges": int(model.graph_encoder.edge_index.shape[1]),
                    "evidence_names": list(model.graph_encoder.evidence_names),
                    "layers": model.graph_encoder.num_layers,
                },
                "response_parameterization": {
                    "equation": (
                        "S_ng=[mu_cg,var_cg,x_ng-mu_cg]; "
                        "H0_ng=GeneIdentityCrossAttn(S_ng,cell_context_n); "
                        "L_r=D_r^-1 A_r H0 W_r; "
                        "T_i=[L_physical_i+e_physical,L_genetic_i+e_genetic]; "
                        "X_i=LN(H0_i+gamma*CrossAttn(H0_i,T_i,T_i)); "
                        "H1=LN(H0+alpha*GlobalLinearAttn(X)); "
                        "I_t=f_structural(H1_t,technology); "
                        "I_t_star=I_t+AdditiveInteract_t({I}); "
                        "R_g=I_g+sum_t 1[g!=t] I_t_star*gain(H1_g); "
                        "delta_g=D(R_g); y_hat_g=mu_cg+delta_g"
                    ),
                    "propagation_update": (
                        "direct_intervention_broadcast_after_symmetric_additive_"
                        "intervention_interaction"
                    ),
                    "graph_features": "typed_relation_tokens_condition_global_attention_inputs",
                    "non_neighbor_and_isolated_nodes": (
                        "global_attention_connects_all_nodes_before_"
                        "background_modulated_intervention_broadcast"
                    ),
                    "multi_target_aggregation": (
                        "permutation_equivariant_additive_intervention_"
                        "interaction_then_sum"
                    ),
                    "target_interaction_graph_conditioning": (
                        "symmetric_pairwise_intervention_content_only"
                    ),
                    "target_interaction_update": "I_star=I+delta_I",
                    "post_propagation_synergy_residual": False,
                    "node_revisiting": False,
                    "edge_reuse": False,
                    "schedule": "none",
                    "runtime_recurrence": False,
                },
                "absolute_prediction": (
                    "cell_line_control_mean_plus_direct_decoded_perturbation_effect"
                ),
                "decoder": "delta=D(total_response_R)_without_background_remodulation",
            },
            "input_encoding": {
                "features": [
                    f"gene_identity_{model.gene_embedding.embedding_dim}d",
                    "cell_line_control_mean",
                    "cell_line_control_variance",
                    "paired_control_expression_offset_from_mean",
                ],
                "expression_binning": False,
                "expression_standardization": False,
                "robust_scale_estimation": False,
                "statistics_source": "cell_line_matched_control_pool",
            },
        }
        if self._optimizer is not None:
            payload["optimizer"] = self._optimizer.state_dict()
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, out)
        return out

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint state into the runner."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        architecture = payload.get("architecture")
        if architecture != GPM_ARCHITECTURE_VERSION:
            raise ValueError(
                f"Checkpoint architecture {architecture!r} is incompatible with the current architecture {GPM_ARCHITECTURE_VERSION!r}. "
                "Retrain the model to generate a compatible checkpoint."
            )
        checkpoint_genes = tuple(str(g) for g in payload.get("gene_names", ()))
        model = unwrap_compiled_model(self.model)
        model_genes = model.gene_names
        if checkpoint_genes != model_genes:
            raise ValueError("The checkpoint gene panel does not match the current gene_names.")
        checkpoint_technologies = tuple(
            str(name) for name in payload.get("technology_names", ())
        )
        if checkpoint_technologies != model.technology_names:
            raise ValueError(
                "The checkpoint perturbation technology vocabulary or ordering does not match the current experiment."
            )
        checkpoint_technology_dim = int(payload.get("technology_embedding_dim", -1))
        if checkpoint_technology_dim != model.technology_embedding.embedding_dim:
            raise ValueError("The checkpoint technology embedding dimension does not match the current configuration.")
        model.load_state_dict(payload["model"])
        model.gene_embedding_source = str(
            payload.get("gene_embedding_source", model.gene_embedding_source)
        )
        if self._optimizer is not None and "optimizer" in payload:
            self._optimizer.load_state_dict(payload["optimizer"])
        if "loss_weights" in payload:
            self.loss_weights = merge_loss_weights(payload["loss_weights"])
            self.criterion = GPMLoss(
                blur=self.loss_blur,
                sinkhorn_iters=self.loss_sinkhorn_iters,
                weights=self.loss_weights,
            )
