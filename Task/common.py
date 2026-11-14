from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import scanpy as sc

from Data.registry import DATASET_DISPLAY_NAMES, resolve_single_dataset_keys
from GPM.gpm import GPM_ARCHITECTURE_VERSION, GPMRunner, unwrap_compiled_model
from GPM.model import resolve_gene_embedding_path
from GPM.paths import file_identity, manifest_artifact_path, project_path
from GPM.causal_inference import resolve_biogrid_path
from GPM.conditions import normalize_tech_name
from GPM.train_config import build_gpm_train_config, load_training_settings
from GPM.utils import (
    GPMSample,
    count_perturbation_groups,
    materialize_sample_expression,
)
from Tools.evaluation import evaluate_perturbation_effects, evaluation_report_to_dict
from Tools.utils import (
    AnnDataExpressionStore,
    build_gpm_samples,
    ctrl_pool_indices,
    panel_gene_names,
    split_indices,
)
from Tools.memory import ensure_memory_headroom, memory_status, release_memory


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_TRAINING_HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss_total",
    "train_loss_mse_expression",
    "train_loss_sinkhorn",
    "val_mean_perturbation_effect_pcc",
    "selection_score",
    "selection_metric",
    "checkpoint_selection_eligible",
    "is_best",
    "early_stopping_active",
    "stale_epochs",
    "train_groups_this_epoch",
    "completed_train_groups",
    "min_groups",
    "train_n_samples",
    "train_n_response_samples",
    "train_response_rms",
    "train_n_gradient_samples",
    "train_grad_context_encoder",
    "train_grad_technology",
    "train_grad_node_encoder",
    "train_grad_graph_encoder",
    "train_grad_intervention_generator",
    "train_grad_propagator",
    "train_grad_decoder",
)

_EARLY_STOPPING_METRIC = "mean_dataset_condition_perturbation_effect_pcc"


def _training_history_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}_training_history.csv")


def _write_training_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TRAINING_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _gene_embedding_identity(path: Optional[str | Path]) -> dict[str, Any]:
    """Fingerprint the gene prior so checkpoints are tied to their initialization."""
    source = resolve_gene_embedding_path(path).resolve()
    if not source.is_file():
        return {"status": "missing_random_fallback"}
    return {"status": "pretrained", **file_identity(source)}


def _dataset_identity(
    manifest: Mapping[str, Any], manifest_path: Path, keys: Sequence[str],
) -> dict[str, Any]:
    identity = {}
    for key in keys:
        metadata = manifest["datasets"][key]
        identity[key] = {
            field: metadata.get(field) for field in ("dataset_id", "cell_line", "technology")
        } | {
            field: file_identity(manifest_artifact_path(metadata[field], manifest_path))
            for field in ("h5ad", "control_means")
        }
    return identity


def _load_control_centres(
    metadata: Mapping[str, Any], manifest_path: Path, gene_names: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = manifest_artifact_path(metadata["control_means"], manifest_path)
    with np.load(path, allow_pickle=False) as saved:
        stored_genes = np.asarray(saved["gene_names"]).astype(str).tolist()
        lookup = {gene.upper(): index for index, gene in enumerate(stored_genes)}
        if len(lookup) != len(stored_genes):
            raise ValueError(f"{path}: control-mean gene panel contains duplicates")
        missing = [gene for gene in gene_names if str(gene).upper() not in lookup]
        if missing:
            raise ValueError(
                f"{path}: control means miss {len(missing)} fold genes: {missing[:5]}"
            )
        indices = np.asarray([lookup[str(gene).upper()] for gene in gene_names], dtype=np.int64)
        global_mean = np.asarray(saved["global_mean"], dtype=np.float32)[indices]
        batch_names = np.asarray(saved["batch_names"]).astype(str)
        by_batch = {
            str(batch): np.asarray(saved[f"batch_mean_{i}"], dtype=np.float32)[indices]
            for i, batch in enumerate(batch_names)
        }
    return global_mean, by_batch


def _subset_to_fold_panel(
    data: sc.AnnData, gene_names: Sequence[str], *, label: str
) -> sc.AnnData:
    """Return a backed view containing only the current fold's ordered genes."""
    observed = panel_gene_names(data)
    requested = [str(gene) for gene in gene_names]
    if observed == requested:
        return data
    lookup = {gene.upper(): index for index, gene in enumerate(observed)}
    if len(lookup) != len(observed):
        data.file.close()
        raise ValueError(f"{label}: stored H5AD gene panel contains duplicates")
    missing = [gene for gene in requested if gene.upper() not in lookup]
    if missing:
        data.file.close()
        raise ValueError(
            f"{label}: stored H5AD misses {len(missing)} fold genes: {missing[:5]}"
        )
    return data[:, [lookup[gene.upper()] for gene in requested]]


def _build_samples(
    adata: sc.AnnData,
    *,
    key: str,
    split_column: str,
    gene_names: list[str],
    seed: int,
    technology: str,
    cell_line: str,
    technology_to_id: Mapping[str, int],
    test_only: bool = False,
) -> tuple[
    list[GPMSample],
    list[GPMSample],
    list[GPMSample],
]:
    control_pool = ctrl_pool_indices(adata)
    expression_store = AnnDataExpressionStore(adata)
    result: list[list[GPMSample]] = []
    for split in ("train", "val", "test"):
        if test_only and split != "test":
            result.append([])
            continue
        indices = split_indices(
            adata,
            split,
            split_column=split_column,
            # Sample basal inputs for perturbed cells from the relevant control pool.
            perturb_only=True,
        )
        print(
            f"[samples] {key}/{split}: using all {indices.size} processed cells "
            "from the shared CPU expression cache",
            flush=True,
        )
        result.append(build_gpm_samples(
            adata,
            cell_indices=indices,
            control_pool_indices=control_pool,
            gene_names=gene_names,
            sample_seed=seed,
            default_tech=technology,
            default_cell_line=cell_line,
            panel_key=key,
            expression_store=expression_store,
            technology_to_id=technology_to_id,
        ))
    return result[0], result[1], result[2]


def train_with_early_stopping(
    runner: GPMRunner,
    train_samples: Sequence[GPMSample],
    val_samples: Sequence[GPMSample],
    config: Any,
    checkpoint: Path,
    validation_control_means: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    groups_per_epoch = count_perturbation_groups(
        train_samples, config.pert_group_size
    )
    if groups_per_epoch == 0:
        raise RuntimeError(
            f"No complete training perturbation group (group size={config.pert_group_size})"
        )
    unwrap_compiled_model(runner.model).set_gene_panel(list(train_samples[0].gene_names))
    if not val_samples:
        raise RuntimeError("Checkpoint selection by perturbation effect Pearson correlation requires a nonempty validation set.")
    validation_panels = {sample.resolved_panel_key() for sample in val_samples}
    missing_control_means = sorted(validation_panels - set(validation_control_means))
    if missing_control_means:
        raise KeyError(f"Saved control means are missing for validation datasets: {missing_control_means}")

    best_pcc = float("-inf")
    best_epoch = 0
    stale = 0
    completed_groups = 0
    min_groups = max(0, int(config.min_groups))
    max_train_groups = groups_per_epoch * int(config.epochs)
    selection_epoch = (
        1 if min_groups == 0 else (min_groups + groups_per_epoch - 1) // groups_per_epoch
    )
    print(
        f"[early stopping] Perturbation groups per epoch: {groups_per_epoch}. "
        f"Checkpoint selection starts at min_groups={min_groups} "
        f"(expected at epoch {selection_epoch}). Patience is counted from the following validation epoch. "
        f"patience={int(config.patience)}.",
        flush=True,
    )
    if max_train_groups < min_groups:
        print(
            f"[early stopping] The scheduled run can process at most {max_train_groups} groups, below the checkpoint selection threshold. "
            "Training will complete all scheduled epochs and save the final checkpoint.",
            flush=True,
        )
    history_path = _training_history_path(checkpoint)
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(config.epochs) + 1):
        learning_rate = runner.current_lr()
        train_metric = runner._run_epoch_on_samples(
            train_samples,
            train=True,
            group_size=config.pert_group_size,
            micro_batch_size=config.batch_size,
            seed=config.seed,
            print_every=config.print_every_n_groups,
            epoch=epoch,
            split_name="train",
        )
        completed_groups += int(train_metric.n_groups)
        print(
            f"[training groups] epoch={epoch} completed_this_epoch="
            f"{train_metric.n_groups} cumulative={completed_groups}/{min_groups}",
            flush=True,
        )
        score = runner.condition_mean_effect_pcc(
            val_samples,
            control_means_by_panel=validation_control_means,
            batch_size=config.batch_size,
            desc=f"val epoch {epoch} perturbation effect",
        )
        # Start checkpoint selection and patience tracking after the minimum group count.
        checkpoint_selection_eligible = completed_groups >= min_groups
        had_eligible_checkpoint = best_epoch > 0
        is_best = checkpoint_selection_eligible and score > best_pcc
        early_stopping_active = checkpoint_selection_eligible and had_eligible_checkpoint
        if is_best:
            best_pcc, best_epoch, stale = score, epoch, 0
            runner.save_checkpoint(checkpoint, extra={
                "epoch": epoch,
                "selection_score": score,
                "selection_metric": _EARLY_STOPPING_METRIC,
                "selection_mode": "max",
                "min_groups": min_groups,
                "completed_train_groups": completed_groups,
                "training_history_csv": project_path(history_path),
            })
        elif early_stopping_active:
            stale += 1
        else:
            stale = 0
        history_rows.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss_total": train_metric.loss_total,
            "train_loss_mse_expression": train_metric.loss_mse_expression,
            "train_loss_sinkhorn": train_metric.loss_sinkhorn,
            "val_mean_perturbation_effect_pcc": score,
            "selection_score": score,
            "selection_metric": _EARLY_STOPPING_METRIC,
            "checkpoint_selection_eligible": checkpoint_selection_eligible,
            "is_best": is_best,
            "early_stopping_active": early_stopping_active,
            "stale_epochs": stale,
            "train_groups_this_epoch": train_metric.n_groups,
            "completed_train_groups": completed_groups,
            "min_groups": min_groups,
            "train_n_samples": train_metric.n_samples,
            "train_n_response_samples": train_metric.n_response_samples,
            "train_response_rms": train_metric.response_rms,
            "train_n_gradient_samples": train_metric.n_gradient_samples,
            "train_grad_context_encoder": train_metric.grad_context_encoder,
            "train_grad_technology": train_metric.grad_technology,
            "train_grad_node_encoder": train_metric.grad_node_encoder,
            "train_grad_graph_encoder": train_metric.grad_graph_encoder,
            "train_grad_intervention_generator": (
                train_metric.grad_intervention_generator
            ),
            "train_grad_propagator": train_metric.grad_propagator,
            "train_grad_decoder": train_metric.grad_decoder,
        })
        _write_training_history(history_path, history_rows)
        if early_stopping_active and not is_best:
            if stale >= int(config.patience):
                break
        if epoch < int(config.epochs):
            runner.step_lr_scheduler()
    threshold_reached = completed_groups >= min_groups
    if best_epoch == 0:
        # Save the final model if training ends before checkpoint selection begins.
        best_epoch, best_pcc = epoch, score
        runner.save_checkpoint(checkpoint, extra={
            "epoch": epoch,
            "selection_score": score,
            "selection_metric": _EARLY_STOPPING_METRIC,
            "selection_mode": "final_epoch_below_min_groups",
            "min_groups": min_groups,
            "completed_train_groups": completed_groups,
            "training_history_csv": project_path(history_path),
        })
    runner.load_checkpoint(checkpoint)
    return {
        "best_epoch": best_epoch,
        "best_pcc": best_pcc,
        "selection_metric": _EARLY_STOPPING_METRIC,
        "selection_mode": (
            "max" if threshold_reached else "final_epoch_below_min_groups"
        ),
        "min_groups": min_groups,
        "completed_train_groups": completed_groups,
        "groups_per_epoch": groups_per_epoch,
        "checkpoint_selection_started_at_epoch": (
            selection_epoch if threshold_reached else None
        ),
        "early_stopping_started_after_groups": min_groups,
        "stopped_after": epoch,
        "training_history_csv": project_path(history_path),
    }


def _effect_rows(
    samples: Sequence[GPMSample], predictions: Sequence[Any], centres: Mapping[str, np.ndarray],
    global_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted, truth, labels, batches, match_levels = [], [], [], [], []
    target_rows = materialize_sample_expression(samples, target=True)
    for sample, output, target_row in zip(samples, predictions, target_rows):
        batch = str(sample.batch_id or "batch0")
        centre = centres.get(batch, global_mean)
        predicted.append(np.asarray(output.pred_expression, dtype=np.float32) - centre)
        truth.append(target_row - centre)
        labels.append(str(sample.condition))
        batches.append(batch)
        match_levels.append(str(sample.control_match_level or "unknown"))
    return (
        np.stack(predicted), np.stack(truth), np.asarray(labels, dtype=str),
        np.asarray(batches, dtype=str), np.asarray(match_levels, dtype=str),
    )


def _aggregate(labels: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        condition: np.mean(rows[labels == condition], axis=0).astype(np.float32)
        for condition in sorted(set(labels.astype(str))) if condition != "ctrl"
    }


def _evaluate(
    name: str, labels: np.ndarray, predicted: np.ndarray, truth: np.ndarray, genes: list[str]
) -> dict[str, Any]:
    pred_map, truth_map = _aggregate(labels, predicted), _aggregate(labels, truth)
    common = sorted(set(pred_map) & set(truth_map))
    report = evaluate_perturbation_effects(
        name, {key: pred_map[key] for key in common}, {key: truth_map[key] for key in common}, genes
    )
    return evaluation_report_to_dict(report)


def train_fold(
    manifest: Mapping[str, Any], fold: Mapping[str, Any], *, manifest_path: Path,
    model_dir: Path, train_config: Optional[Path], force_train: bool, force_predict: bool,
    archived_checkpoint: Optional[Path] = None,
) -> Path:
    task = str(manifest["task"])
    fold_name = str(fold["name"])
    if fold.get("status") == "skipped":
        raise ValueError(f"Fold {fold_name} is skipped: {fold.get('reason')}")
    settings = load_training_settings(train_config)
    seed = int(settings["seed"])
    genes = [str(g) for g in fold["gene_panel"]]
    dataset_keys = list(fold["dataset_keys"])
    techs = sorted({
        normalize_tech_name(str(manifest["datasets"][key]["technology"]))
        for key in dataset_keys
    })
    config = build_gpm_train_config(
        config_path=train_config, technology_names=techs
    )
    technology_to_id = {
        name: idx for idx, name in enumerate(config.technology_names)
    }
    runner = GPMRunner.from_config(config, gene_names=genes)
    datasets: dict[str, sc.AnnData] = {}
    split_samples: dict[str, tuple[list[GPMSample], list[GPMSample], list[GPMSample]]] = {}
    controls: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    evaluation_groups: dict[str, dict[str, str]] = {}
    for key in dataset_keys:
        metadata = manifest["datasets"][key]
        path = manifest_artifact_path(metadata["h5ad"], manifest_path)
        ensure_memory_headroom(
            f"before loading training dataset {key}", config.min_free_memory_gib
        )
        data = _subset_to_fold_panel(
            sc.read_h5ad(path, backed="r"), genes,
            label=f"{key}/{fold_name}",
        )
        datasets[key] = data
        controls[key] = _load_control_centres(metadata, manifest_path, genes)
        if "evaluation_group" in data.obs:
            evaluation_groups[key] = data.obs["evaluation_group"].astype(str).to_dict()
        train_part, val_part, test_part = _build_samples(
            data,
            key=key,
            split_column=str(fold["split_column"]),
            gene_names=genes,
            seed=seed,
            technology=str(metadata["technology"]),
            cell_line=str(metadata["cell_line"]),
            technology_to_id=technology_to_id,
            test_only=archived_checkpoint is not None,
        )
        split_samples[key] = (train_part, val_part, test_part)
        release_memory()
        print(memory_status(f"indexed batched samples for {key}"), flush=True)

    train_samples = [sample for key in dataset_keys for sample in split_samples[key][0]]
    val_samples = [sample for key in dataset_keys for sample in split_samples[key][1]]

    fold_identity = {
        key: value for key, value in fold.items() if key != "report"
    }
    split_identity = {
        key: {
            split: fold.get("report", {}).get(key, {}).get(
                "split_perturbations", {}
            ).get(split, [])
            for split in ("train", "val", "test")
        }
        for key in dataset_keys
    }
    dataset_identity = _dataset_identity(manifest, manifest_path, dataset_keys) if archived_checkpoint is None else {}
    if archived_checkpoint is None:
        biogrid_source = resolve_biogrid_path(config.biogrid_path)
        fingerprint_payload = {
            "architecture": GPM_ARCHITECTURE_VERSION,
            "task": task,
            "fold": fold_identity,
            "split_perturbations": split_identity,
            "datasets": dataset_identity,
            "settings": {key: value for key, value in settings.items() if key not in {"biogrid_path", "gene_embedding_path"}},
            # Include the early-stopping threshold in the checkpoint fingerprint.
            "early_stopping": {
                "min_groups": int(config.min_groups),
                "patience": int(config.patience),
            },
            "biogrid": file_identity(biogrid_source),
            "gene_embedding": _gene_embedding_identity(config.gene_embedding_path),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()[:10]
        stem = f"{task}_{fold_name}_seed{seed}_{fingerprint}".replace("/", "_").replace(" ", "_")
    if archived_checkpoint is not None:
        checkpoint = archived_checkpoint
        stem = checkpoint.stem.removesuffix("_best")
        fingerprint = stem.rsplit("_", 1)[-1]
    else:
        checkpoint = model_dir / f"{stem}_best.pt"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / f"{stem}_metrics.json"
    history_path = _training_history_path(checkpoint)
    if archived_checkpoint is not None or (checkpoint.is_file() and not force_train):
        runner.load_checkpoint(checkpoint)
        release_memory()
        training = {
            "loaded_existing_checkpoint": True,
            "training_history_csv": (
                project_path(history_path) if history_path.is_file() else None
            ),
        }
        trained_now = False
    else:
        release_memory()
        training = train_with_early_stopping(
            runner,
            train_samples,
            val_samples,
            config,
            checkpoint,
            validation_control_means={
                key: controls[key][0]
                for key in sorted({sample.resolved_panel_key() for sample in val_samples})
            },
        )
        trained_now = True

    metrics: dict[str, Any] = {}
    match_counts: dict[str, Any] = {
        "train": {
            level: sum(1 for sample in train_samples if sample.control_match_level == level)
            for level in sorted({str(sample.control_match_level) for sample in train_samples})
        },
        "val": {
            level: sum(1 for sample in val_samples if sample.control_match_level == level)
            for level in sorted({str(sample.control_match_level) for sample in val_samples})
        },
        "test_by_dataset": {},
    }
    for key in dataset_keys:
        test_samples = split_samples[key][2]
        if not test_samples:
            continue
        npz_path = model_dir / f"{stem}_{key}_test.npz"
        cached_predictions = checkpoint.parent / npz_path.name if archived_checkpoint else npz_path
        if cached_predictions.is_file() and not force_predict and not trained_now:
            with np.load(cached_predictions, allow_pickle=False) as saved:
                pred_delta = np.asarray(saved["pred_delta"])
                true_delta = np.asarray(saved["true_delta"])
                labels = np.asarray(saved["perturbation"]).astype(str)
                batches = np.asarray(saved["batch"]).astype(str)
                levels = np.asarray(saved["control_match_level"]).astype(str)
        else:
            global_mean, by_batch = controls[key]
            runner.set_ctrl_mean(global_mean)
            outputs = runner.predict(test_samples, batch_size=config.batch_size, desc=f"{fold_name}:{key}")
            pred_delta, true_delta, labels, batches, levels = _effect_rows(
                test_samples, outputs, by_batch, global_mean
            )
            np.savez_compressed(
                npz_path,
                gene_names=np.asarray(genes, dtype=str), perturbation=labels,
                pred_delta=pred_delta, true_delta=true_delta, batch=batches,
                control_match_level=levels,
            )
        metrics[key] = _evaluate(key, labels, pred_delta, true_delta, genes)
        match_counts["test_by_dataset"][key] = {
            str(level): int((levels == level).sum()) for level in sorted(set(levels))
        }
        if task == "double_unseen_pert":
            group_by_id = evaluation_groups.get(key, {})
            sample_groups = np.asarray([
                group_by_id.get(sample.sample_id, "unknown") for sample in test_samples
            ])
            metrics[key]["groups"] = {}
            for group in fold["evaluation_groups"]:
                mask = sample_groups == group
                if mask.any():
                    metrics[key]["groups"][group] = _evaluate(
                        f"{key}:{group}", labels[mask], pred_delta[mask], true_delta[mask], genes
                    )
        if "outputs" in locals():
            del outputs
        del pred_delta, true_delta, labels, batches, levels, test_samples
        release_memory()

    payload = {
        "task": task, "fold": fold_name, "seed": seed,
        "run_fingerprint": fingerprint,
        "checkpoint": project_path(checkpoint), "training": training,
        "metrics": metrics, "control_matching": match_counts,
        "memory_safety": {
            "sample_policy": "all processed samples; no train/validation/test cap",
            "expression_loading": (
                "CPU-resident float32 expression cache with double-buffer prefetch"
            ),
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "test_samples_by_dataset": {
                key: len(split_samples[key][2]) for key in dataset_keys
            },
        },
        "effect_definition": "expression minus same-dataset same-batch control mean; dataset mean fallback",
        "de_selection": "per-condition abs(mean(true_delta)); independent for combination perturbations",
        "manifest": project_path(manifest_path),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    for data in datasets.values():
        if getattr(data, "file", None) is not None:
            data.file.close()
    del datasets, split_samples, train_samples, val_samples, controls, runner
    release_memory()
    return metrics_path


def run_cli(
    task: str,
    *,
    allow_dataset_selection: bool = False,
    default_datasets: Sequence[str] = (),
) -> None:
    parser = argparse.ArgumentParser(description=f"Train GPM task: {task}")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "Division" / "outputs" / task / "manifest.json")
    parser.add_argument("--fold", action="append", default=[])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Use a saved checkpoint for one selected fold.")
    parser.add_argument("--force-predict", action="store_true")
    if allow_dataset_selection:
        parser.add_argument("datasets",nargs="*",help="Dataset names or keys to train; names with spaces/parentheses must be quoted",)
        parser.add_argument("--dataset",dest="dataset_options",action="append",default=[],help="Repeatable alternative to positional dataset names",)
    args = parser.parse_args()
    if args.model_dir is None:
        output_name = "replay" if args.checkpoint is not None else "model"
        args.model_dir = Path(__file__).parent / output_name / task
    manifest = _read_json(args.manifest)
    if manifest.get("task") != task:
        raise ValueError(f"Manifest task is {manifest.get('task')!r}, expected {task!r}")
    selected = set(args.fold)
    if allow_dataset_selection:
        requested = [*args.datasets, *args.dataset_options]
        if not requested:
            requested = list(default_datasets)
        if requested:
            requested_keys = resolve_single_dataset_keys(requested)
            selected.update(requested_keys)
            print("Single datasets: " + ", ".join(
                DATASET_DISPLAY_NAMES[key] for key in requested_keys
            ))
    available = {str(fold["name"]) for fold in manifest["folds"]}
    missing = selected - available
    if missing:
        raise ValueError(
            f"Datasets/folds absent from manifest: {sorted(missing)}; available={sorted(available)}"
        )
    if args.checkpoint is not None:
        if args.force_train or len(selected) != 1:
            parser.error("--checkpoint requires one selected fold and cannot be used with --force-train")
        if not args.checkpoint.is_file():
            parser.error(f"Checkpoint not found: {args.checkpoint}")
    outputs = []
    for fold in manifest["folds"]:
        if selected and fold["name"] not in selected:
            continue
        if fold.get("status") == "skipped":
            print(f"[skip] {fold['name']}: {fold.get('reason')}")
            continue
        outputs.append(project_path(train_fold(
            manifest, fold, manifest_path=args.manifest, model_dir=args.model_dir,
            train_config=args.train_config, force_train=args.force_train,
            force_predict=args.force_predict, archived_checkpoint=args.checkpoint,
        )))
    print(json.dumps({"task": task, "metrics": outputs}, ensure_ascii=False, indent=2))
