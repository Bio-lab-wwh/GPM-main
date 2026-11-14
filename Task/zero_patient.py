from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scanpy as sc

from GPM.gpm import GPM_ARCHITECTURE_VERSION, GPMRunner, unwrap_compiled_model
from GPM.causal_inference import resolve_biogrid_path
from GPM.paths import file_identity, manifest_artifact_path, project_path
from GPM.conditions import normalize_tech_name
from GPM.train_config import build_gpm_train_config, load_training_settings
from GPM.utils import GPMSample, count_perturbation_groups
from Tools.memory import release_memory
from Tools.utils import (
    AnnDataExpressionStore,
    build_gpm_samples,
    ctrl_pool_indices,
    split_indices,
)
from Task.common import (
    _build_samples,
    _dataset_identity,
    _effect_rows,
    _evaluate,
    _gene_embedding_identity,
    _load_control_centres,
    _read_json,
    _subset_to_fold_panel,
    _write_training_history,
    train_with_early_stopping,
)


FINETUNE_EPOCHS = 10


def _samples_for_split(
    adata: sc.AnnData,
    *,
    key: str,
    split: str,
    split_column: str,
    genes: list[str],
    seed: int,
    technology: str,
    cell_line: str,
    technology_to_id: Mapping[str, int],
    expression_store: AnnDataExpressionStore,
    control_pool: np.ndarray,
) -> list[GPMSample]:
    rows = split_indices(
        adata, split, split_column=split_column, perturb_only=True
    )
    return build_gpm_samples(
        adata,
        cell_indices=rows,
        control_pool_indices=control_pool,
        gene_names=genes,
        sample_seed=seed,
        default_tech=technology,
        default_cell_line=cell_line,
        panel_key=key,
        expression_store=expression_store,
        technology_to_id=technology_to_id,
    )


def _stage_evaluation(
    runner: GPMRunner,
    samples: Sequence[GPMSample],
    *,
    genes: list[str],
    global_mean: np.ndarray,
    by_batch: Mapping[str, np.ndarray],
    batch_size: int,
    output_dir: Path,
    stage: str,
    epoch: int,
    checkpoint: Path,
    save_npz: bool,
    training: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    runner.set_ctrl_mean(global_mean)
    outputs = runner.predict(samples, batch_size=batch_size, desc=f"zero_patient:{stage}")
    pred, truth, labels, batches, levels = _effect_rows(
        samples, outputs, by_batch, global_mean
    )
    npz_path = output_dir / f"{stage}_test.npz" if save_npz else None
    if npz_path is not None:
        np.savez_compressed(
            npz_path,
            gene_names=np.asarray(genes, dtype=str),
            perturbation=labels,
            pred_delta=pred,
            true_delta=truth,
            batch=batches,
            control_match_level=levels,
        )
    metrics = _evaluate("frangieh", labels, pred, truth, genes)
    metric_path = output_dir / f"{stage}_metrics.json"
    payload = {
        "schema_version": 1,
        "model": "GPM",
        "task": "zero_patient",
        "stage": stage,
        "finetune_epoch": epoch,
        "checkpoint": project_path(checkpoint),
        "prediction_file": project_path(npz_path) if npz_path is not None else None,
        "n_test_cells": len(samples),
        "n_test_perturbations": len(set(labels.astype(str))),
        "metrics": metrics,
        "training": dict(training),
        "protocol": {
            "patient_controls_are_independent_training_samples": False,
            "patient_controls_used_for_target_hvg_selection": True,
            "patient_controls_used_as_basal_inputs_and_effect_centres": True,
            "strict_patient_dataset_blind_zero_shot": False,
            "patient_test_disjoint_from_finetune": True,
            "seen_unseen_relative_to_pretraining_reported_together": True,
        },
    }
    metric_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    del outputs, pred, truth, labels, batches, levels
    release_memory()
    return metric_path, metrics


def _run(args: argparse.Namespace) -> Path:
    manifest = _read_json(args.manifest)
    if manifest.get("task") != "zero_patient":
        raise ValueError(f"Expected zero_patient manifest, got {manifest.get('task')!r}")
    fold = manifest["folds"][0]
    pretrain_keys = list(map(str, fold["pretrain_dataset_keys"]))
    patient_key = str(fold["patient_dataset_key"])
    genes = list(map(str, fold["gene_panel"]))
    split_column = str(fold["split_column"])
    settings = load_training_settings(args.train_config)
    seed = int(settings["seed"])
    techs = sorted({
        normalize_tech_name(str(manifest["datasets"][key]["technology"]))
        for key in fold["dataset_keys"]
    })
    config = build_gpm_train_config(
        config_path=args.train_config, technology_names=techs
    )
    technology_to_id = {
        name: index for index, name in enumerate(config.technology_names)
    }

    datasets: dict[str, sc.AnnData] = {}
    source_train: list[GPMSample] = []
    source_val: list[GPMSample] = []
    source_control_means: dict[str, np.ndarray] = {}
    patient_meta = manifest["datasets"][patient_key]
    patient_path = manifest_artifact_path(patient_meta["h5ad"], args.manifest)
    patient = _subset_to_fold_panel(
        sc.read_h5ad(patient_path, backed="r"), genes,
        label="frangieh/zero_patient",
    )
    datasets[patient_key] = patient
    patient_store = AnnDataExpressionStore(patient)
    patient_control_pool = ctrl_pool_indices(patient)
    test_samples = _samples_for_split(
        patient,
        key=patient_key,
        split=str(fold.get("patient_test_split", "test")),
        split_column=split_column,
        genes=genes,
        seed=seed + 3,
        technology=str(patient_meta["technology"]),
        cell_line=str(patient_meta["cell_line"]),
        technology_to_id=technology_to_id,
        expression_store=patient_store,
        control_pool=patient_control_pool,
    )
    if not test_samples:
        raise ValueError("zero_patient has an empty test sample set")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    global_mean, by_batch = _load_control_centres(patient_meta, args.manifest, genes)
    if args.checkpoint is not None:
        pretrained = args.checkpoint
        archive_stem = pretrained.stem.removesuffix("_pretrained")
        finetuned = pretrained.with_name(f"{archive_stem}_finetuned_epoch10.pt")
        runner = GPMRunner.from_config(config, gene_names=genes)
        archive_metrics = {}
        for stage, epoch, checkpoint in (
            ("zero_shot", 0, pretrained),
            ("finetune_epoch10", FINETUNE_EPOCHS, finetuned),
        ):
            runner.load_checkpoint(checkpoint)
            path, _ = _stage_evaluation(
                runner, test_samples, genes=genes, global_mean=global_mean, by_batch=by_batch,
                batch_size=config.batch_size, output_dir=args.model_dir,
                stage=f"{archive_stem}_{stage}", epoch=epoch, checkpoint=checkpoint,
                save_npz=True, training={"loaded_existing_checkpoint": True},
            )
            archive_metrics[stage] = project_path(path)
        replay_summary = args.model_dir / f"{archive_stem}_metrics.json"
        replay_summary.write_text(json.dumps(archive_metrics, indent=2), encoding="utf-8")
        for data in datasets.values():
            data.file.close()
        return replay_summary

    for key in pretrain_keys:
        meta = manifest["datasets"][key]
        path = manifest_artifact_path(meta["h5ad"], args.manifest)
        data = _subset_to_fold_panel(
            sc.read_h5ad(path, backed="r"), genes,
            label=f"{key}/zero_patient",
        )
        datasets[key] = data
        train, val, _ = _build_samples(
            data,
            key=key,
            split_column=split_column,
            gene_names=genes,
            seed=seed,
            technology=str(meta["technology"]),
            cell_line=str(meta["cell_line"]),
            technology_to_id=technology_to_id,
        )
        source_train.extend(train)
        source_val.extend(val)
        source_control_means[key] = _load_control_centres(meta, args.manifest, genes)[0]

    finetune_samples = _samples_for_split(
        patient,
        key=patient_key,
        split=str(fold.get("patient_finetune_split", "finetune")),
        split_column=split_column,
        genes=genes,
        seed=seed + 2,
        technology=str(patient_meta["technology"]),
        cell_line=str(patient_meta["cell_line"]),
        technology_to_id=technology_to_id,
        expression_store=patient_store,
        control_pool=patient_control_pool,
    )
    if not source_train or not source_val or not finetune_samples:
        raise ValueError("zero_patient has an empty train/val/fine-tune/test sample set")
    if {
        sample.condition for sample in finetune_samples
    } & {sample.condition for sample in test_samples}:
        raise RuntimeError("Frangieh fine-tune/test conditions overlap")

    identity = hashlib.sha256(json.dumps({
        "architecture": GPM_ARCHITECTURE_VERSION,
        "datasets": _dataset_identity(manifest, args.manifest, fold["dataset_keys"]),
        "technology_names": techs,
        "fold": {key: value for key, value in fold.items() if key != "report"},
        "settings": {key: value for key, value in settings.items() if key not in {"biogrid_path", "gene_embedding_path"}},
        "biogrid": file_identity(resolve_biogrid_path(config.biogrid_path)),
        "early_stopping": {
            "min_groups": int(config.min_groups),
            "patience": int(config.patience),
        },
        "gene_embedding": _gene_embedding_identity(config.gene_embedding_path),
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
    stem = f"zero_patient_{fold['name']}_seed{seed}_{identity}".replace(" ", "_")
    pretrain_checkpoint = args.model_dir / f"{stem}_pretrained.pt"
    finetuned_checkpoint = args.model_dir / f"{stem}_finetuned_epoch10.pt"
    summary_path = args.model_dir / f"{stem}_metrics.json"

    expected_metrics = [args.model_dir / f"{stem}_zero_shot_metrics.json"] + [
        args.model_dir / f"{stem}_finetune_epoch{epoch:02d}_metrics.json"
        for epoch in range(1, FINETUNE_EPOCHS + 1)
    ]
    complete = (
        summary_path.is_file()
        and
        pretrain_checkpoint.is_file()
        and finetuned_checkpoint.is_file()
        and all(path.is_file() for path in expected_metrics)
        and (args.model_dir / f"{stem}_zero_shot_test.npz").is_file()
        and (args.model_dir / f"{stem}_finetune_epoch10_test.npz").is_file()
    )
    if complete and not args.force_train and not args.force_predict:
        for data in datasets.values():
            if getattr(data, "file", None) is not None:
                data.file.close()
        return summary_path

    runner = GPMRunner.from_config(config, gene_names=genes)
    if pretrain_checkpoint.is_file() and not args.force_train:
        runner.load_checkpoint(pretrain_checkpoint)
        release_memory()
        pretraining = {"loaded_existing_checkpoint": True}
    else:
        release_memory()
        pretraining = train_with_early_stopping(
            runner,
            source_train,
            source_val,
            config,
            pretrain_checkpoint,
            validation_control_means=source_control_means,
        )
    zero_stage = f"{stem}_zero_shot"
    zero_metric, zero_metrics = _stage_evaluation(
        runner,
        test_samples,
        genes=genes,
        global_mean=global_mean,
        by_batch=by_batch,
        batch_size=config.batch_size,
        output_dir=args.model_dir,
        stage=zero_stage,
        epoch=0,
        checkpoint=pretrain_checkpoint,
        save_npz=True,
        training=pretraining,
    )

    # Always reconstruct the ten-epoch trajectory from the immutable pretrained
    runner.load_checkpoint(pretrain_checkpoint)
    unwrap_compiled_model(runner.model).set_gene_panel(genes)
    if count_perturbation_groups(finetune_samples, config.pert_group_size) == 0:
        raise RuntimeError("No complete Frangieh fine-tuning perturbation group")
    history_rows: list[dict[str, Any]] = []
    metric_files = [zero_metric]
    stage_metrics: dict[str, Any] = {"zero_shot": zero_metrics}
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        lr = runner.current_lr()
        train_metric = runner._run_epoch_on_samples(
            finetune_samples,
            train=True,
            group_size=config.pert_group_size,
            micro_batch_size=config.batch_size,
            seed=config.seed + 20_000,
            print_every=config.print_every_n_groups,
            epoch=epoch,
            split_name="finetune",
        )
        print(
            f"[GPM fine-tune] epoch={epoch} loss={train_metric.loss_total:.6f} "
            f"mse={train_metric.loss_mse_expression:.6f} "
            f"sinkhorn={train_metric.loss_sinkhorn:.6f}",
            flush=True,
        )
        stage = f"{stem}_finetune_epoch{epoch:02d}"
        metric_path, metrics = _stage_evaluation(
            runner,
            test_samples,
            genes=genes,
            global_mean=global_mean,
            by_batch=by_batch,
            batch_size=config.batch_size,
            output_dir=args.model_dir,
            stage=stage,
            epoch=epoch,
            checkpoint=finetuned_checkpoint if epoch == FINETUNE_EPOCHS else pretrain_checkpoint,
            save_npz=epoch == FINETUNE_EPOCHS,
            training={"finetune_epoch": epoch, "train_loss_total": train_metric.loss_total},
        )
        metric_files.append(metric_path)
        stage_metrics[f"finetune_epoch_{epoch}"] = metrics
        history_rows.append({
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss_total": train_metric.loss_total,
            "train_loss_mse_expression": train_metric.loss_mse_expression,
            "train_loss_sinkhorn": train_metric.loss_sinkhorn,
            "val_mean_perturbation_effect_pcc": "",
            "selection_score": "",
            "selection_metric": "not_applicable_fixed_10_epoch_finetune",
            "is_best": "",
            "stale_epochs": "",
            "train_groups_this_epoch": train_metric.n_groups,
            "completed_train_groups": "",
            "min_groups": "",
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
        _write_training_history(
            finetuned_checkpoint.with_name(
                f"{finetuned_checkpoint.stem}_training_history.csv"
            ),
            history_rows,
        )
        if epoch < FINETUNE_EPOCHS:
            runner.step_lr_scheduler()
    runner.save_checkpoint(
        finetuned_checkpoint,
        extra={"finetune_epochs": FINETUNE_EPOCHS, "base_checkpoint": project_path(pretrain_checkpoint)},
    )

    summary = {
        "schema_version": 1,
        "model": "GPM",
        "task": "zero_patient",
        "fold": fold["name"],
        "manifest": project_path(args.manifest),
        "pretrained_checkpoint": project_path(pretrain_checkpoint),
        "finetuned_epoch10_checkpoint": project_path(finetuned_checkpoint),
        "metric_files": [project_path(path) for path in metric_files],
        "prediction_files": [
            project_path(args.model_dir / f"{zero_stage}_test.npz"),
            project_path(args.model_dir / f"{stem}_finetune_epoch10_test.npz"),
        ],
        "metrics": stage_metrics,
        "fine_tune_epochs": FINETUNE_EPOCHS,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    for data in datasets.values():
        if getattr(data, "file", None) is not None:
            data.file.close()
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPM zero_patient experiment")
    parser.add_argument("--manifest",type=Path,default=Path(__file__).parents[1] / "Division" / "outputs" / "zero_patient" / "manifest.json",)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Replay archived zero-shot and epoch-10 checkpoints.")
    parser.add_argument("--force-predict", action="store_true")
    args = parser.parse_args()
    if args.model_dir is None:
        output_name = "replay" if args.checkpoint is not None else "model"
        args.model_dir = Path(__file__).parent / output_name / "zero_patient"
    if args.checkpoint is not None:
        if args.force_train or not args.checkpoint.name.endswith("_pretrained.pt"):
            parser.error("--checkpoint requires an archived *_pretrained.pt and cannot be used with --force-train")
        finetuned = args.checkpoint.with_name(
            f"{args.checkpoint.stem.removesuffix('_pretrained')}_finetuned_epoch10.pt"
        )
        for checkpoint in (args.checkpoint, finetuned):
            if not checkpoint.is_file():
                parser.error(f"Checkpoint not found: {checkpoint}")
    result = _run(args)
    print(json.dumps({"task": "zero_patient", "metrics": project_path(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
