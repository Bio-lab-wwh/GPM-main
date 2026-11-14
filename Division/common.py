from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from typing import Any, Iterable, Mapping, Optional, Sequence
import anndata as ad
import numpy as np

from Data.annotation import default_gene2go_path
from GPM.paths import manifest_artifact_path, project_path
from Data.preprocessing import (
    CONTROL,
    DEFAULT_MAX_CELLS_PER_PERTURBATION,
    DEFAULT_MAX_CONTROL_CELLS,
    PreprocessConfig,
    all_perturbation_targets,
    build_feature_panel,
    condition_targets,
    prepare_base_dataset,
    single_conditions,
    subset_to_panel,
    write_dataset_artifacts,
    write_json,
)
from Data.registry import (
    DATASET_DISPLAY_NAMES,
    SINGLE_DATASET_KEYS,
    load_dataset,
    resolve_single_dataset_keys,
)
from Tools.memory import ensure_memory_headroom, memory_status, release_memory

TASK_DATASETS = {
    "single_unseen_pert": SINGLE_DATASET_KEYS,
    "double_unseen_pert": ("norman",),
    "unseen_cell": ("replogle_rpe1", "replogle_k562", "nadig_hepg2", "nadig_jurkat"),
    "unseen_pert_cell": (
        "replogle_rpe1", "replogle_k562", "nadig_hepg2", "nadig_jurkat",
        "tian_crispri", "xu",
    ),
    "across_tech": ("tian_crispri", "tian_crispra"),
    "zero_patient": (
        "dixit", "xu", "yao", "tian_crispri", "tian_crispra", "norman",
        "frangieh",
    ),
}


def parse_sources(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must be DATASET=PATH, got {value!r}")
        key, path = value.split("=", 1)
        result[key.strip()] = Path(path).expanduser()
    return result


def load_base_datasets(
    keys: Sequence[str],
    sources: Mapping[str, Path],
    config: PreprocessConfig,
) -> tuple[dict[str, ad.AnnData], dict[str, Any], dict[str, Any]]:
    datasets: dict[str, ad.AnnData] = {}
    specs: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for key in keys:
        ensure_memory_headroom(
            f"before loading {key}", config.min_free_memory_gib
        )
        raw, spec = load_dataset(key, sources.get(key), biogrid_path=config.biogrid_path)
        data, report = prepare_base_dataset(raw, spec, config)
        datasets[key] = data
        reports[key] = report
        specs[key] = spec
        del raw
        release_memory()

        targets = all_perturbation_targets(data)
        targets_in_panel = int(report.get("n_targets_in_panel", 0))
        print(
            memory_status(
                f"prepared {key} "
                f"({data.n_obs} cells, {data.n_vars} measured genes, "
                f"{len(single_conditions(data))} single perturbations, "
                f"targets_in_own_panel={targets_in_panel}/{len(targets)})"
            ),
            flush=True,
        )
    return datasets, specs, reports

def split_items(
    items: Iterable[str], *, seed: int, train_fraction: float = 0.7,
    val_fraction_of_train: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    items = sorted(set(map(str, items)))
    if len(items) < 3:
        raise ValueError(f"At least three perturbations are required, got {len(items)}")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(items, dtype=object)
    rng.shuffle(shuffled)
    n_train_pool = min(len(items) - 1, max(1, int(round(len(items) * train_fraction))))
    train_pool = list(map(str, shuffled[:n_train_pool]))
    test = sorted(map(str, shuffled[n_train_pool:]))
    n_val = min(len(train_pool) - 1, max(1, int(round(len(train_pool) * val_fraction_of_train))))
    val = sorted(train_pool[:n_val])
    train = sorted(train_pool[n_val:])
    return train, val, test


def condition_split(adata: ad.AnnData, train: set[str], val: set[str], test: set[str]) -> np.ndarray:
    conditions = adata.obs["condition"].astype(str)
    labels = np.full(adata.n_obs, "drop", dtype=object)
    labels[(conditions == CONTROL).to_numpy()] = "train"
    labels[conditions.isin(train).to_numpy()] = "train"
    labels[conditions.isin(val).to_numpy()] = "val"
    labels[conditions.isin(test).to_numpy()] = "test"
    return labels


def _target_control_fit_masks(
    datasets: Mapping[str, ad.AnnData], target_keys: Sequence[str]
) -> dict[str, np.ndarray]:
    """Select HVGs exclusively from controls of the target dataset(s)."""
    targets = set(map(str, target_keys))
    unknown = targets - set(datasets)
    if unknown:
        raise KeyError(f"Unknown target dataset(s) for HVG selection: {sorted(unknown)}")
    masks: dict[str, np.ndarray] = {}
    for key, data in datasets.items():
        conditions = data.obs["condition"].astype(str).to_numpy()
        masks[key] = (
            conditions == CONTROL
            if key in targets
            else np.zeros(data.n_obs, dtype=bool)
        )
    if not any(bool(mask.any()) for mask in masks.values()):
        raise ValueError(f"Target dataset(s) have no control cells: {sorted(targets)}")
    return masks


def _build_target_control_panel(
    datasets: Mapping[str, ad.AnnData],
    *,
    target_keys: Sequence[str],
    config: PreprocessConfig,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    panel, report = build_feature_panel(
        datasets,
        n_hvg=config.n_hvg,
        fit_masks=_target_control_fit_masks(datasets, target_keys),
        max_fit_cells_per_dataset=config.hvg_max_cells_per_dataset,
        seed=seed,
    )
    report["hvg_fit_scope"] = "target_control_cells_only"
    report["hvg_target_datasets"] = list(map(str, target_keys))
    report["perturbed_cells_used_for_hvg"] = 0
    return panel, report


def _panel_union(panels: Mapping[str, Sequence[str]]) -> list[str]:
    """Return one deterministic storage panel containing every fold panel once."""
    if not panels:
        raise ValueError("Cannot build a storage panel from zero ready folds")
    canonical: dict[str, str] = {}
    for panel in panels.values():
        for gene in panel:
            canonical.setdefault(str(gene).upper(), str(gene))
    return [canonical[key] for key in sorted(canonical)]


def cell_validation_split(adata: ad.AnnData, *, seed: int, test_mask: np.ndarray) -> np.ndarray:
    """Cross-cell tasks: held-out perturbations test; 10% train conditions val."""
    labels = np.full(adata.n_obs, "train", dtype=object)
    labels[np.asarray(test_mask, dtype=bool)] = "test"
    conditions = adata.obs["condition"].astype(str).to_numpy()
    candidates = sorted(set(conditions[(labels == "train") & (conditions != CONTROL)]))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    n_val = max(1, int(round(0.1 * len(candidates)))) if len(candidates) > 1 else 0
    labels[np.isin(conditions, candidates[:n_val]) & (labels == "train")] = "val"
    return labels


STANDARD_SPLITS = ("train", "val", "test", "drop")


def _condition_stats(conditions: Any) -> dict[str, Any]:
    values = conditions.astype(str)
    counts = values.value_counts()
    perturbations = counts.drop(CONTROL, errors="ignore").sort_index()
    return {
        "n_cells": int(len(values)),
        "n_control_cells": int((values == CONTROL).sum()),
        "n_perturbed_cells": int((values != CONTROL).sum()),
        "n_perturbations": int(len(perturbations)),
        "perturbations": [str(x) for x in perturbations.index],
        "perturbation_cell_counts": {
            str(condition): int(count) for condition, count in perturbations.items()
        },
    }


def _partition_details(adata: ad.AnnData, column: str) -> dict[str, Any]:
    if column not in adata.obs:
        raise KeyError(f"Split column {column!r} is absent from AnnData.obs")
    labels = adata.obs[column].astype(str)
    observed = list(STANDARD_SPLITS)
    observed.extend(sorted(set(labels.unique()) - set(STANDARD_SPLITS)))
    splits = {
        split: _condition_stats(adata.obs.loc[labels == split, "condition"])
        for split in observed
        if bool((labels == split).any()) or split in STANDARD_SPLITS
    }
    result: dict[str, Any] = {
        "split_cells": {split: int(stats["n_cells"]) for split, stats in splits.items()},
        "split_perturbations": {
            split: list(stats["perturbations"]) for split, stats in splits.items()
        },
        "splits": splits,
    }
    if "evaluation_group" in adata.obs:
        groups: dict[str, Any] = {}
        group_values = adata.obs["evaluation_group"].astype(str)
        for split in observed:
            split_mask = labels == split
            if not split_mask.any():
                continue
            groups[split] = {
                group: _condition_stats(
                    adata.obs.loc[split_mask & (group_values == group), "condition"]
                )
                for group in sorted(group_values[split_mask].unique())
            }
        result["evaluation_groups"] = groups
    return result


def _split_report(adata: ad.AnnData, column: str) -> dict[str, Any]:
    overall = _condition_stats(adata.obs["condition"])
    pert_counts = list(overall["perturbation_cell_counts"].values())
    conditions = adata.obs["condition"].astype(str)
    batches = adata.obs["gpm_batch"].astype(str)
    report = {
        "n_cells": overall["n_cells"],
        "n_genes": int(adata.n_vars),
        "n_control_cells": overall["n_control_cells"],
        "n_perturbed_cells": overall["n_perturbed_cells"],
        "n_perturbations": overall["n_perturbations"],
        "perturbation_cell_counts": overall["perturbation_cell_counts"],
        "perturbation_cells": {
            "min": int(min(pert_counts)) if pert_counts else 0,
            "median": float(np.median(pert_counts)) if pert_counts else 0.0,
            "max": int(max(pert_counts)) if pert_counts else 0,
        },
        "n_batches": int(batches.nunique()),
        "batch_cells": {
            str(batch): int(count)
            for batch, count in batches.value_counts().sort_index().items()
        },
        "batch_control_cells": {
            str(batch): int(count)
            for batch, count in batches[conditions == CONTROL].value_counts().sort_index().items()
        },
    }
    report.update(_partition_details(adata, column))
    return report


def _summary_rows(task: str, folds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_name = str(fold.get("name", "unknown"))
        status = str(fold.get("status", "ready"))
        if status == "skipped":
            rows.append({
                "task": task, "fold": fold_name, "status": status,
                "dataset": "ALL", "scope": "fold", "split": "skipped",
                "evaluation_group": "", "n_genes": 0, "n_perturbations": 0,
                "n_cells": 0, "n_control_cells": 0, "n_perturbed_cells": 0,
                "perturbations": [], "perturbation_cell_counts": {},
                "reason": str(fold.get("reason", "")),
            })
            continue

        aggregate: dict[str, dict[str, Any]] = {}
        for dataset_key, report in fold.get("report", {}).items():
            all_stats = {
                "n_cells": int(report.get("n_cells", 0)),
                "n_control_cells": int(report.get("n_control_cells", 0)),
                "n_perturbed_cells": int(report.get("n_perturbed_cells", 0)),
                "n_perturbations": int(report.get("n_perturbations", 0)),
                "perturbations": sorted(report.get("perturbation_cell_counts", {})),
                "perturbation_cell_counts": dict(report.get("perturbation_cell_counts", {})),
            }
            split_entries = {"all": all_stats, **dict(report.get("splits", {}))}
            for split, stats in split_entries.items():
                rows.append({
                    "task": task, "fold": fold_name, "status": status,
                    "dataset": str(dataset_key), "scope": "dataset", "split": str(split),
                    "evaluation_group": "", "n_genes": int(report.get("n_genes", 0)),
                    "n_perturbations": int(stats.get("n_perturbations", 0)),
                    "n_cells": int(stats.get("n_cells", 0)),
                    "n_control_cells": int(stats.get("n_control_cells", 0)),
                    "n_perturbed_cells": int(stats.get("n_perturbed_cells", 0)),
                    "perturbations": list(stats.get("perturbations", [])),
                    "perturbation_cell_counts": dict(stats.get("perturbation_cell_counts", {})),
                    "reason": "",
                })
                bucket = aggregate.setdefault(str(split), {
                    "n_cells": 0, "n_control_cells": 0, "n_perturbed_cells": 0,
                    "perturbations": set(), "perturbation_cell_counts": {},
                })
                for field in ("n_cells", "n_control_cells", "n_perturbed_cells"):
                    bucket[field] += int(stats.get(field, 0))
                bucket["perturbations"].update(map(str, stats.get("perturbations", [])))
                for condition, count in stats.get("perturbation_cell_counts", {}).items():
                    bucket["perturbation_cell_counts"][str(condition)] = (
                        bucket["perturbation_cell_counts"].get(str(condition), 0) + int(count)
                    )

            for split, groups in report.get("evaluation_groups", {}).items():
                for group, stats in groups.items():
                    rows.append({
                        "task": task, "fold": fold_name, "status": status,
                        "dataset": str(dataset_key), "scope": "evaluation_group",
                        "split": str(split), "evaluation_group": str(group),
                        "n_genes": int(report.get("n_genes", 0)),
                        "n_perturbations": int(stats.get("n_perturbations", 0)),
                        "n_cells": int(stats.get("n_cells", 0)),
                        "n_control_cells": int(stats.get("n_control_cells", 0)),
                        "n_perturbed_cells": int(stats.get("n_perturbed_cells", 0)),
                        "perturbations": list(stats.get("perturbations", [])),
                        "perturbation_cell_counts": dict(stats.get("perturbation_cell_counts", {})),
                        "reason": "",
                    })

        for split, stats in aggregate.items():
            perturbations = sorted(stats["perturbations"])
            rows.append({
                "task": task, "fold": fold_name, "status": status,
                "dataset": "ALL", "scope": "fold", "split": split,
                "evaluation_group": "", "n_genes": len(fold.get("gene_panel", [])),
                "n_perturbations": len(perturbations), "n_cells": stats["n_cells"],
                "n_control_cells": stats["n_control_cells"],
                "n_perturbed_cells": stats["n_perturbed_cells"],
                "perturbations": perturbations,
                "perturbation_cell_counts": stats["perturbation_cell_counts"],
                "reason": "",
            })
    return rows


def _write_split_summaries(
    task: str, output_dir: Path, folds: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    rows = _summary_rows(task, folds)
    json_path = output_dir / "split_summary.json"
    csv_path = output_dir / "split_summary.csv"
    write_json(json_path, {"schema_version": 1, "task": task, "rows": rows})
    fields = [
        "task", "fold", "status", "dataset", "scope", "split", "evaluation_group",
        "n_genes", "n_perturbations", "n_cells", "n_control_cells",
        "n_perturbed_cells", "perturbations", "perturbation_cell_counts", "reason",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["perturbations"] = ";".join(row["perturbations"])
            csv_row["perturbation_cell_counts"] = json.dumps(
                row["perturbation_cell_counts"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow(csv_row)

    print("\nSplit summary (fold totals):")
    print("fold\tsplit\tperturbations\tperturbed_cells\tcontrols\ttotal_cells")
    visible_splits = {
        "all", "train", "val", "finetune", "test", "inference", "drop", "skipped"
    }
    for row in rows:
        if row["scope"] == "fold" and row["split"] in visible_splits:
            print(
                f"{row['fold']}\t{row['split']}\t{row['n_perturbations']}\t"
                f"{row['n_perturbed_cells']}\t{row['n_control_cells']}\t{row['n_cells']}"
            )
    return {"json": json_path.name, "csv": csv_path.name}


def _persist(
    task: str,
    output_dir: Path,
    datasets: Mapping[str, ad.AnnData],
    specs: Mapping[str, Any],
    base_reports: Mapping[str, Any],
    panel_report: Mapping[str, Any],
    folds: list[dict[str, Any]],
    existing_manifest: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    new_dataset_keys = list(datasets)
    artifacts: dict[str, Any] = {}
    for key in new_dataset_keys:
        ensure_memory_headroom(f"before writing {key}", 1.0)
        artifacts[key] = write_dataset_artifacts(
            datasets[key], dataset_key=key, output_dir=output_dir
        )
        if isinstance(datasets, dict):
            del datasets[key]
        release_memory()
        print(memory_status(f"wrote and released {key}"), flush=True)
    manifest = copy.deepcopy(dict(existing_manifest)) if existing_manifest else {}
    if manifest and manifest.get("task") != task:
        raise ValueError(
            f"Existing manifest task is {manifest.get('task')!r}, expected {task!r}"
        )
    dataset_metadata = dict(manifest.get("datasets", {}))
    for key in new_dataset_keys:
        dataset_metadata[key] = {
            "dataset_id": specs[key].dataset_id,
            "cell_line": specs[key].cell_line,
            "technology": specs[key].technology,
            **artifacts[key],
            "base_report": base_reports[key],
        }
    existing_folds = {
        str(fold["name"]): fold for fold in manifest.get("folds", [])
    }
    for fold in folds:
        existing_folds[str(fold["name"])] = fold
    merged_panel = copy.deepcopy(manifest.get("feature_panel", {}))
    if "per_dataset" in panel_report:
        merged_panel.setdefault("per_dataset", {}).update(panel_report["per_dataset"])
    elif panel_report:
        merged_panel = dict(panel_report)
    dataset_keys = list(manifest.get("dataset_keys", []))
    dataset_keys.extend(key for key in new_dataset_keys if key not in dataset_keys)
    manifest.update({
        "schema_version": 4,
        "task": task,
        "dataset_keys": dataset_keys,
        "datasets": dataset_metadata,
        "feature_panel": merged_panel,
        "folds": list(existing_folds.values()),
    })
    manifest["split_summary_files"] = _write_split_summaries(
        task, output_dir, manifest["folds"]
    )
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    return path


def _single_fold_is_reusable(
    manifest: Mapping[str, Any],
    key: str,
    output_dir: Path,
    config: PreprocessConfig,
) -> bool:
    metadata = manifest.get("datasets", {}).get(key)
    fold = next(
        (item for item in manifest.get("folds", []) if str(item.get("name")) == key),
        None,
    )
    if not metadata or not fold or fold.get("status") == "skipped":
        return False
    if (
        fold.get("hvg_fit_scope") != "target_control_cells_only"
        or metadata.get("base_report", {}).get("ppi_filter", {}).get("enabled") is not False
    ):
        return False
    base_report = metadata.get("base_report", {})
    if base_report.get("go_filter_enabled", "go_filter" in base_report) != config.use_go_filter:
        return False
    cap_report = base_report.get(
        "memory_safety_cell_cap", {}
    )
    expected_cap = config.max_cells_per_perturbation
    expected_controls = config.max_control_cells
    if (
        cap_report.get("max_cells_per_perturbation") != expected_cap
        or cap_report.get("max_control_cells") != expected_controls
        or cap_report.get("sampling_seed") != config.sampling_seed
        or cap_report.get("sampling_strategy") != "condition_then_batch_stratified"
        or cap_report.get("cap_validation_passed") is not True
    ):
        return False
    required = (metadata.get("h5ad"), metadata.get("control_means"))
    return all(value and manifest_artifact_path(value, output_dir / "manifest.json").is_file() for value in required)


def _upgrade_existing_single_reports(
    manifest: dict[str, Any], output_dir: Path
) -> None:
    """Add detailed split counts from stored obs without changing stored labels."""
    for fold in manifest.get("folds", []):
        if fold.get("status") == "skipped":
            continue
        split_column = str(fold.get("split_column", "split"))
        for key in fold.get("dataset_keys", []):
            report = fold.setdefault("report", {}).setdefault(key, {})
            if "splits" in report:
                continue
            metadata = manifest.get("datasets", {}).get(key, {})
            h5ad_value = metadata.get("h5ad")
            if not h5ad_value:
                continue
            h5ad_path = manifest_artifact_path(h5ad_value, output_dir / "manifest.json")
            if not h5ad_path.is_file():
                continue
            stored = ad.read_h5ad(h5ad_path, backed="r")
            try:
                overall = _condition_stats(stored.obs["condition"])
                report.setdefault("n_cells", overall["n_cells"])
                report.setdefault("n_control_cells", overall["n_control_cells"])
                report.setdefault("n_perturbed_cells", overall["n_perturbed_cells"])
                report.setdefault("n_perturbations", overall["n_perturbations"])
                report.setdefault(
                    "perturbation_cell_counts", overall["perturbation_cell_counts"]
                )
                report.setdefault("n_genes", int(stored.n_vars))
                report.update(_partition_details(stored, split_column))
            finally:
                if getattr(stored, "file", None) is not None:
                    stored.file.close()


def build_single_unseen_pert(args: argparse.Namespace) -> Path:
    keys = tuple(args.dataset_keys)
    if not keys:
        raise ValueError("single_unseen_pert requires at least one dataset")

    manifest_path = args.output_dir / "manifest.json"
    existing: Optional[dict[str, Any]] = None
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("task") != "single_unseen_pert":
            raise ValueError(
                f"Existing manifest task is {existing.get('task')!r}, "
                "expected 'single_unseen_pert'"
            )
        if int(existing.get("schema_version", 0)) < 4:
            print(
                "[rebuild] legacy single-dataset manifest uses the retired "
                "HVG/PPI protocol; rebuilding the requested datasets"
            )
            existing = None
        else:
            _upgrade_existing_single_reports(existing, args.output_dir)

    reusable = {
        key for key in keys
        if existing is not None and _single_fold_is_reusable(
            existing, key, args.output_dir, args.config
        )
    }
    for key in keys:
        if key in reusable:
            print(f"[reuse] {DATASET_DISPLAY_NAMES[key]}: existing split and artifacts")
    new_keys = tuple(key for key in keys if key not in reusable)
    current_manifest = existing
    result_path = manifest_path
    if not new_keys:
        return _persist(
            "single_unseen_pert", args.output_dir, {}, {}, {},
            {"per_dataset": {}}, [], existing_manifest=current_manifest,
        )

    # Replogle RPE1 and K562 are both large.  Load, preprocess, panel-select,
    # persist and release one dataset before opening the next one.  Persisting
    # incrementally also makes a completed dataset reusable if a later input
    # fails, without changing any split or feature-selection semantics.
    for key in new_keys:
        datasets, specs, reports = load_base_datasets(
            (key,), args.sources, args.config
        )
        source_data = datasets.pop(key)
        restricted, target_report = restrict_targets_to_gene_intersection(
            {key: source_data},
            log_label=f"single_unseen_pert/{key}",
        )
        source_data = restricted[key]
        reports[key]["target_gene_intersection_filter"] = target_report
        seed_offset = list(SINGLE_DATASET_KEYS).index(key)
        train, val, test = split_items(
            single_conditions(source_data), seed=args.seed + seed_offset
        )
        labels = condition_split(source_data, set(train), set(val), set(test))
        source_data.obs["split"] = labels
        ensure_memory_headroom(
            f"before HVG selection for {key}", args.config.min_free_memory_gib
        )
        panel, panel_report = _build_target_control_panel(
            {key: source_data}, target_keys=(key,), config=args.config,
            seed=args.seed,
        )
        item = subset_to_panel(source_data, panel)
        del source_data
        release_memory()
        fold = {
            "name": key, "dataset_keys": [key], "split_column": "split",
            "gene_panel": panel,
            "hvg_target_datasets": [key],
            "hvg_fit_scope": "target_control_cells_only",
            "report": {key: _split_report(item, "split")},
        }
        result_path = _persist(
            "single_unseen_pert", args.output_dir, {key: item}, specs, reports,
            {"per_dataset": {key: panel_report}}, [fold],
            existing_manifest=current_manifest,
        )
        del item, datasets, specs, reports, labels, panel, panel_report, fold
        release_memory()
        with result_path.open("r", encoding="utf-8") as handle:
            current_manifest = json.load(handle)
    return result_path


def build_double_unseen_pert(args: argparse.Namespace) -> Path:
    key = "norman"
    datasets, specs, reports = load_base_datasets((key,), args.sources, args.config)
    datasets, target_report = restrict_targets_to_gene_intersection(
        datasets,
        log_label="double_unseen_pert/norman",
    )
    reports[key]["target_gene_intersection_filter"] = target_report
    data = datasets[key]
    single_genes = sorted(condition_targets(c)[0] for c in single_conditions(data))
    seen, val_genes, unseen = split_items(single_genes, seed=args.seed)
    seen_set, unseen_set = set(seen) | set(val_genes), set(unseen)
    combos = [c for c in data.obs["condition"].astype(str).unique() if len(condition_targets(c)) == 2]
    rng = np.random.default_rng(args.seed + 101)
    zero_unseen = [c for c in combos if not (set(condition_targets(c)) & unseen_set)]
    rng.shuffle(zero_unseen)
    n_zero_test = max(1, int(round(0.3 * len(zero_unseen)))) if zero_unseen else 0
    zero_test = set(zero_unseen[:n_zero_test])
    train_conditions = {c for c in single_conditions(data) if condition_targets(c)[0] in seen_set}
    train_conditions.update(set(zero_unseen) - zero_test)
    val_conditions = {c for c in train_conditions if condition_targets(c)[0] in set(val_genes)}
    train_conditions -= val_conditions
    test_conditions = set(single_conditions(data)) - train_conditions - val_conditions
    test_conditions.update(set(combos) - train_conditions - val_conditions)
    labels = condition_split(data, train_conditions, val_conditions, test_conditions)
    data.obs["split"] = labels
    groups = np.full(data.n_obs, "control", dtype=object)
    for i, condition in enumerate(data.obs["condition"].astype(str)):
        targets = set(condition_targets(condition))
        if len(targets) == 1:
            groups[i] = "single"
        elif len(targets) == 2:
            groups[i] = f"{len(targets & unseen_set)}/2 unseen"
    data.obs["evaluation_group"] = groups
    ensure_memory_headroom(
        "before HVG selection for norman", args.config.min_free_memory_gib
    )
    panel, panel_report = _build_target_control_panel(
        {key: data}, target_keys=(key,), config=args.config, seed=args.seed,
    )
    panel_report["target_gene_intersection_filter"] = target_report
    data = subset_to_panel(data, panel)
    group_counts = data.obs.loc[data.obs["split"] == "test", "evaluation_group"].value_counts().to_dict()
    fold = {
        "name": "norman", "dataset_keys": [key], "split_column": "split", "gene_panel": panel,
        "hvg_target_datasets": [key], "hvg_fit_scope": "target_control_cells_only",
        "unseen_genes": sorted(unseen_set), "evaluation_groups": ["single", "0/2 unseen", "1/2 unseen", "2/2 unseen"],
        "test_group_cells": {str(k): int(v) for k, v in group_counts.items()},
        "report": {key: _split_report(data, "split")},
    }
    return _persist("double_unseen_pert", args.output_dir, {key: data}, specs, reports, panel_report, [fold])


def _restrict_conditions(data: ad.AnnData, allowed: set[str]) -> ad.AnnData:
    keep = data.obs["condition"].astype(str).isin(allowed | {CONTROL})
    return data[keep].copy()


def _target_intersection_plan(
    datasets: Mapping[str, ad.AnnData],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    if not datasets:
        raise ValueError("No datasets supplied for target/gene intersection")

    gene_sets = {
        key: {str(g).strip().upper() for g in data.var_names}
        for key, data in datasets.items()
    }
    common = set.intersection(*gene_sets.values())
    usable_by_dataset: dict[str, set[str]] = {}
    report: dict[str, Any] = {}

    for key, data in datasets.items():
        conditions = data.obs["condition"].astype(str)
        non_control = sorted(
            str(c) for c in conditions.unique() if str(c) != CONTROL
        )
        targets = {
            target.upper()
            for condition in non_control
            for target in condition_targets(condition)
        }
        usable = {CONTROL}
        for condition in non_control:
            condition_genes = condition_targets(condition)
            if condition_genes and all(
                target.upper() in common for target in condition_genes
            ):
                usable.add(condition)

        retained = sorted(usable - {CONTROL})
        dropped = sorted(set(non_control) - set(retained))
        usable_by_dataset[key] = usable
        report[key] = {
            # Backward-compatible fields used in existing manifests.
            "dropped_perturbations": dropped,
            "n_dropped": len(dropped),
            # Explicit audit fields.
            "n_genes": len(gene_sets[key]),
            "n_perturbations_before": len(non_control),
            "n_perturbations_after": len(retained),
            "retained_perturbations": retained,
            "n_targets": len(targets),
            "n_targets_in_own_panel": len(targets & gene_sets[key]),
            "n_targets_in_common_panel": len(targets & common),
            "missing_target_genes": sorted(targets - common),
            "n_cells_before": int(data.n_obs),
            "n_cells_after": int(conditions.isin(usable).sum()),
        }

    keys = list(gene_sets)
    leave_one_out: dict[str, int] = {}
    for omitted in keys:
        others = [gene_sets[key] for key in keys if key != omitted]
        leave_one_out[omitted] = (
            len(set.intersection(*others)) if others else len(gene_sets[omitted])
        )

    report["_summary"] = {
        "n_common_genes": len(common),
        "dataset_gene_counts": {
            key: len(gene_sets[key]) for key in keys
        },
        "leave_one_out_common_genes": leave_one_out,
    }
    return usable_by_dataset, report


def target_gene_intersection_report(
    datasets: Mapping[str, ad.AnnData],
) -> dict[str, Any]:
    """Audit target compatibility without changing any dataset."""
    _, report = _target_intersection_plan(datasets)
    return report


def _print_target_intersection(label: str, report: Mapping[str, Any]) -> None:
    summary = dict(report.get("_summary", {}))
    common = int(summary.get("n_common_genes", 0))
    dataset_keys = [key for key in report if key != "_summary"]
    retention = "; ".join(
        f"{key} {report[key]['n_perturbations_before']}"
        f"->{report[key]['n_perturbations_after']}"
        for key in dataset_keys
    )
    print(f"[{label}] common measured genes={common}; {retention}", flush=True)

    own_panel_issues = [
        f"{key} {report[key]['n_targets_in_own_panel']}/{report[key]['n_targets']}"
        for key in dataset_keys
        if report[key]["n_targets_in_own_panel"] < report[key]["n_targets"]
    ]
    if own_panel_issues:
        print(
            f"[{label}] targets present in own measured expression panel: "
            + "; ".join(own_panel_issues),
            flush=True,
        )

    missing_targets = [
        f"{key}: {', '.join(report[key].get('missing_target_genes', []))}"
        for key in dataset_keys
        if report[key].get("missing_target_genes")
    ]
    if missing_targets:
        print(
            f"[{label}] removed conditions with unmeasured targets: "
            + "; ".join(missing_targets),
            flush=True,
        )


def _intersection_error_context(report: Mapping[str, Any]) -> str:
    summary = dict(report.get("_summary", {}))
    common = int(summary.get("n_common_genes", 0))
    gene_counts = dict(summary.get("dataset_gene_counts", {}))
    leave_one_out = dict(summary.get("leave_one_out_common_genes", {}))
    bottleneck = max(leave_one_out, key=leave_one_out.get) if leave_one_out else None

    retention = ", ".join(
        f"{key}={report[key]['n_perturbations_before']}"
        f"->{report[key]['n_perturbations_after']}"
        for key in report
        if key != "_summary"
    )
    genes = ", ".join(f"{key}={value}" for key, value in gene_counts.items())
    bottleneck_text = (
        f"; largest leave-one-out intersection: remove {bottleneck} -> "
        f"{leave_one_out[bottleneck]} genes"
        if bottleneck is not None
        else ""
    )
    return (
        f"common_genes={common}; gene_counts=[{genes}]; "
        f"perturbations=[{retention}]{bottleneck_text}"
    )


def restrict_targets_to_gene_intersection(
    datasets: Mapping[str, ad.AnnData],
    *,
    log_label: Optional[str] = None,
) -> tuple[dict[str, ad.AnnData], dict[str, Any]]:
    """Keep only perturbations whose targets exist in every dataset gene panel."""
    usable_by_dataset, report = _target_intersection_plan(datasets)
    if log_label:
        _print_target_intersection(log_label, report)

    result = datasets if isinstance(datasets, dict) else dict(datasets)
    for key in list(result):
        data = result[key]
        conditions = data.obs["condition"].astype(str)
        keep = conditions.isin(usable_by_dataset[key]).to_numpy(dtype=bool)
        if not bool(keep.all()):
            result[key] = data[keep].copy()
            del data
            release_memory()
    return result, report

def build_unseen_cell(args: argparse.Namespace) -> Path:
    keys = TASK_DATASETS["unseen_cell"]
    datasets, specs, reports = load_base_datasets(keys, args.sources, args.config)
    common = set.intersection(*(set(single_conditions(datasets[key])) for key in keys))
    if len(common) < 10:
        raise ValueError(f"Only {len(common)} common perturbations across four cell lines")
    for key in list(datasets):
        old = datasets[key]
        datasets[key] = _restrict_conditions(old, common)
        del old
        release_memory()
    datasets, target_report = restrict_targets_to_gene_intersection(datasets)
    common = set.intersection(*(set(single_conditions(data)) for data in datasets.values()))
    fold_panels: dict[str, list[str]] = {}
    fold_panel_reports: dict[str, Any] = {}
    for heldout in keys:
        ensure_memory_headroom(
            f"before target-control HVG selection for unseen_cell/{heldout}",
            args.config.min_free_memory_gib,
        )
        fold_panels[heldout], fold_panel_reports[heldout] = _build_target_control_panel(
            datasets,
            target_keys=(heldout,),
            config=args.config,
            seed=args.seed,
        )
    storage_panel = _panel_union(fold_panels)
    panel_report = {
        "hvg_selection_protocol": "per_fold_target_control_cells_only",
        "per_fold": fold_panel_reports,
        "storage_panel_is_fold_union": True,
        "n_storage_union_genes": len(storage_panel),
        "target_gene_intersection_filter": target_report,
    }
    for key in list(datasets):
        old = datasets[key]
        datasets[key] = subset_to_panel(old, storage_panel)
        del old
        release_memory()
    folds: list[dict[str, Any]] = []
    for offset, heldout in enumerate(keys):
        column = f"split_{heldout}"
        fold_report: dict[str, Any] = {}
        train_seen: set[str] = set()
        for key_index, (key, data) in enumerate(datasets.items()):
            if key == heldout:
                continue
            data.obs[column] = cell_validation_split(
                data,
                seed=args.seed + offset * 100 + key_index,
                test_mask=np.zeros(data.n_obs, dtype=bool),
            )
            train_seen.update(
                data.obs.loc[data.obs[column].astype(str) == "train", "condition"].astype(str)
            )
        heldout_data = datasets[heldout]
        heldout_conditions = heldout_data.obs["condition"].astype(str)
        # Reserve this split label for the standard TxPert protocol. GPM uses held-out controls for panel selection and inference.
        heldout_data.obs[column] = np.where(
            heldout_conditions == CONTROL,
            "train",
            np.where(heldout_conditions.isin(train_seen), "test", "drop"),
        )
        panel = fold_panels[heldout]
        for key, data in datasets.items():
            fold_report[key] = _split_report(data, column)
            fold_report[key]["n_genes"] = len(panel)
        folds.append({
            "name": heldout, "heldout_dataset": heldout, "dataset_keys": list(keys),
            "split_column": column, "gene_panel": panel,
            "hvg_target_datasets": [heldout],
            "hvg_fit_scope": "target_control_cells_only",
            "storage_panel_is_fold_union": True,
            "heldout_controls_used_for_optimization": False,
            "heldout_controls_used_for_feature_selection": True,
            "heldout_controls_available_at_inference": True,
            "n_common_perturbations": len(common), "report": fold_report,
        })
    return _persist("unseen_cell", args.output_dir, datasets, specs, reports, panel_report, folds)


def balance_perturbed_cells(datasets: Mapping[str, ad.AnnData], *, seed: int) -> tuple[dict[str, ad.AnnData], dict[str, Any]]:
    pert_cells = {
        key: int((data.obs["condition"].astype(str) != CONTROL).sum()) for key, data in datasets.items()
    }
    target = min(pert_cells.values())
    result = datasets if isinstance(datasets, dict) else dict(datasets)
    selected_report: dict[str, Any] = {"target_perturbed_cells": target, "datasets": {}}
    for offset, (key, data) in enumerate(datasets.items()):
        condition_values = data.obs["condition"].astype(str).to_numpy()
        counts = data.obs.loc[condition_values != CONTROL, "condition"].astype(str).value_counts()
        conditions = counts.index.to_numpy(dtype=object)
        rng = np.random.default_rng(seed + offset)
        rng.shuffle(conditions)
        chosen: list[str] = []
        chosen_indices: list[int] = []
        total = 0
        for condition in conditions:
            budget = target - total
            if budget <= 50:
                break
            candidates = np.where(condition_values == str(condition))[0]
            take = min(len(candidates), budget)
            if take <= 50:
                continue
            if take < len(candidates):
                candidates = np.sort(rng.choice(candidates, size=take, replace=False))
            chosen.append(str(condition))
            chosen_indices.extend(map(int, candidates))
            total += int(take)
        control_indices = np.where(condition_values == CONTROL)[0]
        keep_indices = np.sort(np.asarray([*control_indices, *chosen_indices], dtype=np.int64))
        result[key] = data[keep_indices].copy()
        del data
        release_memory()
        selected_report["datasets"][key] = {
            "before_perturbed_cells": pert_cells[key], "after_perturbed_cells": total,
            "n_selected_perturbations": len(chosen), "selected_perturbations": sorted(chosen),
            "last_condition_may_be_cell_subsampled": True,
        }
    return result, selected_report


def build_unseen_pert_cell(args: argparse.Namespace) -> Path:
    keys = TASK_DATASETS["unseen_pert_cell"]
    datasets, specs, reports = load_base_datasets(keys, args.sources, args.config)
    datasets, target_report = restrict_targets_to_gene_intersection(datasets)
    datasets, balance_report = balance_perturbed_cells(datasets, seed=args.seed)
    folds: list[dict[str, Any]] = []
    fold_panels: dict[str, list[str]] = {}
    fold_panel_reports: dict[str, Any] = {}
    for offset, heldout in enumerate(keys):
        column = f"split_{heldout}"
        source_keys = [key for key in keys if key != heldout]
        source_perturbations = set().union(*(
            set(single_conditions(datasets[key])) for key in source_keys
        ))
        if len(source_perturbations) < 2:
            folds.append({
                "name": heldout, "heldout_dataset": heldout, "status": "skipped",
                "reason": "fewer than two source perturbations for train/validation",
            })
            continue
        # Define unseen test conditions against all source conditions, including validation.
        unseen = set(single_conditions(datasets[heldout])) - source_perturbations
        if len(unseen) < 10:
            folds.append({
                "name": heldout, "heldout_dataset": heldout, "status": "skipped",
                "reason": f"only {len(unseen)} unseen perturbations after filtering (<10)",
            })
            continue

        # Use the same held-out validation conditions for every source cell line.
        candidates = np.asarray(sorted(source_perturbations), dtype=object)
        rng = np.random.default_rng(args.seed + offset * 100)
        rng.shuffle(candidates)
        n_val = min(len(candidates) - 1, max(1, int(round(0.1 * len(candidates)))))
        validation_perturbations = set(map(str, candidates[:n_val]))
        for key in source_keys:
            conditions = datasets[key].obs["condition"].astype(str)
            datasets[key].obs[column] = np.where(
                conditions == CONTROL,
                "train",
                np.where(conditions.isin(validation_perturbations), "val", "train"),
            )
        fold_report: dict[str, Any] = {}
        heldout_data = datasets[heldout]
        heldout_conditions = heldout_data.obs["condition"].astype(str)
        # Reserve this split label for the standard TxPert protocol. GPM uses held-out controls for panel selection and inference.
        heldout_data.obs[column] = np.where(
            heldout_conditions == CONTROL,
            "train",
            np.where(heldout_conditions.isin(unseen), "test", "drop"),
        )
        # Fail fast if future preprocessing changes violate double-zero-shot isolation.
        split_targets = {
            split: set().union(*(
                set(data.obs.loc[
                    data.obs[column].astype(str) == split, "condition"
                ].astype(str)) - {CONTROL}
                for data in datasets.values()
            ))
            for split in ("train", "val", "test")
        }
        if any(
            split_targets[left] & split_targets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise RuntimeError(f"{heldout}: train/val/test perturbations are not disjoint")
        ensure_memory_headroom(
            f"before target-control HVG selection for unseen_pert_cell/{heldout}",
            args.config.min_free_memory_gib,
        )
        panel, current_panel_report = _build_target_control_panel(
            datasets,
            target_keys=(heldout,),
            config=args.config,
            seed=args.seed,
        )
        fold_panels[heldout] = panel
        fold_panel_reports[heldout] = current_panel_report
        for key, data in datasets.items():
            fold_report[key] = _split_report(data, column)
            fold_report[key]["n_genes"] = len(panel)
        folds.append({
            "name": heldout, "heldout_dataset": heldout, "status": "ready",
            "dataset_keys": list(keys), "split_column": column, "gene_panel": panel,
            "hvg_target_datasets": [heldout],
            "hvg_fit_scope": "target_control_cells_only",
            "storage_panel_is_fold_union": True,
            "unseen_perturbations": sorted(unseen),
            "heldout_controls_used_for_optimization": False,
            "heldout_controls_used_for_feature_selection": True,
            "heldout_controls_available_at_inference": True,
            "validation_perturbations": sorted(validation_perturbations),
            "global_target_disjoint_validation": True,
            "test_unseen_against_train_and_validation": True,
            "report": fold_report,
        })
    storage_panel = _panel_union(fold_panels)
    panel_report = {
        "hvg_selection_protocol": "per_fold_target_control_cells_only",
        "per_fold": fold_panel_reports,
        "storage_panel_is_fold_union": True,
        "n_storage_union_genes": len(storage_panel),
        "balancing": balance_report,
        "target_gene_intersection_filter": target_report,
    }
    for key in list(datasets):
        old = datasets[key]
        datasets[key] = subset_to_panel(old, storage_panel)
        del old
        release_memory()
    return _persist("unseen_pert_cell", args.output_dir, datasets, specs, reports, panel_report, folds)


def build_across_tech(args: argparse.Namespace) -> Path:
    keys = TASK_DATASETS["across_tech"]
    datasets, specs, reports = load_base_datasets(keys, args.sources, args.config)
    # Cross-dataset expression features must share one gene space, but
    # perturbations do not need to occur in both technologies. Split the union
    # globally so a held-out perturbation is absent from training in both KD
    # (CRISPRi) and OE (CRISPRa).
    datasets, target_report = restrict_targets_to_gene_intersection(datasets)
    perturbations_by_dataset = {
        key: set(single_conditions(data)) for key, data in datasets.items()
    }
    perturbation_union = set.union(*perturbations_by_dataset.values())
    train, val, test = split_items(perturbation_union, seed=args.seed)
    for data in datasets.values():
        data.obs["split"] = condition_split(data, set(train), set(val), set(test))
    ensure_memory_headroom(
        "before cross-dataset HVG selection for across_tech",
        args.config.min_free_memory_gib,
    )
    panel, panel_report = _build_target_control_panel(
        datasets,
        target_keys=keys,
        config=args.config,
        seed=args.seed,
    )
    panel_report["target_gene_intersection_filter"] = target_report
    for key in list(datasets):
        old = datasets[key]
        datasets[key] = subset_to_panel(old, panel)
        del old
        release_memory()
    fold = {
        "name": "tian_mixed_tech_unseen_perturbations",
        "dataset_keys": list(keys), "split_column": "split", "gene_panel": panel,
        "hvg_target_datasets": list(keys),
        "hvg_fit_scope": "target_control_cells_only",
        "perturbation_set": "union_across_technologies",
        "shared_condition_split_across_technologies": False,
        "global_unseen_perturbation_split": True,
        "n_union_perturbations": len(perturbation_union),
        "n_perturbations_by_dataset": {
            key: len(values) for key, values in perturbations_by_dataset.items()
        },
        "train_perturbations": train,
        "validation_perturbations": val,
        "unseen_test_perturbations": test,
        "batch_matched_controls_required": True,
        "report": {key: _split_report(data, "split") for key, data in datasets.items()},
    }
    return _persist("across_tech", args.output_dir, datasets, specs, reports, panel_report, [fold])


def build_zero_patient(args: argparse.Namespace) -> Path:
    """Six-source pretraining plus condition-disjoint Frangieh adaptation/test."""
    pretrain_keys = (
        "dixit", "xu", "yao", "tian_crispri", "tian_crispra", "norman"
    )
    patient_key = "frangieh"
    keys = (*pretrain_keys, patient_key)

    # Load and audit the six source datasets first.  If their common measured-gene
    # space is already invalid, fail before paying the Frangieh I/O cost.
    datasets, specs, reports = load_base_datasets(
        pretrain_keys, args.sources, args.config
    )

    old_norman = datasets["norman"]
    datasets["norman"] = _restrict_conditions(
        old_norman, set(single_conditions(old_norman))
    )
    del old_norman
    release_memory()

    source_audit = target_gene_intersection_report(datasets)
    _print_target_intersection("zero_patient/source preflight", source_audit)
    source_common = int(source_audit["_summary"]["n_common_genes"])
    source_perturbations = set().union(*(
        set(source_audit[key]["retained_perturbations"])
        for key in pretrain_keys
    ))
    if source_common < int(args.config.n_hvg):
        raise ValueError(
            "zero_patient source datasets do not share enough measured genes "
            f"for {args.config.n_hvg} HVGs; "
            + _intersection_error_context(source_audit)
        )
    if len(source_perturbations) < 2:
        raise ValueError(
            "zero_patient source target/gene intersection leaves fewer than two "
            "pretraining perturbations; "
            + _intersection_error_context(source_audit)
        )

    # Only after the source preflight succeeds do we load the expensive patient
    # dataset.
    patient_datasets, patient_specs, patient_reports = load_base_datasets(
        (patient_key,), args.sources, args.config
    )
    datasets.update(patient_datasets)
    specs.update(patient_specs)
    reports.update(patient_reports)

    datasets, target_report = restrict_targets_to_gene_intersection(
        datasets, log_label="zero_patient/final intersection"
    )
    final_common = int(target_report["_summary"]["n_common_genes"])
    if final_common < int(args.config.n_hvg):
        raise ValueError(
            "zero_patient source+Frangieh common measured gene space is too small for "
            f"{args.config.n_hvg} HVGs; "
            + _intersection_error_context(target_report)
        )

    pretrain_perturbations = set().union(*(
        set(single_conditions(datasets[key])) for key in pretrain_keys
    ))
    if len(pretrain_perturbations) < 2:
        raise ValueError(
            "zero_patient source+Frangieh target/gene intersection leaves fewer "
            "than two pretraining perturbations; "
            + _intersection_error_context(target_report)
        )

    shuffled = np.asarray(sorted(pretrain_perturbations), dtype=object)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(shuffled)
    n_val = min(
        len(shuffled) - 1,
        max(1, int(round(0.1 * len(shuffled)))),
    )
    validation_perturbations = set(map(str, shuffled[:n_val]))
    for key in pretrain_keys:
        conditions = datasets[key].obs["condition"].astype(str)
        datasets[key].obs["split"] = np.where(
            conditions == CONTROL,
            "train",
            np.where(
                conditions.isin(validation_perturbations),
                "val",
                "train",
            ),
        )

    patient_conditions = sorted(single_conditions(datasets[patient_key]))
    if len(patient_conditions) < 2:
        raise ValueError(
            "Frangieh has fewer than two perturbations after the common-gene "
            "filter; " + _intersection_error_context(target_report)
        )

    patient_shuffled = np.asarray(patient_conditions, dtype=object)
    patient_rng = np.random.default_rng(args.seed + 10_001)
    patient_rng.shuffle(patient_shuffled)
    n_test = min(
        len(patient_shuffled) - 1,
        max(1, int(round(0.3 * len(patient_shuffled)))),
    )
    test_perturbations = set(map(str, patient_shuffled[:n_test]))
    finetune_perturbations = set(patient_conditions) - test_perturbations

    patient = datasets[patient_key]
    conditions = patient.obs["condition"].astype(str)
    patient.obs["split"] = np.where(
        conditions == CONTROL,
        "inference",
        np.where(
            conditions.isin(test_perturbations),
            "test",
            "finetune",
        ),
    )

    # Target-aware transfer protocol: feature selection may use the target
    # patient's unperturbed controls, but no patient perturbation cell.  The
    # source/target measured-gene intersection is still enforced above.
    ensure_memory_headroom(
        "before cross-dataset HVG selection for zero_patient",
        args.config.min_free_memory_gib,
    )
    panel, panel_report = _build_target_control_panel(
        datasets,
        target_keys=(patient_key,),
        config=args.config,
        seed=args.seed,
    )
    panel_report["target_gene_intersection_filter"] = target_report
    panel_report["hvg_fit_patient_perturbed_cells"] = 0

    for key in list(datasets):
        old = datasets[key]
        datasets[key] = subset_to_panel(old, panel)
        del old
        release_memory()

    patient = datasets[patient_key]
    patient_train = set(patient.obs.loc[
        patient.obs["split"].astype(str) == "finetune", "condition"
    ].astype(str))
    patient_test = set(patient.obs.loc[
        patient.obs["split"].astype(str) == "test", "condition"
    ].astype(str))
    if patient_train & patient_test:
        raise RuntimeError("Frangieh fine-tune/test perturbations are not disjoint")
    if (patient.obs.loc[
        patient.obs["condition"].astype(str) == CONTROL, "split"
    ].astype(str) != "inference").any():
        raise RuntimeError("Frangieh controls leaked into an optimization split")

    fold = {
        "name": "frangieh_patient",
        "dataset_keys": list(keys),
        "pretrain_dataset_keys": list(pretrain_keys),
        "patient_dataset_key": patient_key,
        "split_column": "split",
        "gene_panel": panel,
        "hvg_target_datasets": [patient_key],
        "hvg_fit_scope": "target_control_cells_only",
        "perturbation_set": "single_target_only",
        "pretrain_validation_fraction": 0.1,
        "pretrain_validation_perturbations": sorted(validation_perturbations),
        "patient_finetune_fraction": 0.7,
        "patient_test_fraction": 0.3,
        "patient_finetune_split": "finetune",
        "patient_test_split": "test",
        "patient_control_split": "inference",
        "patient_finetune_perturbations": sorted(finetune_perturbations),
        "patient_test_perturbations": sorted(test_perturbations),
        "patient_test_disjoint_from_finetune": True,
        "patient_test_may_overlap_pretraining": True,
        "patient_controls_used_for_pretraining": False,
        "patient_controls_used_for_feature_selection": True,
        "patient_controls_used_as_independent_finetune_examples": False,
        "patient_controls_available_for_feature_selection_pairing_and_effect_centring": True,
        "strict_patient_dataset_blind_zero_shot": False,
        "pretraining_technologies": sorted({
            specs[key].technology for key in pretrain_keys
        }),
        "report": {
            key: _split_report(data, "split")
            for key, data in datasets.items()
        },
    }
    return _persist(
        "zero_patient",
        args.output_dir,
        datasets,
        specs,
        reports,
        panel_report,
        [fold],
    )


BUILDERS = {
    "single_unseen_pert": build_single_unseen_pert,
    "double_unseen_pert": build_double_unseen_pert,
    "unseen_cell": build_unseen_cell,
    "unseen_pert_cell": build_unseen_pert_cell,
    "across_tech": build_across_tech,
    "zero_patient": build_zero_patient,
}


def run_cli(
    task: str,
    *,
    allow_dataset_selection: bool = False,
    default_datasets: Sequence[str] = (),
) -> None:
    parser = argparse.ArgumentParser(description=f"Prepare GPM task: {task}")
    parser.add_argument("--source", action="append", default=[], help="DATASET_KEY=RAW_PATH")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / task)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--min-pert-cells-exclusive", type=int, default=50)
    parser.add_argument(
        "--max-cells-per-perturbation",
        type=int,
        default=DEFAULT_MAX_CELLS_PER_PERTURBATION,
        help="Seeded per-perturbation cap after >50-cell eligibility filtering; 0 disables",
    )
    parser.add_argument(
        "--max-control-cells",
        type=int,
        default=DEFAULT_MAX_CONTROL_CELLS,
        help="Seeded dataset-level control cap; 0 disables",
    )
    parser.add_argument(
        "--hvg-max-cells-per-dataset", type=int, default=10000,
        help="Maximum target-control cells per dataset used to fit an HVG panel",
    )
    parser.add_argument(
        "--min-free-memory-gib", type=float, default=4.0,
        help="Stop cleanly before loading another large matrix below this free-memory threshold",
    )
    parser.add_argument("--biogrid-path", type=Path, default=None)
    parser.add_argument("--gene2go-path", type=Path, default=default_gene2go_path())
    parser.add_argument(
        "--go-filter", action=argparse.BooleanOptionalAction, default=False,
        help="Restrict targets to GO-annotated genes for baseline-matched preprocessing.",
    )
    if allow_dataset_selection:
        parser.add_argument(
            "datasets",
            nargs="*",
            help="Dataset names or keys; names containing spaces/parentheses must be quoted",
        )
        parser.add_argument(
            "--dataset",
            dest="dataset_options",
            action="append",
            default=[],
            help="Repeatable alternative to positional dataset names",
        )
    ns = parser.parse_args()
    if allow_dataset_selection:
        requested = [*ns.datasets, *ns.dataset_options]
        if not requested:
            requested = list(default_datasets)
        ns.dataset_keys = resolve_single_dataset_keys(
            requested or ("replogle_rpe1", "replogle_k562")
        )
        print("Single datasets: " + ", ".join(DATASET_DISPLAY_NAMES[key] for key in ns.dataset_keys))
    ns.sources = parse_sources(ns.source)
    ns.config = PreprocessConfig(
        n_hvg=ns.n_hvg,
        min_pert_cells_exclusive=ns.min_pert_cells_exclusive,
        biogrid_path=ns.biogrid_path,
        gene2go_path=ns.gene2go_path,
        use_go_filter=ns.go_filter,
        max_cells_per_perturbation=(
            ns.max_cells_per_perturbation if ns.max_cells_per_perturbation > 0 else None
        ),
        max_control_cells=(ns.max_control_cells if ns.max_control_cells > 0 else None),
        hvg_max_cells_per_dataset=max(1, ns.hvg_max_cells_per_dataset),
        sampling_seed=ns.seed,
        min_free_memory_gib=max(0.0, ns.min_free_memory_gib),
    )
    path = BUILDERS[task](ns)
    print(json.dumps({"task": task, "manifest": project_path(path)}, ensure_ascii=False, indent=2))
