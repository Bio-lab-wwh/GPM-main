"""Export test-set differential-expression truth for all six GPM tasks.

The script reads the manifests produced by ``Division`` and processes each
ready fold independently.  Within a fold, only datasets that actually contain
test perturbation cells are evaluated.  This is important for cross-dataset
tasks: cells and controls from different datasets are never pooled.

For every test perturbation, its observed cells are compared with all observed
control cells from the same processed dataset using a two-sided asymptotic
Mann--Whitney U test (Wilcoxon rank-sum, tie corrected, no continuity
correction).  Benjamini--Hochberg correction is applied independently for each
perturbation across the complete fold gene panel.  The default DE definition is
strictly ``q < 0.05``.

The processed H5AD files already contain Division's normalize-total/log1p
expression.  This script deliberately does not transform them again.

Outputs for every task/fold/test-dataset combination
-----------------------------------------------------
``*_test_de.npz``
    Complete gene-aligned matrices, including raw P values, BH q values,
    observed mean effects and a boolean DE mask.
``*_test_de_significant.csv``
    Long table containing only genes passing the requested FDR threshold.
``*_test_de_summary.json``
    Human-readable provenance and DE-gene lists per perturbation.

A top-level ``index.json`` records every generated or reused artifact.

Examples
--------
Run all six tasks using the default Division output directories::
    python DE_Genes/save_test_de_genes.py

Run or rebuild selected tasks::

    python DE_Genes/save_test_de_genes.py --task unseen_cell --task zero_patient
    python DE_Genes/save_test_de_genes.py --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.stats import mannwhitneyu


TASKS: tuple[str, ...] = (
    "single_unseen_pert",
    "double_unseen_pert",
    "unseen_cell",
    "unseen_pert_cell",
    "across_tech",
    "zero_patient",
)

DEFAULT_FDR_THRESHOLD = 0.05
DEFAULT_CHUNK_ROWS = 1024
DEFAULT_MAX_RANK_ELEMENTS = 10_000_000
CONTROL_CONDITION = "ctrl"

SCRIPT_DIR = Path(__file__).resolve().parent
GPM_ROOT = SCRIPT_DIR.parent
DEFAULT_DIVISION_ROOT = GPM_ROOT / "Division" / "outputs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results"

if __package__ in (None, ""):
    sys.path.insert(0, str(GPM_ROOT))

from GPM.paths import manifest_artifact_path, project_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return result or "item"


def _resolve_artifact(value: Any, manifest_path: Path) -> Path:
    return manifest_artifact_path(value, manifest_path)


def _panel_columns(data: ad.AnnData, panel: Sequence[str], *, label: str) -> np.ndarray:
    observed = data.var_names.astype(str).tolist()
    lookup: dict[str, int] = {}
    for index, gene in enumerate(observed):
        key = gene.upper()
        if key in lookup:
            raise ValueError(f"{label}: H5AD contains duplicate gene symbol {gene!r}")
        lookup[key] = index
    missing = [str(gene) for gene in panel if str(gene).upper() not in lookup]
    if missing:
        raise ValueError(
            f"{label}: {len(missing)} fold genes are absent from H5AD: {missing[:5]}"
        )
    return np.asarray([lookup[str(gene).upper()] for gene in panel], dtype=np.int64)


def _dense_rows(
    data: ad.AnnData,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    chunk_rows: int,
) -> np.ndarray:
    """Read selected backed AnnData rows and ordered columns into float32 RAM.

    HDF5 permits only one array-valued fancy index in a single operation.  In
    particular, older AnnData/h5py versions reject ``data[row_ids, col_ids]``.
    Read each row block with a slice over all stored columns first, then apply
    the fold-panel column index to the resulting in-memory dense/sparse block.
    """
    row_indices = np.asarray(rows, dtype=np.int64).reshape(-1)
    if row_indices.size == 0:
        return np.empty((0, columns.size), dtype=np.float32)
    result = np.empty((row_indices.size, columns.size), dtype=np.float32)
    source = data.X
    step = max(1, int(chunk_rows))
    for start in range(0, row_indices.size, step):
        stop = min(start + step, row_indices.size)
        # Only the row axis uses fancy indexing against the backed HDF5
        # dataset.  Column reordering happens after the block is in memory.
        block = source[row_indices[start:stop], :]
        if hasattr(block, "to_memory"):
            block = block.to_memory()
        if sparse.issparse(block):
            values = np.asarray(block[:, columns].toarray(), dtype=np.float32)
        else:
            values = np.asarray(block, dtype=np.float32)[:, columns]
        if values.shape != (stop - start, columns.size):
            raise ValueError(
                "Unexpected expression block shape: "
                f"{values.shape} != {(stop - start, columns.size)}"
            )
        result[start:stop] = values
    return result


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    finite = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 1.0)
    order = np.argsort(finite, kind="mergesort")
    ranks = np.arange(1, finite.size + 1, dtype=np.float64)
    ranked = finite[order] * (float(finite.size) / ranks)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _wilcoxon_rank_sum(
    perturbed: np.ndarray,
    controls: np.ndarray,
    *,
    max_rank_elements: int,
) -> np.ndarray:
    """Return two-sided, tie-corrected asymptotic rank-sum P values."""
    if perturbed.ndim != 2 or controls.ndim != 2:
        raise ValueError("Perturbed and control expression must be two-dimensional")
    if perturbed.shape[1] != controls.shape[1]:
        raise ValueError("Perturbed and control expression have different gene counts")
    if perturbed.shape[0] == 0 or controls.shape[0] == 0:
        raise ValueError("Wilcoxon comparison requires non-empty groups")
    if max_rank_elements < 1:
        raise ValueError("max_rank_elements must be positive")

    rows_per_gene = int(perturbed.shape[0] + controls.shape[0])
    block_size = max(1, int(max_rank_elements) // max(1, rows_per_gene))
    block_size = min(block_size, perturbed.shape[1])
    p_values = np.ones(perturbed.shape[1], dtype=np.float64)
    for start in range(0, perturbed.shape[1], block_size):
        stop = min(start + block_size, perturbed.shape[1])
        result = mannwhitneyu(
            perturbed[:, start:stop],
            controls[:, start:stop],
            use_continuity=False,
            alternative="two-sided",
            axis=0,
            method="asymptotic",
        )
        raw = np.asarray(result.pvalue, dtype=np.float64).reshape(-1)
        p_values[start:stop] = np.where(
            np.isfinite(raw), np.clip(raw, 0.0, 1.0), 1.0
        )
    return p_values


def _condition_groups(
    data: ad.AnnData,
    test_rows: np.ndarray,
    perturbations: Sequence[str],
) -> np.ndarray:
    """Return one optional double-perturbation evaluation group per condition."""
    if "evaluation_group" not in data.obs:
        return np.full(len(perturbations), "", dtype=str)
    conditions = data.obs["condition"].astype(str).to_numpy()
    groups = data.obs["evaluation_group"].astype(str).to_numpy()
    test_mask = np.zeros(data.n_obs, dtype=bool)
    test_mask[np.asarray(test_rows, dtype=np.int64)] = True
    result: list[str] = []
    for condition in perturbations:
        values = sorted(set(groups[test_mask & (conditions == condition)]))
        if len(values) > 1:
            raise ValueError(
                f"Condition {condition!r} has multiple evaluation groups: {values}"
            )
        result.append(values[0] if values else "")
    return np.asarray(result, dtype=str)


def _signature(
    *,
    task: str,
    fold: str,
    dataset: str,
    manifest_path: Path,
    h5ad_path: Path,
    split_column: str,
    panel: Sequence[str],
    perturbations: Sequence[str],
    fdr_threshold: float,
) -> str:
    manifest_stat = manifest_path.stat()
    h5ad_stat = h5ad_path.stat()
    payload = {
        "task": task,
        "fold": fold,
        "dataset": dataset,
        "manifest": project_path(manifest_path),
        "manifest_size": int(manifest_stat.st_size),
        "manifest_mtime_ns": int(manifest_stat.st_mtime_ns),
        "h5ad": project_path(h5ad_path),
        "h5ad_size": int(h5ad_stat.st_size),
        "h5ad_mtime_ns": int(h5ad_stat.st_mtime_ns),
        "split_column": split_column,
        "panel": list(map(str, panel)),
        "test_perturbations": list(map(str, perturbations)),
        "fdr_threshold": float(fdr_threshold),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _npz_is_current(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            return str(np.asarray(saved["signature"]).item()) == signature
    except (KeyError, OSError, ValueError):
        return False


def _write_npz_atomic(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_significant_csv(
    path: Path,
    *,
    task: str,
    fold: str,
    dataset: str,
    gene_names: np.ndarray,
    perturbations: np.ndarray,
    evaluation_groups: np.ndarray,
    mean_effect: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
    de_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = (
        "task",
        "fold",
        "dataset",
        "perturbation",
        "evaluation_group",
        "gene",
        "mean_effect",
        "p_value",
        "q_value",
    )
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row, perturbation in enumerate(perturbations.astype(str)):
                selected = np.flatnonzero(de_mask[row])
                order = selected[
                    np.argsort(q_values[row, selected], kind="mergesort")
                ]
                for column in order:
                    writer.writerow({
                        "task": task,
                        "fold": fold,
                        "dataset": dataset,
                        "perturbation": perturbation,
                        "evaluation_group": str(evaluation_groups[row]),
                        "gene": str(gene_names[column]),
                        "mean_effect": float(mean_effect[row, column]),
                        "p_value": float(p_values[row, column]),
                        "q_value": float(q_values[row, column]),
                    })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_existing_entry(
    summary_path: Path,
    *,
    npz_path: Path,
    csv_path: Path,
    signature: str,
) -> dict[str, Any] | None:
    if not (
        _npz_is_current(npz_path, signature)
        and csv_path.is_file()
        and summary_path.is_file()
    ):
        return None
    try:
        summary = _read_json(summary_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if summary.get("signature") != signature:
        return None
    return summary


def _print_de_coverage(
    *,
    task: str,
    fold: str,
    dataset: str,
    coverage: Mapping[str, Any],
    fdr_threshold: float,
) -> None:
    """Print the test-set DE coverage used to select later analyses."""
    print(
        f"[DE summary] {task}/{fold}/{dataset} | "
        f"Test perturbations={int(coverage['n_test_perturbations'])} | "
        f"Perturbations with DE genes={int(coverage['n_perturbations_with_de'])} | "
        f"Perturbations with at least 10 DE genes={int(coverage['n_perturbations_with_at_least_10_de_genes'])} | "
        f"Perturbations with at least 20 DE genes={int(coverage['n_perturbations_with_at_least_20_de_genes'])} | "
        f"DE criterion: q < {fdr_threshold:g}",
        flush=True,
    )


def _compute_one_dataset(
    *,
    task: str,
    fold: Mapping[str, Any],
    dataset: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    output_root: Path,
    fdr_threshold: float,
    chunk_rows: int,
    max_rank_elements: int,
    force: bool,
) -> dict[str, Any] | None:
    fold_name = str(fold["name"])
    split_column = str(fold["split_column"])
    gene_names = np.asarray([str(gene) for gene in fold["gene_panel"]], dtype=str)
    if gene_names.size == 0 or len(set(gene_names.astype(str))) != gene_names.size:
        raise ValueError(f"{task}/{fold_name}: fold gene panel is empty or duplicated")

    metadata = manifest["datasets"][dataset]
    h5ad_path = _resolve_artifact(metadata["h5ad"], manifest_path)
    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)

    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if split_column not in data.obs:
            raise KeyError(f"{h5ad_path} has no obs[{split_column!r}]")
        if "condition" not in data.obs:
            raise KeyError(f"{h5ad_path} has no obs['condition']")
        splits = data.obs[split_column].astype(str).to_numpy()
        conditions = data.obs["condition"].astype(str).to_numpy()
        test_rows = np.flatnonzero(
            (splits == "test") & (conditions != CONTROL_CONDITION)
        )
        if test_rows.size == 0:
            return None
        perturbations = np.asarray(
            sorted(set(conditions[test_rows].astype(str))), dtype=str
        )
        control_rows = np.flatnonzero(conditions == CONTROL_CONDITION)
        if control_rows.size == 0:
            raise ValueError(f"{task}/{fold_name}/{dataset}: no control cells")
        evaluation_groups = _condition_groups(
            data, test_rows, perturbations.astype(str).tolist()
        )

        signature = _signature(
            task=task,
            fold=fold_name,
            dataset=dataset,
            manifest_path=manifest_path,
            h5ad_path=h5ad_path,
            split_column=split_column,
            panel=gene_names.astype(str).tolist(),
            perturbations=perturbations.astype(str).tolist(),
            fdr_threshold=fdr_threshold,
        )
        fold_dir = output_root / task / _safe_slug(fold_name)
        stem = f"{_safe_slug(dataset)}_test_de"
        npz_path = fold_dir / f"{stem}.npz"
        csv_path = fold_dir / f"{stem}_significant.csv"
        summary_path = fold_dir / f"{stem}_summary.json"
        if not force:
            existing = _load_existing_entry(
                summary_path,
                npz_path=npz_path,
                csv_path=csv_path,
                signature=signature,
            )
            if existing is not None:
                print(
                    f"[reuse] {task}/{fold_name}/{dataset}: {npz_path}",
                    flush=True,
                )
                _print_de_coverage(
                    task=task,
                    fold=fold_name,
                    dataset=dataset,
                    coverage=existing["de_coverage"],
                    fdr_threshold=fdr_threshold,
                )
                return existing

        columns = _panel_columns(
            data, gene_names.astype(str).tolist(),
            label=f"{task}/{fold_name}/{dataset}",
        )
        controls = _dense_rows(
            data, control_rows, columns, chunk_rows=chunk_rows
        )
        control_mean = controls.mean(axis=0, dtype=np.float64)
        n_perturbations = perturbations.size
        n_genes = gene_names.size
        mean_perturbed = np.empty((n_perturbations, n_genes), dtype=np.float64)
        mean_effect = np.empty_like(mean_perturbed)
        p_values = np.ones_like(mean_perturbed)
        q_values = np.ones_like(mean_perturbed)
        n_perturbed_cells = np.empty(n_perturbations, dtype=np.int64)

        print(
            f"[DE] {task}/{fold_name}/{dataset}: "
            f"{n_perturbations} test perturbations, {n_genes} genes, "
            f"{control_rows.size} same-dataset controls",
            flush=True,
        )
        for row, condition in enumerate(perturbations.astype(str)):
            condition_rows = np.flatnonzero(
                (splits == "test") & (conditions == condition)
            )
            perturbed = _dense_rows(
                data, condition_rows, columns, chunk_rows=chunk_rows
            )
            n_perturbed_cells[row] = int(perturbed.shape[0])
            mean_perturbed[row] = perturbed.mean(axis=0, dtype=np.float64)
            mean_effect[row] = mean_perturbed[row] - control_mean
            p_values[row] = _wilcoxon_rank_sum(
                perturbed,
                controls,
                max_rank_elements=max_rank_elements,
            )
            q_values[row] = _benjamini_hochberg(p_values[row])
            print(
                f"  [{row + 1}/{n_perturbations}] {condition}: "
                f"cells={condition_rows.size}, "
                f"DE(q<{fdr_threshold:g})={int(np.sum(q_values[row] < fdr_threshold))}",
                flush=True,
            )

        de_mask = np.isfinite(q_values) & (q_values < float(fdr_threshold))
        n_de_genes_per_perturbation = np.sum(
            de_mask, axis=1, dtype=np.int64
        )
        de_coverage = {
            "n_test_perturbations": int(n_perturbations),
            "n_perturbations_with_de": int(
                np.sum(n_de_genes_per_perturbation > 0)
            ),
            "n_perturbations_with_at_least_10_de_genes": int(
                np.sum(n_de_genes_per_perturbation >= 10)
            ),
            "n_perturbations_with_at_least_20_de_genes": int(
                np.sum(n_de_genes_per_perturbation >= 20)
            ),
        }
        _print_de_coverage(
            task=task,
            fold=fold_name,
            dataset=dataset,
            coverage=de_coverage,
            fdr_threshold=fdr_threshold,
        )
        _write_npz_atomic(npz_path, {
            "signature": np.asarray(signature),
            "test_alternative": np.asarray("two-sided"),
            "multiple_testing": np.asarray("Benjamini-Hochberg per perturbation across fold gene panel"),
            "fdr_threshold": np.asarray(fdr_threshold, dtype=np.float64),
            "task": np.asarray(task),
            "fold": np.asarray(fold_name),
            "dataset": np.asarray(dataset),
            "split_column": np.asarray(split_column),
            "split_value": np.asarray("test"),
            "control_condition": np.asarray(CONTROL_CONDITION),
            "control_scope": np.asarray("all control cells from the same processed dataset"),
            "expression_source": np.asarray("Division processed H5AD X; no additional transform"),
            "gene_names": gene_names,
            "perturbations": perturbations,
            "evaluation_groups": evaluation_groups,
            "mean_control_expression": control_mean,
            "mean_perturbed_expression": mean_perturbed,
            "mean_effect": mean_effect,
            "p_values": p_values,
            "q_values": q_values,
            "is_de": de_mask,
            "n_de_genes_per_perturbation": n_de_genes_per_perturbation,
            "n_perturbed_cells": n_perturbed_cells,
            "n_control_cells": np.asarray(control_rows.size, dtype=np.int64),
        })
        _write_significant_csv(
            csv_path,
            task=task,
            fold=fold_name,
            dataset=dataset,
            gene_names=gene_names,
            perturbations=perturbations,
            evaluation_groups=evaluation_groups,
            mean_effect=mean_effect,
            p_values=p_values,
            q_values=q_values,
            de_mask=de_mask,
        )

        perturbation_summaries = []
        for row, condition in enumerate(perturbations.astype(str)):
            selected = np.flatnonzero(de_mask[row])
            order = selected[
                np.argsort(q_values[row, selected], kind="mergesort")
            ]
            perturbation_summaries.append({
                "perturbation": condition,
                "evaluation_group": str(evaluation_groups[row]),
                "n_perturbed_cells": int(n_perturbed_cells[row]),
                "n_de_genes": int(selected.size),
                "de_genes": [str(gene_names[index]) for index in order],
            })
        summary: dict[str, Any] = {
            "signature": signature,
            "task": task,
            "fold": fold_name,
            "dataset": dataset,
            "manifest": project_path(manifest_path),
            "h5ad": project_path(h5ad_path),
            "split_column": split_column,
            "split_value": "test",
            "method": {
                "test": "two-sided Wilcoxon rank-sum / asymptotic Mann-Whitney U",
                "tie_correction": True,
                "continuity_correction": False,
                "multiple_testing": "Benjamini-Hochberg independently per perturbation across the complete fold gene panel",
                "fdr_threshold": float(fdr_threshold),
                "de_rule": f"q < {fdr_threshold:g}",
                "effect_size_cutoff": None,
                "expression": "Division processed H5AD X; no additional transform",
                "control_scope": "all control cells from the same processed dataset; datasets are never pooled",
            },
            "n_genes": int(n_genes),
            "n_test_perturbations": int(n_perturbations),
            "n_test_perturbed_cells": int(n_perturbed_cells.sum()),
            "n_control_cells": int(control_rows.size),
            "n_de_gene_pairs": int(de_mask.sum()),
            "de_coverage": de_coverage,
            "npz": project_path(npz_path),
            "significant_csv": project_path(csv_path),
            "summary_json": project_path(summary_path),
            "perturbations": perturbation_summaries,
        }
        _write_json_atomic(summary_path, summary)
        return summary
    finally:
        if getattr(data, "file", None) is not None:
            data.file.close()


def _process_task(
    *,
    task: str,
    manifest_path: Path,
    output_root: Path,
    fdr_threshold: float,
    chunk_rows: int,
    max_rank_elements: int,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    manifest = _read_json(manifest_path)
    if manifest.get("task") != task:
        raise ValueError(
            f"{manifest_path}: manifest task {manifest.get('task')!r} != {task!r}"
        )
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for fold in manifest.get("folds", []):
        fold_name = str(fold.get("name", "unknown"))
        if fold.get("status") == "skipped":
            skipped.append({
                "task": task,
                "fold": fold_name,
                "reason": str(fold.get("reason", "fold marked skipped")),
            })
            print(f"[skip fold] {task}/{fold_name}: {fold.get('reason')}", flush=True)
            continue
        for dataset in map(str, fold.get("dataset_keys", [])):
            if dataset not in manifest.get("datasets", {}):
                raise KeyError(f"{task}/{fold_name}: manifest has no dataset {dataset!r}")
            result = _compute_one_dataset(
                task=task,
                fold=fold,
                dataset=dataset,
                manifest=manifest,
                manifest_path=manifest_path,
                output_root=output_root,
                fdr_threshold=fdr_threshold,
                chunk_rows=chunk_rows,
                max_rank_elements=max_rank_elements,
                force=force,
            )
            if result is None:
                skipped.append({
                    "task": task,
                    "fold": fold_name,
                    "dataset": dataset,
                    "reason": "dataset has no perturbed test cells in this fold",
                })
                continue
            results.append(result)
    return results, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save test-only Wilcoxon/BH-FDR DE genes for the six GPM tasks"
    )
    parser.add_argument(
        "--division-root",
        type=Path,
        default=DEFAULT_DIVISION_ROOT,
        help="Directory containing <task>/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for DE result artifacts",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        default=[],
        help="Task to process; repeatable. Omit to process all six tasks.",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=DEFAULT_FDR_THRESHOLD,
        help="Strict BH q-value cutoff used for is_de and significant CSV",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="Rows read from backed H5AD per expression block",
    )
    parser.add_argument(
        "--max-rank-elements",
        type=int,
        default=DEFAULT_MAX_RANK_ELEMENTS,
        help="Approximate maximum observations in one rank-test block",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even when their input signature is current",
    )
    parser.add_argument(
        "--skip-missing-manifests",
        action="store_true",
        help="Process available tasks instead of failing when a manifest is absent",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a task error and continue with the remaining tasks",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not np.isfinite(args.fdr_threshold) or not 0.0 < args.fdr_threshold < 1.0:
        raise ValueError("--fdr-threshold must lie strictly between 0 and 1")
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    if args.max_rank_elements < 1:
        raise ValueError("--max-rank-elements must be positive")

    tasks = tuple(dict.fromkeys(args.task or TASKS))
    manifest_paths = {
        task: args.division_root / task / "manifest.json" for task in tasks
    }
    missing = {
        task: path for task, path in manifest_paths.items() if not path.is_file()
    }
    if missing and not args.skip_missing_manifests:
        formatted = "\n".join(f"  {task}: {path}" for task, path in missing.items())
        raise FileNotFoundError(
            "Missing Division manifests. Run the corresponding Division tasks first "
            "or pass --skip-missing-manifests:\n" + formatted
        )

    all_results: list[dict[str, Any]] = []
    all_skipped: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for task in tasks:
        manifest_path = manifest_paths[task]
        if not manifest_path.is_file():
            entry = {
                "task": task,
                "reason": f"missing manifest: {manifest_path}",
            }
            all_skipped.append(entry)
            print(f"[skip task] {task}: {entry['reason']}", flush=True)
            continue
        print(f"\n========== {task} ==========", flush=True)
        try:
            results, skipped = _process_task(
                task=task,
                manifest_path=manifest_path,
                output_root=args.output_dir,
                fdr_threshold=float(args.fdr_threshold),
                chunk_rows=int(args.chunk_rows),
                max_rank_elements=int(args.max_rank_elements),
                force=bool(args.force),
            )
            all_results.extend(results)
            all_skipped.extend(skipped)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            entry = {
                "task": task,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(entry)
            print(f"[task error] {task}: {type(exc).__name__}: {exc}", flush=True)

    index = {
        "fdr_threshold": float(args.fdr_threshold),
        "division_root": project_path(args.division_root),
        "output_dir": project_path(args.output_dir),
        "tasks_requested": list(tasks),
        "n_results": len(all_results),
        "results": all_results,
        "skipped": all_skipped,
        "errors": errors,
    }
    index_path = args.output_dir / "index.json"
    _write_json_atomic(index_path, index)
    print(
        f"\nSaved DE index: {project_path(index_path)} "
        f"(results={len(all_results)}, skipped={len(all_skipped)}, errors={len(errors)})",
        flush=True,
    )
    if errors:
        raise RuntimeError(
            f"{len(errors)} task(s) failed; see {project_path(index_path)} for details"
        )


if __name__ == "__main__":
    main()
