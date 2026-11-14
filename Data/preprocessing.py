from __future__ import annotations

import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from Data.annotation import (
    GeneNameMapper,
    build_pert_names,
    filter_perturbations_by_min_cells,
    filter_perturbations_in_go,
    default_gene2go_path,
    load_gene2go,
    normalize_total_log1p,
    split_pert_target_genes,
    standardize_condition_gene_names,
    standardize_var_names,
)
from Data.data_tools import auto_add_gpm_metadata
from GPM.paths import project_path


CONTROL = "ctrl"
DEFAULT_HVG = 2_000
DEFAULT_MIN_PERT_CELLS_EXCLUSIVE = 50
DEFAULT_TARGET_SUM = 10_000.0
DEFAULT_MAX_CELLS_PER_PERTURBATION = 128
DEFAULT_MAX_CONTROL_CELLS = 5_000
DATA_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_id: str
    cell_line: str
    technology: str
    default_source: Path
    perturbation_columns: tuple[str, ...] = ()
    batch_columns: tuple[str, ...] = ()
    control_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreprocessConfig:
    n_hvg: int = DEFAULT_HVG
    min_pert_cells_exclusive: int = DEFAULT_MIN_PERT_CELLS_EXCLUSIVE
    target_sum: float = DEFAULT_TARGET_SUM
    biogrid_path: Optional[Path] = None
    gene2go_path: Path = default_gene2go_path()
    use_go_filter: bool = False
    # Caps are applied independently per perturbation after the >50-cell
    # eligibility filter; controls use their own dataset-level cap.
    max_cells_per_perturbation: Optional[int] = DEFAULT_MAX_CELLS_PER_PERTURBATION
    max_control_cells: Optional[int] = DEFAULT_MAX_CONTROL_CELLS
    hvg_max_cells_per_dataset: int = 10_000
    sampling_seed: int = 42
    min_free_memory_gib: float = 4.0


def resolve_source(spec: DatasetSpec, override: Optional[Union[str, Path]] = None) -> Path:
    if override is not None:
        return Path(override)
    return spec.default_source


def first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    direct = {str(c): str(c) for c in frame.columns}
    folded = {str(c).casefold(): str(c) for c in frame.columns}
    for name in candidates:
        if name in direct:
            return direct[name]
        hit = folded.get(str(name).casefold())
        if hit is not None:
            return hit
    return None


def load_h5ad_dataset(
    spec: DatasetSpec,
    source: Optional[Union[str, Path]] = None,
    *,
    perturbation_column: Optional[str] = None,
    batch_column: Optional[str] = None,
) -> ad.AnnData:
    path = resolve_source(spec, source)
    if not path.is_file():
        raise FileNotFoundError(f"{spec.key}: missing h5ad: {path}")
    raw = sc.read_h5ad(path)
    pert_col = perturbation_column or first_column(raw.obs, spec.perturbation_columns)
    batch_col = batch_column or first_column(raw.obs, spec.batch_columns)
    if pert_col is None:
        raise ValueError(
            f"{spec.key}: cannot identify perturbation column; obs={list(raw.obs.columns)!r}"
        )
    return standardize_adapter_output(
        raw,
        spec,
        perturbation_column=pert_col,
        batch_column=batch_col,
        source=str(path),
    )


def standardize_adapter_output(
    raw: ad.AnnData,
    spec: DatasetSpec,
    *,
    perturbation_column: str,
    batch_column: Optional[str],
    source: str,
) -> ad.AnnData:
    source = project_path(Path(source))
    out = auto_add_gpm_metadata(
        raw,
        dataset_id=spec.dataset_id,
        data_display_name=spec.key,
        tech_name=spec.technology,
        cell_line_fixed=spec.cell_line,
        pert_col=perturbation_column,
        batch_col=batch_column,
        extra_control_substrings=spec.control_tokens,
        source_script=source,
        copy_matrix=False,
    )
    out.obs["condition"] = out.obs["condition"].astype(str)
    out.obs["gpm_condition"] = out.obs["condition"].astype(str)
    out.obs["gpm_dataset"] = spec.key
    out.obs["gpm_cell_line"] = spec.cell_line
    out.obs["gpm_perturbation_tech"] = spec.technology
    if "gpm_batch" not in out.obs:
        out.obs["gpm_batch"] = "batch0"
    batch = out.obs["gpm_batch"].fillna("batch0").astype(str).str.strip()
    out.obs["gpm_batch"] = batch.mask(batch.isin(("", "nan", "None")), "batch0")
    canonical_obs = [
        "condition", "gpm_condition", "gpm_is_control", "gpm_dataset",
        "gpm_cell_line", "gpm_perturbation_tech", "gpm_batch",
        "cell_line", "cell_type", "data_name", "tech_name",
    ]
    out.obs = out.obs[[column for column in canonical_obs if column in out.obs]].copy()
    out.obs_names = pd.Index(
        [f"{spec.key}:{x}" for x in out.obs_names.astype(str)],
        dtype=object,
        name=None,
    )
    out.obs_names_make_unique()
    out.var.index.name = None
    out.uns["gpm_source"] = source
    out.uns["gpm_dataset_spec"] = {
        "key": spec.key,
        "dataset_id": spec.dataset_id,
        "cell_line": spec.cell_line,
        "technology": spec.technology,
        "default_source": project_path(spec.default_source),
        "perturbation_columns": list(spec.perturbation_columns),
        "batch_columns": list(spec.batch_columns),
        "control_tokens": list(spec.control_tokens),
    }
    return out


def _matrix_values(x: Any, max_values: int = 200_000) -> np.ndarray:
    values = x.data if sp.issparse(x) else np.asarray(x).ravel()
    if values.size > max_values:
        step = max(1, values.size // max_values)
        values = values[::step][:max_values]
    return np.asarray(values, dtype=np.float64)


def choose_raw_count_matrix(adata: ad.AnnData) -> tuple[Any, str]:
    """Select a non-negative, integer-like count matrix without metadata guesses."""
    candidates: list[tuple[str, Any]] = []
    for key in ("counts", "count", "raw_counts", "UMI", "UMIs"):
        if key in adata.layers:
            candidates.append((f"layers/{key}", adata.layers[key]))
    candidates.append(("X", adata.X))
    if adata.raw is not None and adata.raw.n_vars == adata.n_vars:
        candidates.append(("raw/X", adata.raw.X))

    diagnostics: list[str] = []
    for name, matrix in candidates:
        values = _matrix_values(matrix)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            diagnostics.append(f"{name}: empty/non-finite")
            continue
        nonnegative = bool(np.min(finite) >= -1e-8)
        integer_like = bool(np.mean(np.isclose(finite, np.rint(finite), atol=1e-6)) >= 0.999)
        diagnostics.append(
            f"{name}: min={float(np.min(finite)):.4g}, max={float(np.max(finite)):.4g}, "
            f"integer_like={integer_like}"
        )
        if nonnegative and integer_like:
            return matrix, name
    raise ValueError(
        "No reliable raw UMI/count matrix found. " + "; ".join(diagnostics)
    )


def prepare_raw_count_matrix(adata: ad.AnnData) -> tuple[ad.AnnData, dict[str, Any]]:
    # Dataset adapters already return an owned AnnData.  Mutating it here avoids
    # keeping two full raw matrices alive at the largest point in preprocessing.
    out = adata
    counts, source = choose_raw_count_matrix(out)
    if counts is not out.X:
        out.X = counts.copy()
    out.layers.clear()
    out.raw = None
    totals = np.asarray(counts.sum(axis=1)).ravel()
    keep = np.isfinite(totals) & (totals > 0)
    removed = int((~keep).sum())
    if removed:
        out = out[keep].copy()
    return out, {
        "count_source": source,
        "n_zero_or_invalid_cells_removed": removed,
        "median_total_counts": float(np.median(totals[keep])) if keep.any() else 0.0,
    }


def _cap_cells_per_condition(
    adata: ad.AnnData,
    *,
    max_cells_per_perturbation: Optional[int],
    max_control_cells: Optional[int],
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Reproducibly random, batch-stratified condition caps before normalization."""
    pert_cap = None if not max_cells_per_perturbation else int(max_cells_per_perturbation)
    ctrl_cap = None if not max_control_cells else int(max_control_cells)
    if pert_cap is None and ctrl_cap is None:
        return adata, {"enabled": False, "n_cells_before": int(adata.n_obs), "n_cells_after": int(adata.n_obs)}

    conditions = adata.obs["condition"].astype(str).to_numpy()
    batches = (
        adata.obs["gpm_batch"].fillna("batch0").astype(str).to_numpy()
        if "gpm_batch" in adata.obs else np.full(adata.n_obs, "batch0", dtype=object)
    )
    keep: list[int] = []
    dataset_offset = sum((i + 1) * ord(c) for i, c in enumerate(str(adata.uns.get("gpm_dataset_spec", {}).get("key", "dataset"))))
    effective_seed = int(seed) + dataset_offset
    rng = np.random.default_rng(effective_seed)
    capped: dict[str, dict[str, int]] = {}
    for condition in sorted(set(conditions)):
        indices = np.flatnonzero(conditions == condition)
        cap = ctrl_cap if condition == CONTROL else pert_cap
        selected = indices
        if cap is not None and indices.size > cap:
            # Preserve small batches so later batch-matched control pairing does
            # not silently degrade to the dataset-level fallback.
            grouped = [
                indices[batches[indices] == batch]
                for batch in sorted(set(batches[indices]))
            ]
            grouped = [grouped[int(i)] for i in rng.permutation(len(grouped))]
            anchors: list[int] = []
            leftovers: list[np.ndarray] = []
            for group in grouped:
                shuffled = np.asarray(group, dtype=np.int64).copy()
                rng.shuffle(shuffled)
                if len(anchors) < cap:
                    anchors.append(int(shuffled[0]))
                    leftovers.append(shuffled[1:])
                else:
                    leftovers.append(shuffled)
            remaining = cap - len(anchors)
            pool = np.concatenate(leftovers) if leftovers else np.empty(0, dtype=np.int64)
            rng.shuffle(pool)
            selected = np.sort(np.asarray([*anchors, *pool[:remaining]], dtype=np.int64))
            capped[condition] = {"before": int(indices.size), "after": int(selected.size)}
        keep.extend(map(int, selected))
    keep_array = np.asarray(sorted(keep), dtype=np.int64)
    if np.unique(keep_array).size != keep_array.size:
        raise RuntimeError("Cell-cap sampling produced duplicate row indices")
    result = adata if keep_array.size == adata.n_obs else adata[keep_array].copy()

    # Fail immediately if a future refactor changes the cap semantics or loses
    # a batch even though the cap was large enough to preserve every batch.
    retained_conditions = result.obs["condition"].astype(str).to_numpy()
    retained_batches = result.obs["gpm_batch"].fillna("batch0").astype(str).to_numpy()
    for condition in sorted(set(conditions)):
        before_indices = np.flatnonzero(conditions == condition)
        condition_cap = ctrl_cap if condition == CONTROL else pert_cap
        expected = (
            before_indices.size
            if condition_cap is None else min(before_indices.size, condition_cap)
        )
        after_indices = np.flatnonzero(retained_conditions == condition)
        if after_indices.size != expected:
            raise RuntimeError(
                f"Cell-cap validation failed for {condition!r}: "
                f"expected {expected}, retained {after_indices.size}"
            )
        before_batch_set = set(batches[before_indices])
        if condition_cap is None or len(before_batch_set) <= condition_cap:
            after_batch_set = set(retained_batches[after_indices])
            missing_batches = before_batch_set - after_batch_set
            if missing_batches:
                raise RuntimeError(
                    f"Cell-cap sampling lost batches for {condition!r}: "
                    f"{sorted(missing_batches)}"
                )
    return result, {
        "enabled": True,
        "max_cells_per_perturbation": pert_cap,
        "max_control_cells": ctrl_cap,
        "sampling_seed": int(seed),
        "effective_dataset_seed": effective_seed,
        "sampling_strategy": "condition_then_batch_stratified",
        "randomized_selection": True,
        "row_order_after_sampling": "sorted_after_random_selection",
        "cap_validation_passed": True,
        "n_cells_before": int(adata.n_obs),
        "n_cells_after": int(result.n_obs),
        "n_cells_removed": int(adata.n_obs - result.n_obs),
        "conditions_capped": capped,
    }


def validate_working_counts(adata: ad.AnnData, *, stage: str) -> dict[str, Any]:
    """Fail early if a preprocessing operation has damaged the raw-count matrix."""
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(
            f"{stage}: count matrix is empty ({adata.n_obs} cells x {adata.n_vars} genes)"
        )
    values = _matrix_values(adata.X)
    if values.size == 0:
        raise ValueError(f"{stage}: count matrix contains no non-zero values")
    if not np.isfinite(values).all():
        raise ValueError(f"{stage}: count matrix contains NaN or infinite values")
    minimum = float(np.min(values))
    integer_fraction = float(np.mean(np.isclose(values, np.rint(values), atol=1e-6)))
    if minimum < -1e-8 or integer_fraction < 0.999:
        raise ValueError(
            f"{stage}: X is no longer a raw non-negative count matrix "
            f"(min={minimum:.4g}, integer_like_fraction={integer_fraction:.4f})"
        )
    return {
        "stage": stage,
        "matrix_location": "X",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "sampled_values": int(values.size),
        "minimum": minimum,
        "maximum": float(np.max(values)),
        "integer_like_fraction": integer_fraction,
    }


def condition_targets(condition: str) -> tuple[str, ...]:
    return tuple(sorted(split_pert_target_genes(str(condition)), key=str.upper))


def single_conditions(adata: ad.AnnData) -> list[str]:
    return sorted(
        c for c in adata.obs["condition"].astype(str).unique()
        if c != CONTROL and len(condition_targets(c)) == 1
    )


def all_perturbation_targets(adata: ad.AnnData) -> set[str]:
    targets: set[str] = set()
    for condition in adata.obs["condition"].astype(str).unique():
        targets.update(condition_targets(condition))
    return targets


def _filter_audit_stats(adata: ad.AnnData) -> dict[str, int]:
    """Return lightweight condition/cell counts for preprocessing audit logs.

    A "perturbation" here means one distinct non-control condition label.
    This helper is reporting-only and does not modify the AnnData object.
    """
    conditions = adata.obs["condition"].astype(str)
    control_mask = conditions == CONTROL
    perturbations = {str(value) for value in conditions[~control_mask].unique()}
    return {
        "n_perturbations": int(len(perturbations)),
        "n_perturbed_cells": int((~control_mask).sum()),
        "n_control_cells": int(control_mask.sum()),
        "n_cells": int(len(conditions)),
    }


def _print_filter_stage(
    dataset: str,
    stage: str,
    before: Mapping[str, int],
    after: Mapping[str, int],
    *,
    detail: str = "",
) -> None:
    """Print one compact filtering summary line."""
    dropped_perts = int(before["n_perturbations"] - after["n_perturbations"])
    suffix = f", {detail}" if detail else ""
    print(
        f"[filter][{dataset}] {stage}: "
        f"{before['n_perturbations']} -> {after['n_perturbations']} perturbations "
        f"(-{dropped_perts}), {before['n_cells']} -> {after['n_cells']} cells{suffix}",
        flush=True,
    )


def prepare_base_dataset(
    adata: ad.AnnData,
    spec: DatasetSpec,
    config: PreprocessConfig,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Prepare one dataset independently; HVG and cross-dataset subset are deferred."""
    report: dict[str, Any] = {"dataset": spec.key, "dataset_id": spec.dataset_id}
    for key in ("matrix_metadata_alignment", "adapter_filtering"):
        if key in adata.uns:
            report[key] = dict(adata.uns[key])
    out, count_report = prepare_raw_count_matrix(adata)
    # ``out`` may be the input object itself.  Drop the duplicate local
    # reference immediately so a later row materialisation can release the
    # original large matrix as soon as its views disappear.
    del adata
    report["counts"] = count_report
    adapter_stats = _filter_audit_stats(out)
    print(
        f"[filter][{spec.key}] start: {adapter_stats['n_perturbations']} perturbations, "
        f"{adapter_stats['n_cells']} cells "
        f"({adapter_stats['n_perturbed_cells']} perturbed + "
        f"{adapter_stats['n_control_cells']} ctrl)",
        flush=True,
    )

    mapper = GeneNameMapper.from_paths(biogrid_path=config.biogrid_path)
    out = standardize_var_names(
        out, mapper, duplicate_reducer="sum", copy_matrix=False
    )
    before_standardization = _filter_audit_stats(out)
    out, cond_report = standardize_condition_gene_names(
        out, mapper, copy_matrix=False
    )
    out.obs["gpm_condition"] = out.obs["condition"].astype(str)
    report["condition_standardization"] = cond_report
    after_standardization = _filter_audit_stats(out)
    if before_standardization["n_perturbations"] != after_standardization["n_perturbations"]:
        _print_filter_stage(
            spec.key,
            "condition names",
            before_standardization,
            after_standardization,
        )

    # BioGRID is a graph prior used by GPM, not an eligibility criterion for
    # expression features or perturbations.  Genes absent from BioGRID remain
    # valid model nodes (with no prior edges and a trainable identity fallback).
    report["ppi_filter"] = {
        "enabled": False,
        "policy": "graph_prior_only_no_gene_or_perturbation_filtering",
    }

    report["go_filter_enabled"] = config.use_go_filter
    if config.use_go_filter:
        before_go = _filter_audit_stats(out)
        gene2go = load_gene2go(config.gene2go_path)
        pert_names = build_pert_names(gene2go, out)
        out, go_report = filter_perturbations_in_go(
            out, pert_names, copy_result=False
        )
        report["go_filter"] = go_report
        after_go = _filter_audit_stats(out)
        _print_filter_stage(spec.key, "GO", before_go, after_go)
        del gene2go, pert_names
    else:
        current = _filter_audit_stats(out)
        print(
            f"[filter][{spec.key}] GO: disabled; "
            f"perturbations={current['n_perturbations']} unchanged",
            flush=True,
        )

    before_size = _filter_audit_stats(out)
    out, size_report = filter_perturbations_by_min_cells(
        out,
        min_cells_exclusive=config.min_pert_cells_exclusive,
        copy_result=False,
    )
    report["perturbation_size_filter"] = size_report
    after_size = _filter_audit_stats(out)
    _print_filter_stage(
        spec.key,
        f"min cells > {config.min_pert_cells_exclusive}",
        before_size,
        after_size,
    )
    if out.n_obs == 0 or (not single_conditions(out) and spec.key != "norman"):
        raise ValueError(f"{spec.key}: no usable perturbations after GO/count filters")
    if not (out.obs["condition"].astype(str) == CONTROL).any():
        raise ValueError(f"{spec.key}: no control cells after filtering")

    before_cap = _filter_audit_stats(out)
    out, cap_report = _cap_cells_per_condition(
        out,
        max_cells_per_perturbation=config.max_cells_per_perturbation,
        max_control_cells=config.max_control_cells,
        seed=config.sampling_seed,
    )
    report["memory_safety_cell_cap"] = cap_report
    after_cap = _filter_audit_stats(out)
    _print_filter_stage(spec.key, "cell cap", before_cap, after_cap)
    print(
        f"[filter][{spec.key}] final: {after_cap['n_perturbations']} perturbations, "
        f"{after_cap['n_cells']} cells "
        f"({after_cap['n_perturbed_cells']} perturbed + "
        f"{after_cap['n_control_cells']} ctrl)",
        flush=True,
    )

    # GO and minimum-cell filtering above intentionally return AnnData
    # views.  Materialise their combined row selection only once, after the
    # existing condition caps have selected the exact final cells.  This keeps
    # the scientific result identical while avoiding several simultaneous
    # copies of the full Replogle matrix.
    if out.is_view:
        out = out.copy()
    gc.collect()

    # Library-size normalisation is deliberately performed on the complete,
    # canonicalised gene universe before cross-dataset intersection/HVG subsetting.
    report["pre_normalization_count_validation"] = validate_working_counts(
        out,
        stage=f"{spec.key}/pre_normalization",
    )
    out = normalize_total_log1p(out, target_sum=config.target_sum, copy=False)
    out.uns["gpm_normalization"]["scope"] = "per_dataset_full_gene_universe"

    targets = all_perturbation_targets(out)
    out.var["gene_name"] = out.var_names.astype(str)
    report["after_base"] = dataset_summary(out)
    target_upper = {str(g).upper() for g in targets}
    panel_upper = {str(g).upper() for g in out.var_names}
    report["n_targets_in_panel"] = len(target_upper & panel_upper)
    return out, report


def _hvg_mask(
    adata: ad.AnnData,
    n_hvg: int,
    *,
    batch_key: Optional[str],
) -> np.ndarray:
    # Callers pass a disposable, already bounded HVG input.
    work = adata
    n_top = min(max(1, int(n_hvg)), work.n_vars)
    if work.n_vars <= n_top:
        return np.ones(work.n_vars, dtype=bool)
    try:
        sc.pp.highly_variable_genes(
            work,
            n_top_genes=n_top,
            batch_key=batch_key if batch_key in work.obs else None,
            subset=False,
            flavor="seurat",
        )
        return work.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
    except (ValueError, KeyError, TypeError):
        x = work.X
        if sp.issparse(x):
            mean = np.asarray(x.mean(axis=0)).ravel()
            variance = np.asarray(x.power(2).mean(axis=0)).ravel() - mean**2
        else:
            variance = np.var(np.asarray(x), axis=0)
        idx = np.argsort(-np.nan_to_num(variance), kind="stable")[:n_top]
        mask = np.zeros(work.n_vars, dtype=bool)
        mask[idx] = True
        return mask


def build_feature_panel(
    datasets: Mapping[str, ad.AnnData],
    *,
    n_hvg: int,
    fit_masks: Optional[Mapping[str, np.ndarray]] = None,
    include_all_targets: bool = True,
    max_fit_cells_per_dataset: int = 10_000,
    seed: int = 42,
) -> tuple[list[str], dict[str, Any]]:
    """Build one common panel after independent per-dataset normalisation."""
    if not datasets:
        raise ValueError("No datasets supplied")
    common_upper: Optional[set[str]] = None
    canonical: dict[str, str] = {}
    for dataset in datasets.values():
        current = {str(g).upper() for g in dataset.var_names}
        common_upper = current if common_upper is None else common_upper & current
        for gene in dataset.var_names.astype(str):
            canonical.setdefault(gene.upper(), gene)
    common_upper = common_upper or set()
    if not common_upper:
        raise ValueError("Cross-dataset gene intersection is empty")
    common = sorted(canonical[g] for g in common_upper)
    if len(common) < int(n_hvg):
        raise ValueError(
            f"Only {len(common)} common measured genes remain; cannot construct {n_hvg} HVGs"
        )

    pooled: list[ad.AnnData] = []
    fit_cell_counts: dict[str, dict[str, int]] = {}
    for key, dataset in datasets.items():
        upper_to_idx = {str(g).upper(): i for i, g in enumerate(dataset.var_names)}
        index = [upper_to_idx[g.upper()] for g in common]
        part = dataset[:, index]
        mask = None if fit_masks is None else fit_masks.get(key)
        if mask is not None:
            part = part[np.asarray(mask, dtype=bool)]
        before = int(part.n_obs)
        if before == 0:
            fit_cell_counts[key] = {"before_cap": 0, "used_for_hvg": 0}
            continue
        limit = max(1, int(max_fit_cells_per_dataset))
        if part.n_obs > limit:
            key_offset = sum((i + 1) * ord(c) for i, c in enumerate(str(key)))
            rng = np.random.default_rng(int(seed) + key_offset)
            rows = np.sort(rng.choice(part.n_obs, size=limit, replace=False))
            part = part[rows]
        part = part.copy()
        part.obs["gpm_hvg_batch"] = key + "|" + part.obs["gpm_batch"].astype(str)
        pooled.append(part)
        fit_cell_counts[key] = {"before_cap": before, "used_for_hvg": int(part.n_obs)}
    if not pooled:
        raise ValueError("HVG fit masks exclude every cell in every dataset")
    hvg_input = ad.concat(pooled, join="inner", merge="same", index_unique=None)
    pooled.clear()
    gc.collect()
    hvg = _hvg_mask(hvg_input, n_hvg, batch_key="gpm_hvg_batch")
    selected = {str(g).upper() for g, flag in zip(hvg_input.var_names, hvg) if flag}
    if len(selected) != int(n_hvg):
        raise RuntimeError(f"HVG selection returned {len(selected)} genes, expected {n_hvg}")

    targets: set[str] = set()
    if include_all_targets:
        for dataset in datasets.values():
            targets.update(g.upper() for g in all_perturbation_targets(dataset))
    missing_targets = targets - common_upper
    if missing_targets:
        raise ValueError(
            "Perturbation targets are absent from the common measured gene space: "
            + ", ".join(sorted(missing_targets)[:20])
        )
    target_in_intersection = targets & common_upper
    panel_upper = selected | target_in_intersection
    panel = [g for g in common if g.upper() in panel_upper]
    return panel, {
        "n_gene_intersection": len(common),
        "n_hvg_requested": int(n_hvg),
        "n_hvg_selected": len(selected),
        "n_target_genes_added": len(target_in_intersection - selected),
        "n_final_panel": len(panel),
        "hvg_selection_protocol": "target_control_cells_only",
        "normalization_before_intersection": True,
        "hvg_fit_uses_explicit_cell_masks": fit_masks is not None,
        "hvg_fit_cell_cap_per_dataset": int(max_fit_cells_per_dataset),
        "hvg_fit_cells": fit_cell_counts,
    }


def subset_to_panel(adata: ad.AnnData, panel: Sequence[str]) -> ad.AnnData:
    upper_to_idx = {str(g).upper(): i for i, g in enumerate(adata.var_names)}
    missing = [g for g in panel if g.upper() not in upper_to_idx]
    if missing:
        raise ValueError(f"Dataset missing {len(missing)} panel genes: {missing[:5]}")
    index = [upper_to_idx[str(g).upper()] for g in panel]
    out = adata[:, index].copy()
    out.var_names = pd.Index([str(g) for g in panel])
    out.var["gene_name"] = out.var_names.astype(str)
    return out


def dataset_summary(adata: ad.AnnData) -> dict[str, Any]:
    cond = adata.obs["condition"].astype(str)
    counts = cond.value_counts()
    perts = counts.drop(CONTROL, errors="ignore")
    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_control_cells": int((cond == CONTROL).sum()),
        "n_perturbed_cells": int((cond != CONTROL).sum()),
        "n_perturbations": int(perts.size),
        "perturbation_cell_counts": {
            str(condition): int(count) for condition, count in perts.sort_index().items()
        },
        "perturbation_cells": {
            "min": int(perts.min()) if not perts.empty else 0,
            "median": float(perts.median()) if not perts.empty else 0.0,
            "max": int(perts.max()) if not perts.empty else 0,
        },
        "n_batches": int(adata.obs["gpm_batch"].nunique()),
        "batch_cells": {
            str(batch): int(count)
            for batch, count in adata.obs["gpm_batch"].astype(str).value_counts().sort_index().items()
        },
        "batch_control_cells": {
            str(batch): int(count)
            for batch, count in adata.obs.loc[cond == CONTROL, "gpm_batch"].astype(str).value_counts().sort_index().items()
        },
        "split_cells": {
            str(k): int(v) for k, v in adata.obs.get(
                "split", pd.Series("unassigned", index=adata.obs_names)
            ).astype(str).value_counts().items()
        },
    }
    if int((cond == CONTROL).sum()) > 0:
        global_mean, by_batch = control_means(adata)
        shifts: dict[str, Any] = {}
        for batch, mean in by_batch.items():
            rms = float(np.sqrt(np.mean((mean - global_mean) ** 2)))
            correlation = (
                float(np.corrcoef(mean, global_mean)[0, 1])
                if np.std(mean) > 1e-12 and np.std(global_mean) > 1e-12
                else float("nan")
            )
            shifts[batch] = {"rms_from_dataset_control_mean": rms, "pearson": correlation}
        summary["control_batch_shift"] = shifts
    return summary


def control_means(adata: ad.AnnData) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ctrl = adata.obs["condition"].astype(str) == CONTROL
    if not ctrl.any():
        raise ValueError("Dataset has no controls")
    global_mean = np.asarray(adata[ctrl].X.mean(axis=0)).ravel().astype(np.float32)
    by_batch: dict[str, np.ndarray] = {}
    for batch in sorted(adata.obs.loc[ctrl, "gpm_batch"].astype(str).unique()):
        mask = ctrl & (adata.obs["gpm_batch"].astype(str) == batch)
        by_batch[batch] = np.asarray(adata[mask].X.mean(axis=0)).ravel().astype(np.float32)
    return global_mean, by_batch


def write_dataset_artifacts(
    adata: ad.AnnData,
    *,
    dataset_key: str,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = dataset_key.lower().replace(" ", "_").replace("(", "_").replace(")", "")
    h5ad_path = output_dir / f"{slug}.h5ad"
    ctrl_path = output_dir / f"{slug}_control_means.npz"
    adata.write_h5ad(h5ad_path, compression="gzip")
    global_mean, by_batch = control_means(adata)
    arrays: dict[str, Any] = {
        "gene_names": np.asarray(adata.var_names.astype(str), dtype=str),
        "global_mean": global_mean,
        "mean": global_mean,
        "dataset_key": np.asarray(dataset_key),
        "batch_names": np.asarray(list(by_batch), dtype=str),
    }
    for i, batch in enumerate(by_batch):
        arrays[f"batch_mean_{i}"] = by_batch[batch]
    np.savez_compressed(ctrl_path, **arrays)
    return {"h5ad": h5ad_path.name, "control_means": ctrl_path.name}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
