from __future__ import annotations

import gzip
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from Data.annotation import GeneNameMapper
from Data.preprocessing import (
    DatasetSpec,
    DATA_DIR,
    first_column,
    resolve_source,
    standardize_adapter_output,
)


SPEC = DatasetSpec(
    key="yao",
    dataset_id="Yao2023_THP1_conventional_KO",
    cell_line="THP-1",
    technology="KO",
    default_source=DATA_DIR / "Yao",
    perturbation_columns=("gpm_raw_condition",),
    batch_columns=("batch", "batch_id", "replicate", "library", "sample"),
    control_tokens=(
        "non-targeting", "non_targeting", "nontargeting", "negative_control",
    ),
)

_H5AD = "GSM6858447_KO_conventional.h5ad.gz"
_PERTURBATIONS = "GSM6858447_KO_conventional_perturbations.txt.gz"
_MIN_ALIGNMENT_FRACTION = 0.95
_POSITIONAL_ALIGNMENT_FRACTION = 0.99
_MIN_PPI_GENES_AFTER_NAME_RESOLUTION = 2_000
_MIN_TARGETS_IN_EXPRESSION_PANEL = 50
_ENSEMBL_TOKEN = re.compile(r"ENSG\d+(?:\.\d+)?", re.IGNORECASE)
_H5AD_RESERVED_VAR_COLUMNS = frozenset({"_index"})


def _clean_barcode(value: object) -> str:
    return str(value).strip().strip('"').strip("'")


def _canonical_barcode(value: object) -> str:
    barcode = _clean_barcode(value)
    head, separator, suffix = barcode.rpartition("-")
    return head if separator and suffix.isdigit() else barcode


def _read_compressed_h5ad(path: Path) -> ad.AnnData:
    if not str(path).endswith(".gz"):
        return sc.read_h5ad(path)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as handle:
            temporary = Path(handle.name)
            with gzip.open(path, "rb") as source:
                shutil.copyfileobj(source, handle, length=16 * 1024 * 1024)
        return sc.read_h5ad(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _is_control_target(value: object) -> bool:
    token = str(value).strip().casefold().replace(" ", "_")
    return token in {"ctrl", "control", "nt", "ntc", "negative_control"} or any(
        marker in token
        for marker in ("non-target", "non_target", "nontarget", "safe-target", "safe_target")
    )


def _canonical_target(value: object, mapper: GeneNameMapper) -> str:
    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        return raw
    resolved = str(mapper.resolve(raw)).strip()
    return resolved or raw


def _build_design_to_obs(
    design_cells: list[str], obs_names: pd.Index
) -> tuple[np.ndarray, dict[str, object]]:
    """Map every perturbation-table column position to one H5AD row position."""
    obs_cells = [_clean_barcode(value) for value in obs_names.astype(str)]
    design_canonical = [_canonical_barcode(value) for value in design_cells]
    obs_canonical = [_canonical_barcode(value) for value in obs_cells]

    positional_fraction = 0.0
    if len(design_cells) == len(obs_cells) and design_cells:
        positional_fraction = float(np.mean(np.fromiter(
            (a == b for a, b in zip(design_canonical, obs_canonical)),
            dtype=bool,
            count=len(design_cells),
        )))
        if positional_fraction >= _POSITIONAL_ALIGNMENT_FRACTION:
            return np.arange(len(design_cells), dtype=np.int64), {
                "alignment": "published_position",
                "positional_barcode_match_fraction": positional_fraction,
                "n_matched_cells": int(len(obs_cells)),
                "n_unmatched_h5ad_cells_removed": 0,
                "n_unmatched_design_columns": 0,
            }

    design_positions: dict[str, list[int]] = defaultdict(list)
    for position, key in enumerate(design_canonical):
        design_positions[key].append(position)

    cursors: dict[str, int] = defaultdict(int)
    mapping = np.full(len(design_cells), -1, dtype=np.int64)
    matched_obs = np.zeros(len(obs_cells), dtype=bool)
    for obs_position, key in enumerate(obs_canonical):
        occurrence = cursors[key]
        positions = design_positions.get(key)
        if positions is not None and occurrence < len(positions):
            design_position = positions[occurrence]
            mapping[design_position] = obs_position
            matched_obs[obs_position] = True
            cursors[key] = occurrence + 1

    matched = int(matched_obs.sum())
    match_fraction = matched / max(1, len(obs_cells))
    if match_fraction < _MIN_ALIGNMENT_FRACTION:
        raise ValueError(
            "Yao perturbation/H5AD alignment is unsafe: "
            f"matched {matched}/{len(obs_cells)} cells ({match_fraction:.2%})"
        )
    return mapping, {
        "alignment": "canonical_barcode_occurrence",
        "positional_barcode_match_fraction": positional_fraction,
        "n_matched_cells": matched,
        "n_unmatched_h5ad_cells_removed": int((~matched_obs).sum()),
        "n_unmatched_design_columns": int((mapping < 0).sum()),
    }


def _assign_conditions(
    path: Path,
    obs_names: pd.Index,
    mapper: GeneNameMapper,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Resolve published guide assignments to canonical single-gene targets."""
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        design_cells = [
            _clean_barcode(value)
            for value in handle.readline().rstrip("\r\n").split("\t")
        ]
    if not design_cells or any(not value for value in design_cells):
        raise ValueError("Yao perturbation matrix has an empty or malformed cell header")

    design_to_obs, alignment = _build_design_to_obs(design_cells, obs_names)
    matched_obs = np.zeros(len(obs_names), dtype=bool)
    valid_design = design_to_obs >= 0
    matched_obs[design_to_obs[valid_design]] = True

    selected_design_positions = np.flatnonzero(valid_design)
    selected_columns = [0] + (selected_design_positions + 1).tolist()

    distinct_target_count = np.zeros(len(obs_names), dtype=np.int8)
    assigned = np.full(len(obs_names), "", dtype=object)
    has_any_guide = np.zeros(len(obs_names), dtype=bool)
    has_control_guide = np.zeros(len(obs_names), dtype=bool)
    raw_to_canonical: dict[str, str] = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        header=None,
        skiprows=1,
        index_col=0,
        compression="infer",
        chunksize=16,
        usecols=selected_columns,
    ):
        if chunk.empty:
            continue
        try:
            values = chunk.fillna(0).to_numpy(dtype=np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Yao perturbation matrix contains non-numeric guide counts") from exc

        obs_positions = np.asarray(
            [design_to_obs[int(column) - 1] for column in chunk.columns],
            dtype=np.int64,
        )
        for row_index, raw_target in enumerate(chunk.index.astype(str)):
            positive = np.flatnonzero(values[row_index] > 0)
            if positive.size == 0:
                continue
            positions = obs_positions[positive]
            has_any_guide[positions] = True
            raw_name = str(raw_target).strip()
            if _is_control_target(raw_name):
                has_control_guide[positions] = True
                continue

            target_name = _canonical_target(raw_name, mapper)
            raw_to_canonical[raw_name] = target_name
            current = assigned[positions]
            first = distinct_target_count[positions] == 0
            if first.any():
                assigned[positions[first]] = target_name
                distinct_target_count[positions[first]] = 1
            different = (~first) & (current != target_name)
            if different.any():
                distinct_target_count[positions[different]] = 2

    single_target = distinct_target_count == 1
    controls = (distinct_target_count == 0) & has_control_guide
    keep = matched_obs & (single_target | controls)
    conditions = np.full(len(obs_names), "", dtype=object)
    conditions[controls] = "ctrl"
    conditions[single_target] = assigned[single_target]

    barcode_counts = Counter(design_cells)
    raw_perturbations = sorted({
        str(value) for value in conditions[keep] if str(value) and str(value) != "ctrl"
    })
    alignment.update({
        "n_h5ad_cells": int(len(obs_names)),
        "n_design_cells": int(len(design_cells)),
        "n_unique_design_barcode_strings": int(len(barcode_counts)),
        "n_duplicate_design_barcode_positions": int(len(design_cells) - len(barcode_counts)),
        "max_design_barcode_multiplicity": int(max(barcode_counts.values(), default=0)),
        "n_multi_target_cells_removed": int((matched_obs & (distinct_target_count > 1)).sum()),
        "n_unassigned_cells_removed": int((matched_obs & ~has_any_guide).sum()),
        "n_single_target_cells": int((matched_obs & single_target).sum()),
        "n_control_cells": int((matched_obs & controls).sum()),
        "n_kept_cells": int(keep.sum()),
        "n_raw_perturbations": int(len(raw_perturbations)),
        "n_perturbation_names_changed": int(sum(k != v for k, v in raw_to_canonical.items())),
    })
    return conditions, keep, alignment


def _canonical_gene_value(value: object, mapper: GeneNameMapper) -> str:
    raw = str(value).strip().strip('"').strip("'")
    if not raw or raw.casefold() in {"nan", "none", "na"}:
        return raw

    official = mapper.maps.official_symbols
    resolved = str(mapper.resolve(raw)).strip()
    if resolved in official:
        return resolved

    match = _ENSEMBL_TOKEN.search(raw)
    if match:
        candidate = str(mapper.resolve(match.group(0))).strip()
        if candidate in official:
            return candidate

    for piece in re.split(r"[|;,]", raw):
        piece = piece.strip()
        if not piece or piece == raw:
            continue
        candidate = str(mapper.resolve(piece)).strip()
        if candidate in official:
            return candidate

    return resolved or raw


def _candidate_gene_columns(raw: ad.AnnData) -> list[tuple[str, np.ndarray]]:
    """Return plausible per-feature identifier columns without assuming Yao's schema."""
    candidates: list[tuple[str, np.ndarray]] = [
        ("var_names", raw.var_names.astype(str).to_numpy())
    ]
    n_vars = int(raw.n_vars)
    min_unique = max(20, int(0.05 * n_vars))
    for column in raw.var.columns:
        series = raw.var[column]
        if len(series) != n_vars:
            continue
        if not (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue
        values = series.astype(str).to_numpy()
        if len(pd.unique(values)) < min_unique:
            continue
        candidates.append((f"var[{column!r}]", values))
    return candidates


def _resolve_expression_gene_names(
    raw: ad.AnnData,
    mapper: GeneNameMapper,
    perturbation_targets: set[str],
) -> dict[str, object]:
    """Choose the Yao feature identifier source that actually represents genes.

    """
    official = mapper.maps.official_symbols
    target_upper = {str(g).upper() for g in perturbation_targets if g}
    scored: list[tuple[int, int, int, str, np.ndarray, np.ndarray]] = []

    for source_name, values in _candidate_gene_columns(raw):
        canonical = np.asarray(
            [_canonical_gene_value(value, mapper) for value in values],
            dtype=object,
        )
        official_mask = np.fromiter(
            (str(g) in official for g in canonical),
            dtype=bool,
            count=len(canonical),
        )
        official_genes = {str(g).upper() for g in canonical[official_mask]}
        target_overlap = len(target_upper & official_genes)
        scored.append((
            len(official_genes),
            target_overlap,
            len(set(map(str, canonical))),
            source_name,
            canonical,
            np.asarray(values, dtype=object),
        ))

    if not scored:
        raise ValueError("Yao H5AD has no usable feature identifier source")

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    n_ppi, target_overlap, n_unique, source_name, canonical, source_values = scored[0]
    required_target_overlap = min(
        _MIN_TARGETS_IN_EXPRESSION_PANEL,
        len(target_upper),
    )

    if n_ppi < _MIN_PPI_GENES_AFTER_NAME_RESOLUTION or (
        required_target_overlap and target_overlap < required_target_overlap
    ):
        diagnostics = "; ".join(
            f"{name}: PPI={ppi}, target_overlap={overlap}"
            for ppi, overlap, _, name, _, _ in scored[:6]
        )
        raise ValueError(
            "Yao gene-name resolution failed before preprocessing: "
            f"best={source_name}, PPI_genes={n_ppi}, "
            f"perturbation_targets_in_expression={target_overlap}/{len(target_upper)}; "
            f"candidates=[{diagnostics}]"
        )

    # Preserve provenance under safe names before removing AnnData/HDF5-reserved
    # dataframe columns.  Yao's correct published gene identifiers are stored in
    # var['_index']; `_index` itself cannot be written as a normal H5AD dataframe
    # column because AnnData reserves that name for the dataframe index.
    original_names = raw.var_names.astype(str).to_numpy(copy=True)
    raw.var["gpm_original_feature_id"] = original_names
    raw.var["gpm_source_gene_identifier"] = np.asarray(source_values, dtype=object)
    reserved = [
        column for column in raw.var.columns
        if str(column) in _H5AD_RESERVED_VAR_COLUMNS
    ]
    if reserved:
        raw.var.drop(columns=reserved, inplace=True)

    # Downstream preprocessing must see only the canonical BioGRID-compatible
    # namespace.  Duplicates are intentional here and are collapsed by
    # standardize_var_names() using the existing raw-count sum reducer.
    raw.var_names = pd.Index(canonical.astype(str), dtype=object)
    raw.var["gene_name"] = np.asarray(canonical, dtype=object)
    raw.var.index.name = None

    bad_columns = [
        str(column) for column in raw.var.columns
        if str(column) in _H5AD_RESERVED_VAR_COLUMNS
    ]
    if bad_columns:
        raise RuntimeError(
            f"Yao reserved AnnData var columns survived gene-name resolution: {bad_columns}"
        )

    report = {
        "gene_name_source": source_name,
        "n_features": int(raw.n_vars),
        "n_unique_canonical_names": int(n_unique),
        "n_biogrid_genes": int(n_ppi),
        "n_perturbation_targets": int(len(target_upper)),
        "n_perturbation_targets_in_expression": int(target_overlap),
    }
    print(
        f"[Yao] genes: {source_name}, {n_ppi} PPI genes, "
        f"targets in expression {target_overlap}/{len(target_upper)}",
        flush=True,
    )
    return report


def load_raw(
    source: Optional[Union[str, Path]] = None, *, biogrid_path: Optional[Path] = None,
) -> ad.AnnData:
    directory = resolve_source(SPEC, source)
    h5ad_path = directory / _H5AD
    perturbation_path = directory / _PERTURBATIONS
    missing = [str(path) for path in (h5ad_path, perturbation_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Yao: missing raw files: {missing}")

    mapper = GeneNameMapper.from_paths(biogrid_path=biogrid_path)
    raw = _read_compressed_h5ad(h5ad_path)

    conditions, keep, alignment = _assign_conditions(
        perturbation_path,
        raw.obs_names,
        mapper,
    )
    perturbation_targets = {
        str(value) for value in conditions[keep]
        if str(value) and str(value) != "ctrl"
    }

    # Materialise the selected cells while the original feature index is still
    # unique.  Canonical gene mapping may create legitimate duplicate symbols;
    # those are collapsed immediately afterwards by the common preprocessing.
    raw.obs["gpm_raw_condition"] = conditions
    raw = raw[keep].copy()
    gene_report = _resolve_expression_gene_names(raw, mapper, perturbation_targets)
    alignment["gene_name_resolution"] = gene_report
    raw.uns["matrix_metadata_alignment"] = alignment

    print(
        f"[Yao] {alignment['n_raw_perturbations']} perturbations, "
        f"{alignment['n_kept_cells']} cells retained "
        f"({alignment['n_single_target_cells']} single-target + "
        f"{alignment['n_control_cells']} ctrl; "
        f"{alignment['n_multi_target_cells_removed']} multi-target removed)",
        flush=True,
    )

    batch = first_column(raw.obs, SPEC.batch_columns)
    return standardize_adapter_output(
        raw,
        SPEC,
        perturbation_column="gpm_raw_condition",
        batch_column=batch,
        source=str(directory),
    )
