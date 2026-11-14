from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack

from Data.matrix_io import clean_string_index, unique_feature_index
from Data.preprocessing import (
    DatasetSpec,
    DATA_DIR,
    first_column,
    resolve_source,
    standardize_adapter_output,
)


SPEC = DatasetSpec(
    key="frangieh",
    dataset_id="Frangieh2021_Melanoma_KO",
    cell_line="Melanoma",
    technology="KO",
    default_source=DATA_DIR / "Frangieh",
    perturbation_columns=("gpm_raw_condition",),
    batch_columns=("gpm_environment_raw",),
    control_tokens=("non-targeting", "non_targeting"),
)

_METADATA = "RNA_metadata.csv"
_GUIDES = "all_sgRNA_assignments.txt"
_EXPRESSION_SUBSETS = (
    "RNA_expression_subset1a.csv.gz",
    "RNA_expression_subset1b_alpha.csv.gz",
    "RNA_expression_subset1b_beta.csv.gz",
    "RNA_expression_subset2.csv.gz",
    "RNA_expression_subset3.csv.gz",
    "RNA_expression_subset4.csv.gz",
)
_GENE_CHUNK_SIZE = 512


def _read_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, index_col=0, dtype=object, low_memory=False)
    metadata.index = clean_string_index(metadata.index)
    type_rows = metadata.index.astype(str).str.strip().str.upper() == "TYPE"
    if bool(type_rows.any()):
        metadata = metadata.loc[~type_rows].copy()
    if metadata.index.has_duplicates:
        examples = metadata.index[metadata.index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"Frangieh metadata cell identifiers are not unique; examples={examples}"
        )
    return metadata


def _detect_separator(path: Path) -> str:
    with path.open("rt", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _read_guides(path: Path) -> pd.DataFrame:
    guides = pd.read_csv(
        path,
        sep=_detect_separator(path),
        engine="c",
        index_col=0,
        dtype=object,
        low_memory=False,
    )
    guides.index = clean_string_index(guides.index)
    if guides.index.has_duplicates:
        examples = guides.index[guides.index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"Frangieh guide cell identifiers are not unique; examples={examples}"
        )
    return guides


def _guide_target(guide: object) -> Optional[str]:
    token = str(guide).strip().strip('"').strip("'")
    if not token or token.casefold() in {"nan", "none", "na"}:
        return None
    if "SITE" in token.upper():
        return "ctrl"
    target = token.split("_", 1)[0].strip()
    return target or None


def _assign_single_target_conditions(
    metadata: pd.DataFrame,
    guides: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    guide_col = first_column(guides, ("sgRNAs", "sgRNA", "guide", "guide_identity"))
    if guide_col is None:
        raise ValueError(
            "Frangieh: cannot identify sgRNA assignment column in "
            f"{list(guides.columns)!r}"
        )

    assignments = guides.reindex(metadata.index)[guide_col]
    conditions = np.full(len(metadata), "", dtype=object)
    keep = np.zeros(len(metadata), dtype=bool)
    n_control = n_single = n_multi = n_unassigned = 0

    for position, raw_value in enumerate(assignments.to_numpy(dtype=object)):
        if pd.isna(raw_value) or not str(raw_value).strip():
            n_unassigned += 1
            continue

        targets: set[str] = set()
        has_control = False
        for guide in (x.strip() for x in str(raw_value).split(",") if x.strip()):
            target = _guide_target(guide)
            if target == "ctrl":
                has_control = True
            elif target:
                targets.add(target)

        if len(targets) == 1:
            conditions[position] = next(iter(targets))
            keep[position] = True
            n_single += 1
        elif not targets and has_control:
            conditions[position] = "ctrl"
            keep[position] = True
            n_control += 1
        elif len(targets) > 1:
            n_multi += 1
        else:
            n_unassigned += 1

    selected = metadata.loc[keep].copy()
    selected["gpm_raw_condition"] = conditions[keep]
    return selected, {
        "guide_assignment_column": str(guide_col),
        "n_metadata_cells": int(len(metadata)),
        "n_single_target_cells": int(n_single),
        "n_control_cells": int(n_control),
        "n_multi_target_cells_removed": int(n_multi),
        "n_unassigned_cells_removed": int(n_unassigned),
        "n_cells_selected_before_expression_alignment": int(keep.sum()),
    }


def _resolve_umi_column(metadata: pd.DataFrame) -> str:
    column = first_column(
        metadata,
        ("UMI_count", "umi_count", "n_umis", "n_umi", "UMIs", "umi"),
    )
    if column is None:
        raise ValueError(
            "Frangieh RNA subsets require per-cell UMI_count for official "
            f"log-CPM -> count restoration; columns={list(metadata.columns)!r}"
        )
    return column


def _read_header_cells(path: Path) -> pd.Index:
    with gzip.open(
        path, "rt", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"Frangieh expression subset is empty: {path}") from exc
    if len(header) < 2:
        raise ValueError(f"Frangieh expression subset has a malformed header: {path}")
    cells = clean_string_index(pd.Index(header[1:], dtype=object))
    if cells.has_duplicates:
        examples = cells[cells.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"Frangieh subset {path.name} contains duplicate cell ids; examples={examples}"
        )
    return cells


def _expression_mode(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Frangieh expression chunk contains no finite values")
    integer_fraction = float(
        np.mean(np.isclose(finite, np.rint(finite), atol=1e-6))
    )
    return "raw_counts" if integer_fraction >= 0.999 else "log_cpm"


def _restore_counts(
    values: np.ndarray,
    umi_scale: np.ndarray,
    mode: str,
) -> np.ndarray:
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Frangieh expression contains NaN or +Inf values")
    if mode == "log_cpm":
        np.exp(values, out=values)
        values *= umi_scale[None, :]
    np.rint(values, out=values)
    if (values < 0).any():
        raise ValueError("Frangieh restored count matrix contains negative values")
    return values.astype(np.int32, copy=False)


def _read_expression_file(
    path: Path,
    header_cells: pd.Index,
    metadata: pd.DataFrame,
    *,
    file_index: int,
) -> tuple[csr_matrix, list[str], pd.Index, str]:
    selected_set = set(metadata.index.astype(str))
    selected_positions = [
        i for i, cell in enumerate(header_cells.astype(str)) if cell in selected_set
    ]
    selected_cells = [str(header_cells[i]) for i in selected_positions]
    if not selected_cells:
        return csr_matrix((0, 0), dtype=np.int32), [], pd.Index([], dtype=object), ""

    selected_index = clean_string_index(pd.Index(selected_cells, dtype=object))
    selected_metadata = metadata.reindex(selected_index)
    umi_col = _resolve_umi_column(selected_metadata)
    umi = pd.to_numeric(
        selected_metadata[umi_col], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(umi).all() or (umi <= 0).any():
        raise ValueError(
            f"Frangieh UMI_count is missing/non-positive in {path.name}"
        )
    umi_scale = (umi / 1_000_000.0).astype(np.float32)

    usecols = [0] + [position + 1 for position in selected_positions]
    blocks: list[csr_matrix] = []
    genes: list[str] = []
    mode: Optional[str] = None

    for chunk in pd.read_csv(
        path,
        index_col=0,
        chunksize=_GENE_CHUNK_SIZE,
        compression="infer",
        usecols=usecols,
        engine="c",
    ):
        if list(map(str, chunk.columns)) != selected_cells:
            raise ValueError(
                f"Frangieh expression columns changed while reading {path.name}"
            )
        genes.extend(map(str, chunk.index))
        values = chunk.to_numpy(dtype=np.float32, copy=True)
        if mode is None:
            mode = _expression_mode(values)
        restored = _restore_counts(values, umi_scale, mode)
        block = csr_matrix(restored).T
        block.eliminate_zeros()
        blocks.append(block)

    if not blocks or mode is None:
        raise ValueError(f"Frangieh expression subset has no data rows: {path}")

    matrix = hstack(blocks, format="csr")
    print(
        f"[Frangieh] subset {file_index}/{len(_EXPRESSION_SUBSETS)}: "
        f"{path.name}, {len(selected_cells)} selected cells, {len(genes)} genes",
        flush=True,
    )
    return matrix, genes, selected_index, mode


def _read_expression_subsets(
    paths: list[Path],
    metadata: pd.DataFrame,
) -> tuple[csr_matrix, list[str], pd.Index, dict[str, object]]:
    selected_set = set(metadata.index.astype(str))
    headers = {path: _read_header_cells(path) for path in paths}

    # The SCP release can be split either by cell columns or by gene rows.
    # Detect the published layout from the headers before parsing any values.
    cell_occurrences = {cell: 0 for cell in selected_set}
    for cells in headers.values():
        present = selected_set & set(cells.astype(str))
        for cell in present:
            cell_occurrences[cell] += 1

    missing = sorted(cell for cell, count in cell_occurrences.items() if count == 0)
    if missing:
        raise ValueError(
            f"Frangieh RNA subsets are missing {len(missing)} retained metadata cells; "
            f"examples={missing[:5]}"
        )

    counts = set(cell_occurrences.values())
    if counts == {1}:
        partition_mode = "cells"
    elif counts == {len(paths)}:
        partition_mode = "genes"
    else:
        distribution = pd.Series(list(cell_occurrences.values())).value_counts().to_dict()
        raise ValueError(
            "Frangieh RNA subset layout is neither a disjoint cell partition nor "
            f"a shared-cell gene partition; cell occurrence counts={distribution}"
        )

    matrices: list[csr_matrix] = []
    modes: set[str] = set()
    genes: list[str] = []

    if partition_mode == "cells":
        row_order: list[str] = []
        expected_genes: Optional[list[str]] = None
        for index, path in enumerate(paths, start=1):
            matrix, file_genes, cells, mode = _read_expression_file(
                path, headers[path], metadata, file_index=index
            )
            if not len(cells):
                print(
                    f"[Frangieh] subset {index}/{len(paths)}: "
                    f"{path.name}, no retained cells; skipped",
                    flush=True,
                )
                continue
            if expected_genes is None:
                expected_genes = file_genes
            elif file_genes != expected_genes:
                raise ValueError(
                    f"Frangieh gene order differs in {path.name}; "
                    "cell-partitioned subsets must contain identical gene rows"
                )
            matrices.append(matrix)
            row_order.extend(map(str, cells))
            modes.add(mode)

        if len(row_order) != len(set(row_order)):
            raise RuntimeError(
                "Frangieh cell-partition assembly produced duplicate cell ids"
            )
        expression_cells = clean_string_index(pd.Index(row_order, dtype=object))
        genes = expected_genes or []
        matrix = vstack(matrices, format="csr")

    else:
        reference_cells: Optional[list[str]] = None
        seen_genes: set[str] = set()
        expression_cells = pd.Index([], dtype=object)
        for index, path in enumerate(paths, start=1):
            matrix_part, file_genes, cells, mode = _read_expression_file(
                path, headers[path], metadata, file_index=index
            )
            current_cells = list(map(str, cells))
            if reference_cells is None:
                reference_cells = current_cells
                expression_cells = clean_string_index(cells)
            elif current_cells != reference_cells:
                raise ValueError(
                    f"Frangieh selected cell order differs in {path.name}; "
                    "gene-partitioned subsets must share the same cell columns"
                )

            duplicate_genes = seen_genes & set(file_genes)
            if duplicate_genes:
                raise ValueError(
                    "Frangieh gene-partitioned subsets contain repeated gene rows; "
                    f"examples={sorted(duplicate_genes)[:5]}"
                )
            seen_genes.update(file_genes)
            genes.extend(file_genes)
            matrices.append(matrix_part)
            modes.add(mode)

        matrix = hstack(matrices, format="csr")

    if not matrices:
        raise ValueError("Frangieh RNA subsets contain no retained expression cells")
    if len(modes) != 1:
        raise ValueError(
            f"Frangieh RNA subsets disagree on expression mode: {sorted(modes)}"
        )
    if matrix.shape != (len(expression_cells), len(genes)):
        raise RuntimeError(
            "Frangieh assembled expression matrix shape does not match cell/gene axes"
        )

    mode = next(iter(modes))
    return matrix, genes, expression_cells, {
        "expression_mode": mode,
        "subset_partition_mode": partition_mode,
        "expression_files": [path.name for path in paths],
        "n_expression_cells": int(matrix.shape[0]),
        "n_expression_genes": int(matrix.shape[1]),
        "count_restoration": (
            "round(exp(expression) * UMI_count / 1e6)"
            if mode == "log_cpm"
            else "already_integer_counts"
        ),
    }

def load_raw(source: Optional[Union[str, Path]] = None) -> ad.AnnData:
    directory = resolve_source(SPEC, source)
    metadata_path = directory / _METADATA
    guides_path = directory / _GUIDES
    expression_paths = [directory / name for name in _EXPRESSION_SUBSETS]
    required = [metadata_path, guides_path, *expression_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Frangieh requires the official 8-file subset release; missing: "
            + ", ".join(missing)
        )

    metadata = _read_metadata(metadata_path)
    guides = _read_guides(guides_path)
    selected_metadata, guide_report = _assign_single_target_conditions(metadata, guides)

    raw_conditions = selected_metadata["gpm_raw_condition"].astype(str)
    raw_perturbations = sorted(set(raw_conditions[raw_conditions != "ctrl"]))
    guide_report["n_raw_perturbations"] = int(len(raw_perturbations))

    environment_col = first_column(
        selected_metadata,
        ("condition", "culture", "treatment", "environment"),
    )
    if environment_col is None:
        raise ValueError(
            "Frangieh metadata has no experimental-environment column; "
            "Control / IFN-gamma / Co-culture must be retained for matched controls"
        )
    selected_metadata["gpm_environment_raw"] = (
        selected_metadata[environment_col]
        .fillna("unknown")
        .astype(str)
    )
    environment_counts = (
        selected_metadata["gpm_environment_raw"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"[Frangieh] {len(raw_perturbations)} perturbations, "
        f"{len(selected_metadata)} cells retained "
        f"({guide_report['n_single_target_cells']} single-target + "
        f"{guide_report['n_control_cells']} ctrl; "
        f"{guide_report['n_multi_target_cells_removed']} multi-target removed)",
        flush=True,
    )
    print(
        "[Frangieh] environments: "
        + ", ".join(f"{key}={int(value)}" for key, value in environment_counts.items()),
        flush=True,
    )

    matrix, genes, expression_cells, expression_report = _read_expression_subsets(
        expression_paths,
        selected_metadata,
    )
    obs = selected_metadata.reindex(expression_cells).copy()
    if obs.isnull().all(axis=1).any():
        raise ValueError("Frangieh expression cells could not be aligned to metadata")

    raw = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=unique_feature_index(genes)),
    )
    raw.var["gene_name"] = np.asarray(genes, dtype=object)
    raw.uns["matrix_metadata_alignment"] = {
        "alignment_mode": "six_official_expression_subsets_by_cell_id",
        "n_metadata_rows": int(len(metadata)),
        "n_guide_rows": int(len(guides)),
        "environment_column": str(environment_col),
        "environment_cell_counts": {
            str(key): int(value) for key, value in environment_counts.items()
        },
        **expression_report,
    }
    raw.uns["adapter_filtering"] = guide_report

    return standardize_adapter_output(
        raw,
        SPEC,
        perturbation_column="gpm_raw_condition",
        batch_column="gpm_environment_raw",
        source=str(directory),
    )
