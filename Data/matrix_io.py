from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy.sparse import csr_matrix

from Data.preprocessing import first_column


def clean_string_index(values: Sequence[object]) -> pd.Index:
    """Return an AnnData-safe string index without inherited CSV column names."""
    return pd.Index([str(value).strip() for value in values], dtype=object, name=None)


def unique_feature_index(values: Sequence[object]) -> pd.Index:
    """Make feature IDs unique while gene symbols remain available in var['gene_name']."""
    clean = clean_string_index(values)
    seen: dict[str, int] = {}
    used: set[str] = set()
    unique: list[str] = []
    for value in clean:
        base = str(value)
        occurrence = seen.get(base, 0)
        candidate = base if occurrence == 0 else f"{base}__dup{occurrence}"
        while candidate in used:
            occurrence += 1
            candidate = f"{base}__dup{occurrence}"
        seen[base] = occurrence + 1
        used.add(candidate)
        unique.append(candidate)
    return pd.Index(unique, dtype=object, name=None)


def read_table(
    path: Path,
    *,
    header: Optional[int] = 0,
    dtype: Any = None,
    keep_default_na: bool = True,
) -> pd.DataFrame:
    sep = "\t" if ".tsv" in path.name or path.suffix == ".txt" else ","
    return pd.read_csv(
        path,
        sep=sep,
        header=header,
        compression="infer",
        dtype=dtype,
        keep_default_na=keep_default_na,
    )


def read_mtx(path: Path) -> csr_matrix:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return csr_matrix(spio.mmread(handle))
    return csr_matrix(spio.mmread(path))


def orient_matrix(matrix: csr_matrix, n_obs: int, n_vars: int) -> csr_matrix:
    if matrix.shape == (n_obs, n_vars):
        return matrix.tocsr()
    if matrix.shape == (n_vars, n_obs):
        return matrix.T.tocsr()
    raise ValueError(
        f"Matrix shape {matrix.shape} does not match {n_obs} cells x {n_vars} genes"
    )


def _canonical_barcode(value: object) -> str:
    barcode = str(value).strip().strip('"').strip("'")
    # 10x matrices frequently append the gem-group suffix while companion
    # metadata omits it. Only remove a terminal numeric suffix.
    head, separator, suffix = barcode.rpartition("-")
    return head if separator and suffix.isdigit() else barcode


def read_mtx_triplet(
    directory: Union[str, Path],
    *,
    matrix_file: str,
    cells_file: str,
    genes_file: str,
    metadata_file: Optional[str] = None,
    cell_id_candidates: Sequence[str] = ("cell", "cell_id", "barcode", "cell_barcode"),
) -> ad.AnnData:
    directory = Path(directory)
    required = [matrix_file, cells_file, genes_file]
    if metadata_file:
        required.append(metadata_file)
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing raw files {missing}")

    cells = read_table(directory / cells_file, header=None).iloc[:, 0].astype(str).str.strip()
    cell_index = clean_string_index(cells)
    if cell_index.has_duplicates:
        duplicated = cell_index[cell_index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Matrix cell barcodes are not unique; examples={duplicated}")
    gene_table = read_table(directory / genes_file, header=None)
    # 10x features.tsv uses Ensembl ID in column 0 and gene symbol in column 1.
    genes = gene_table.iloc[:, 1 if gene_table.shape[1] > 1 else 0].astype(str)
    matrix = orient_matrix(read_mtx(directory / matrix_file), len(cells), len(genes))
    alignment_report = {
        "n_matrix_barcodes": int(len(cells)),
        "n_metadata_rows": 0,
        "n_matched": int(len(cells)),
        "n_matrix_barcodes_without_metadata": 0,
        "n_metadata_rows_without_matrix": 0,
        "used_canonical_barcode_fallback": False,
    }
    if metadata_file:
        metadata = read_table(directory / metadata_file)
        alignment_report["n_metadata_rows"] = int(len(metadata))
        cell_col = first_column(metadata, cell_id_candidates)
        if cell_col is not None:
            metadata_ids = metadata[cell_col].astype(str).str.strip()
            metadata = metadata.set_index(metadata_ids, drop=False)
            if metadata.index.has_duplicates:
                raise ValueError(f"Metadata cell-id column {cell_col!r} contains duplicates")

            canonical_to_metadata: dict[str, str] = {}
            ambiguous: set[str] = set()
            for metadata_id in metadata.index.astype(str):
                canonical = _canonical_barcode(metadata_id)
                if canonical in canonical_to_metadata:
                    ambiguous.add(canonical)
                else:
                    canonical_to_metadata[canonical] = metadata_id
            for canonical in ambiguous:
                canonical_to_metadata.pop(canonical, None)

            matrix_positions: list[int] = []
            matched_metadata_ids: list[str] = []
            used_fallback = False
            for position, matrix_id in enumerate(cells.astype(str)):
                if matrix_id in metadata.index:
                    matched = matrix_id
                else:
                    matched = canonical_to_metadata.get(_canonical_barcode(matrix_id))
                    used_fallback = used_fallback or matched is not None
                if matched is not None:
                    matrix_positions.append(position)
                    matched_metadata_ids.append(matched)

            match_fraction = len(matrix_positions) / max(1, len(cells))
            if match_fraction < 0.8:
                raise ValueError(
                    f"Only {len(matrix_positions)}/{len(cells)} matrix barcodes match metadata "
                    f"using column {cell_col!r}"
                )
            missing_matrix = len(cells) - len(matrix_positions)
            unused_metadata = len(metadata) - len(set(matched_metadata_ids))
            alignment_report.update({
                "n_matched": int(len(matrix_positions)),
                "n_matrix_barcodes_without_metadata": int(missing_matrix),
                "n_metadata_rows_without_matrix": int(unused_metadata),
                "used_canonical_barcode_fallback": bool(used_fallback),
                "metadata_cell_id_column": str(cell_col),
            })
            if missing_matrix:
                warnings.warn(
                    f"Dropping {missing_matrix} matrix cells without metadata; "
                    f"retaining {len(matrix_positions)} aligned cells",
                    RuntimeWarning,
                )
            matrix = matrix[np.asarray(matrix_positions, dtype=np.int64)].tocsr()
            cells = cells.iloc[matrix_positions].reset_index(drop=True)
            obs = metadata.loc[matched_metadata_ids].copy()
            obs.index = clean_string_index(cells)
        elif len(metadata) == len(cells):
            obs = metadata.copy()
            obs.index = clean_string_index(cells)
        else:
            raise ValueError("Metadata has neither a cell-id column nor matching row order")
    else:
        obs = pd.DataFrame(index=clean_string_index(cells))

    feature_ids = gene_table.iloc[:, 0].astype(str)
    var = pd.DataFrame(index=unique_feature_index(feature_ids))
    var["gene_name"] = genes.to_numpy()
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.obs_names = clean_string_index(cells)
    result.obs.index.name = None
    result.var.index.name = None
    result.uns["matrix_metadata_alignment"] = alignment_report
    return result
