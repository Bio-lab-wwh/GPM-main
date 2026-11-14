"""Xu HEK293 CRISPRi raw MatrixMarket adapter."""

import gzip
from pathlib import Path
from typing import Optional, Union
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy.sparse import csr_matrix

from Data.matrix_io import (
    clean_string_index,
    orient_matrix,
    read_table,
    unique_feature_index,
)
from Data.preprocessing import DatasetSpec, DATA_DIR, first_column, resolve_source, standardize_adapter_output

SPEC = DatasetSpec(
    key="xu", dataset_id="Xu2023_HEK293_KD", cell_line="HEK293",
    technology="KD", default_source=DATA_DIR / "Xu",
    perturbation_columns=("target_genes", "target_gene", "perturbation", "condition"),
    batch_columns=("batch", "batch_id", "replicate", "time", "timepoint"),
    control_tokens=("NO-TARGET", "no-target", "non-targeting"),
)


def load_raw(source: Optional[Union[str, Path]] = None):
    directory = resolve_source(SPEC, source)
    matrix_path = directory / "GSM6752591_on_target_whole_tx_count_matrix.mtx.gz"
    obs_path = directory / "GSM6752591_on_target_cell_metadata.csv.gz"
    var_path = directory / "GSM6752591_on_target_whole_tx.Genes.tsv.gz"
    barcode_path = directory / "GSM6752591_on_target_whole_tx.Barcodes.tsv.gz"
    for path in (matrix_path, obs_path, var_path):
        if not path.is_file():
            raise FileNotFoundError(f"Xu: missing raw file {path}")
    obs = pd.read_csv(obs_path)
    cell_col = first_column(
        obs, ("cell", "cell_id", "cell_name", "barcode", "cell_barcode")
    )
    # This GEO file is a one-column TSV of gene symbols. ``sep=None`` is unsafe
    # here because csv.Sniffer may mistake a character inside a gene symbol for
    # a delimiter and then report a variable number of fields.
    var = read_table(
        var_path, header=None, dtype=str, keep_default_na=False
    ).iloc[:, 0].astype(str).str.strip()
    if var.empty or var.eq("").any():
        raise ValueError("Xu gene annotation contains empty gene symbols")
    with gzip.open(matrix_path, "rb") as handle:
        matrix = csr_matrix(spio.mmread(handle))
    alignment = {
        "n_metadata_rows": int(len(obs)),
        "barcode_file_present": bool(barcode_path.is_file()),
        "metadata_cell_id_column": str(cell_col) if cell_col is not None else None,
        "gene_annotation_delimiter": "tab",
        "n_gene_annotation_rows": int(len(var)),
    }
    if barcode_path.is_file():
        barcodes = read_table(
            barcode_path, header=None, dtype=str, keep_default_na=False
        ).iloc[:, 0]
        barcode_index = clean_string_index(barcodes)
        if barcode_index.has_duplicates:
            raise ValueError("Xu whole-transcriptome barcodes are not unique")
        matrix = orient_matrix(matrix, len(barcode_index), len(var))
        if cell_col is not None:
            metadata_index = clean_string_index(obs[cell_col])
            if metadata_index.has_duplicates:
                raise ValueError(f"Xu metadata column {cell_col!r} contains duplicate cells")
            obs = obs.copy()
            obs.index = metadata_index
            aligned = obs.reindex(barcode_index)
            keep = aligned[cell_col].notna().to_numpy()
            if float(keep.mean()) < 0.8:
                raise ValueError(
                    f"Xu metadata matches only {int(keep.sum())}/{len(keep)} barcodes"
                )
            if not keep.all():
                warnings.warn(
                    f"Dropping {int((~keep).sum())} Xu matrix cells without metadata",
                    RuntimeWarning,
                )
            matrix = matrix[np.flatnonzero(keep)].tocsr()
            obs = aligned.iloc[np.flatnonzero(keep)].copy()
            obs.index = barcode_index[keep]
            alignment["alignment_mode"] = "cell_id"
            alignment["n_matched"] = int(keep.sum())
            alignment["n_matrix_barcodes_without_metadata"] = int((~keep).sum())
        elif len(obs) == len(barcode_index):
            obs = obs.copy()
            obs.index = barcode_index
            alignment["alignment_mode"] = "published_row_order_with_barcodes"
            alignment["n_matched"] = int(len(obs))
            alignment["n_matrix_barcodes_without_metadata"] = 0
        else:
            raise ValueError(
                "Xu metadata has no cell-ID column and row count differs from barcodes"
            )
    else:
        warnings.warn(
            "Xu barcode annotation is absent; aligning matrix columns to metadata "
            "by the published row order. Download the matching whole_tx.Barcodes.tsv.gz "
            "file when possible to enable explicit barcode validation.",
            RuntimeWarning,
        )
        matrix = orient_matrix(matrix, len(obs), len(var))
        obs = obs.copy()
        if cell_col is not None:
            obs.index = clean_string_index(obs[cell_col])
            if obs.index.has_duplicates:
                raise ValueError(f"Xu metadata column {cell_col!r} contains duplicate cells")
        else:
            obs.index = clean_string_index(obs.index)
        alignment["alignment_mode"] = "published_metadata_row_order"
        alignment["n_matched"] = int(len(obs))
        alignment["n_matrix_barcodes_without_metadata"] = 0
    raw = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=unique_feature_index(var)),
    )
    raw.var["gene_name"] = var.to_numpy()
    raw.uns["matrix_metadata_alignment"] = alignment
    pert = first_column(raw.obs, SPEC.perturbation_columns)
    if pert is None:
        raise ValueError(f"Xu: no perturbation column in {list(raw.obs.columns)!r}")
    batch = first_column(raw.obs, SPEC.batch_columns)
    return standardize_adapter_output(
        raw, SPEC, perturbation_column=pert, batch_column=batch, source=str(directory)
    )
