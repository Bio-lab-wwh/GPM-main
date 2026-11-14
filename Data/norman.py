"""Norman K562 CRISPRa single/combinatorial raw-count adapter."""

from pathlib import Path
import re
from typing import Optional, Union

import pandas as pd

from Data.matrix_io import read_mtx_triplet
from Data.preprocessing import (
    DatasetSpec, DATA_DIR, first_column, resolve_source, standardize_adapter_output,
)

SPEC = DatasetSpec(
    key="norman", dataset_id="Norman2019_K562_OE", cell_line="K562",
    technology="OE", default_source=DATA_DIR / "Norman",
    perturbation_columns=("perturbation", "guide_identity", "condition", "target", "gene"),
    batch_columns=("batch", "batch_id", "gemgroup", "gem_group", "replicate", "library"),
    control_tokens=("non-targeting", "non_targeting", "NegCtrl", "negative"),
)


def load_raw(source: Optional[Union[str, Path]] = None):
    directory = resolve_source(SPEC, source)
    raw = read_mtx_triplet(
        directory,
        matrix_file="GSE133344_filtered_matrix.mtx.gz",
        cells_file="GSE133344_filtered_barcodes.tsv.gz",
        genes_file="GSE133344_filtered_genes.tsv.gz",
        metadata_file="GSE133344_filtered_cell_identities.csv.gz",
    )
    pert = first_column(raw.obs, SPEC.perturbation_columns)
    if pert is None:
        raise ValueError(f"Norman: no perturbation column in {list(raw.obs.columns)!r}")

    identities = raw.obs[pert].astype(str).str.strip()
    excluded_control = identities == "NegCtrl1_NegCtrl0__NegCtrl1_NegCtrl0"
    number_of_cells = first_column(raw.obs, ("number_of_cells", "number of cells"))
    if number_of_cells is not None:
        singlet = pd.to_numeric(raw.obs[number_of_cells], errors="coerce").fillna(1.0) <= 1.0
    else:
        singlet = pd.Series(True, index=raw.obs_names)
    keep = (~excluded_control) & singlet
    raw = raw[keep.to_numpy()].copy()
    raw.uns["adapter_filtering"] = {
        "excluded_special_negative_control_cells": int(excluded_control.sum()),
        "excluded_doublet_cells": int((~singlet).sum()),
        "n_cells_retained": int(keep.sum()),
    }

    # guide_identity repeats the target pair on both sides of "__", e.g.
    # KLF1_MAP2K6__KLF1_MAP2K6. A target paired with NegCtrl is a single
    # perturbation, not a control.
    def canonical_guide(value: object) -> str:
        token = str(value).strip().split("__", 1)[0]
        if "+" in token:
            parts = [part for part in token.split("+") if part]
        else:
            parts = [part for part in token.split("_") if part]
        while parts and re.fullmatch(r"(?:g|guide)?\d+", parts[-1], flags=re.IGNORECASE):
            parts.pop()
        targets = [
            part for part in parts
            if "negctrl" not in part.casefold() and "nontarget" not in part.casefold()
        ]
        return "+".join(dict.fromkeys(targets)) if targets else "ctrl"
    raw.obs["gpm_raw_condition"] = raw.obs[pert].map(canonical_guide)
    batch = first_column(raw.obs, SPEC.batch_columns)
    return standardize_adapter_output(
        raw, SPEC, perturbation_column="gpm_raw_condition", batch_column=batch, source=str(directory)
    )
