from pathlib import Path
import re
from typing import Optional, Union
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from Data.matrix_io import (
    clean_string_index, orient_matrix, read_mtx, unique_feature_index,
)
from Data.preprocessing import DatasetSpec, DATA_DIR, resolve_source, standardize_adapter_output

SPEC = DatasetSpec(
    key="dixit", dataset_id="Dixit2016_K562_KO", cell_line="K562", technology="KO",
    default_source=DATA_DIR / "Dixit",
    perturbation_columns=("gene", "target_gene", "perturbation", "guide_identity", "gRNA"),
    batch_columns=("batch", "batch_id", "replicate", "library"),
    control_tokens=("non-targeting", "non_targeting"),
)


def _one_column(path: Path) -> pd.Series:
    # These GEO CSVs were exported with a pandas row index; use the first data
    # column, not the serialized numeric index.
    indexed = pd.read_csv(path, index_col=0, compression="infer")
    if indexed.shape[1] >= 1:
        return indexed.iloc[:, 0].astype(str).str.strip().reset_index(drop=True)
    plain = pd.read_csv(path, header=None, compression="infer")
    if plain.shape[1] == 0:
        raise ValueError(f"Dixit annotation file is empty: {path}")
    return plain.iloc[:, -1].astype(str).str.strip().reset_index(drop=True)


def _guide_target(value: object) -> str:
    """Collapse the published guide IDs to gene-level perturbation labels.

    GSM2396858 uses IDs such as ``p_sgEGR1_3`` and
    ``p_INTERGENIC1144056``.  Keeping the plasmid prefix would turn EGR1 into
    the nonexistent symbol ``p_sgEGR1`` and every perturbation would then be
    removed by the PPI/GO filters.
    """
    token = str(value).strip().strip('"').strip("'")
    if "INTERGENIC" in token.upper():
        return "ctrl"
    token = re.sub(r"^p_sg", "", token, flags=re.IGNORECASE)
    token = re.sub(
        r"_(?:(?:g|guide|sg|sgrna)[_-]?)?\d+$", "", token,
        flags=re.IGNORECASE,
    )
    # Compatibility with the control encoding used by other Dixit releases.
    if re.search(r"(?:NON.?TARGET|CONTROL|CTRL|TERGE)", token, flags=re.IGNORECASE):
        return "ctrl"
    return token


def _guide_assignments(path: Path, cells: pd.Index) -> tuple[pd.DataFrame, dict[str, int]]:
    # The GEO cbc_gbc dictionary is guide -> comma-separated cell barcodes,
    # not cell -> guide. Invert it while retaining multi-guide assignments.
    guide_to_cells = pd.read_csv(
        path, index_col=0, header=None, compression="infer", dtype=str
    )
    allowed_cells = set(map(str, cells))
    assignments: dict[str, set[str]] = {}
    for guide, row in guide_to_cells.iterrows():
        target = _guide_target(guide)
        for value in row.dropna().astype(str):
            for barcode in value.replace(" ", "").split(","):
                if barcode in allowed_cells:
                    assignments.setdefault(barcode, set()).add(target)
    selected_cells = [cell for cell in cells if str(cell) in assignments]
    if not selected_cells:
        raise ValueError("Dixit guide dictionary has no cell barcodes present in the matrix")
    conditions = [
        "+".join(sorted(assignments[str(cell)], key=str.upper))
        for cell in selected_cells
    ]
    obs = pd.DataFrame(
        {"gpm_raw_condition": conditions},
        index=clean_string_index(selected_cells),
    )
    return obs, {
        "n_guide_rows": int(len(guide_to_cells)),
        "n_matrix_cells": int(len(cells)),
        "n_assigned_cells": int(len(selected_cells)),
        "n_unassigned_cells_removed": int(len(cells) - len(selected_cells)),
        "n_multi_guide_cells": int(sum(len(assignments[str(cell)]) > 1 for cell in selected_cells)),
    }


def load_raw(source: Optional[Union[str, Path]] = None):
    directory = resolve_source(SPEC, source)
    prefix = "GSM2396858_k562_tfs_7_"
    cells = _one_column(directory / f"{prefix}cellnames.csv.gz")
    raw_gene_names = _one_column(directory / f"{prefix}genenames.csv.gz")
    feature_ids: list[str] = []
    genes: list[str] = []
    for value in raw_gene_names:
        token = str(value)
        feature, separator, symbol = token.partition("_")
        feature_ids.append(feature if separator else token)
        genes.append(symbol if separator and symbol else token)
    # GEO names the matrix with a dot before ``mtx`` while the companion
    # annotation files use an underscore.  Accept the earlier local spelling
    # as a fallback, but prefer the published filename shown in the dataset.
    matrix_candidates = (
        directory / "GSM2396858_k562_tfs_7.mtx.txt.gz",
        directory / f"{prefix}mtx.txt.gz",
    )
    matrix_path = next((path for path in matrix_candidates if path.is_file()), None)
    if matrix_path is None:
        raise FileNotFoundError(
            "Dixit expression matrix is missing; expected one of: "
            + ", ".join(str(path) for path in matrix_candidates)
        )
    matrix = orient_matrix(read_mtx(matrix_path), len(cells), len(genes))
    cell_index = clean_string_index(cells)
    if cell_index.has_duplicates:
        raise ValueError("Dixit matrix cell names are not unique")
    obs, alignment = _guide_assignments(
        directory / f"{prefix}cbc_gbc_dict.csv.gz", cell_index
    )
    keep = cell_index.isin(obs.index)
    if not keep.all():
        warnings.warn(
            f"Dropping {int((~keep).sum())} Dixit matrix cells without perturbation metadata",
            RuntimeWarning,
        )
    matrix = matrix[np.flatnonzero(keep)].tocsr()
    obs = obs.reindex(cell_index[keep]).copy()
    obs["gpm_raw_batch"] = [
        str(cell).split("_", 1)[1] if "_" in str(cell) else "batch0"
        for cell in obs.index
    ]
    raw = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=unique_feature_index(feature_ids)),
    )
    raw.var["gene_name"] = np.asarray(genes, dtype=object)
    raw.uns["matrix_metadata_alignment"] = alignment
    return standardize_adapter_output(
        raw,
        SPEC,
        perturbation_column="gpm_raw_condition",
        batch_column="gpm_raw_batch",
        source=str(directory),
    )
