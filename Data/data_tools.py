from __future__ import annotations

import re
from typing import Any, Optional, Sequence
import anndata as ad
import numpy as np
import pandas as pd

CONTROL = "ctrl"
_SPLIT = re.compile(r"[+,;/|]")
_CONTROL = re.compile(
    r"^(ctrl|control|non[-_ ]?target(?:ing)?|ntc|negative[-_ ]?control|"
    r"no[-_ ]?guide|no[-_ ]?target|nt|mock|empty|unperturbed|wt|dmso)$",
    re.IGNORECASE,
)
BATCH_CANDIDATES = ("batch", "batch_id", "replicate", "library", "sample", "gem_group")


def _is_control(value: str, extras: Sequence[str] = ()) -> bool:
    token = str(value).strip()
    return bool(_CONTROL.fullmatch(token)) or any(x.casefold() in token.casefold() for x in extras)


def build_condition_label(
    raw_pert: str, *, extra_control_substrings: Sequence[str] = ()
) -> tuple[str, bool]:
    token = str(raw_pert).strip().strip("\"'")
    if not token or token.casefold() in {"nan", "none", "na"} or _is_control(token, extra_control_substrings):
        return CONTROL, True
    pieces = [piece.strip() for piece in _SPLIT.split(token) if piece.strip()]
    genes: list[str] = []
    for piece in pieces:
        if _is_control(piece, extra_control_substrings):
            continue
        gene = re.sub(r"_(grna|guide|sg|sgrna|cas9).*$", "", piece, flags=re.IGNORECASE)
        if gene:
            genes.append(gene)
    genes = sorted(set(genes), key=str.upper)
    if not genes:
        return CONTROL, True
    if len(genes) == 1:
        return f"{genes[0]}+ctrl", False
    return "+".join(genes), False


def split_gpm_pert_target_genes(token: str, **_: Any) -> list[str]:
    condition, is_control = build_condition_label(token)
    return [] if is_control else [part for part in condition.split("+") if part != CONTROL]


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def auto_add_gpm_metadata(
    adata: ad.AnnData,
    *,
    dataset_id: str,
    data_display_name: str,
    tech_name: str,
    cell_line_fixed: str,
    pert_col: Optional[str] = None,
    batch_col: Optional[str] = None,
    extra_control_substrings: Sequence[str] = (),
    source_script: Optional[str] = None,
    copy_matrix: bool = True,
) -> ad.AnnData:
    if pert_col is None or pert_col not in adata.obs:
        raise ValueError(f"A valid perturbation column is required, got {pert_col!r}")
    out = adata.copy() if copy_matrix else adata
    raw_conditions = out.obs[pert_col].astype(str)
    parsed = {
        value: build_condition_label(
            value, extra_control_substrings=extra_control_substrings
        )
        for value in raw_conditions.unique()
    }
    condition_labels = {value: item[0] for value, item in parsed.items()}
    control_flags = {value: item[1] for value, item in parsed.items()}
    conditions = raw_conditions.map(condition_labels).to_numpy(dtype=object)
    out.obs["condition"] = conditions
    out.obs["gpm_condition"] = conditions
    out.obs["gpm_is_control"] = raw_conditions.map(control_flags).to_numpy(dtype=bool)
    out.obs["gpm_dataset"] = dataset_id
    out.obs["gpm_cell_line"] = cell_line_fixed
    out.obs["gpm_perturbation_tech"] = tech_name.upper()
    out.obs["cell_line"] = cell_line_fixed
    out.obs["cell_type"] = cell_line_fixed
    out.obs["data_name"] = data_display_name
    out.obs["tech_name"] = tech_name.upper()
    resolved_batch = batch_col or _first_column(out.obs, BATCH_CANDIDATES)
    if resolved_batch is None:
        out.obs["gpm_batch"] = "batch0"
    else:
        # Categorical columns reject fill values that are not existing categories.
        # Convert to object before filling so raw batch annotations remain intact.
        batch_values = out.obs[resolved_batch].astype(object)
        batch_values = batch_values.where(pd.notna(batch_values), "batch0")
        out.obs["gpm_batch"] = batch_values.astype(str)
    out.uns["gpm_schema_version"] = "2.0"
    out.uns["gpm_source_script"] = source_script or "unknown"
    out.uns["gpm_column_mapping"] = {
        "raw_perturbation_obs": pert_col, "batch_obs": resolved_batch,
    }
    return out
