from __future__ import annotations

import csv
import copy
import gzip
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TextIO, Union

import numpy as np
import pandas as pd
import scanpy as sc

from GPM.paths import PROJECT_ROOT

_ENSG_RE = re.compile(r"^ENSG\d", re.IGNORECASE)
_ENSEMBL_TAG_RE = re.compile(r"Ensembl:?\s*(ENSG\d+)", re.IGNORECASE)

DEFAULT_N_TOP_HVG: int = 2000
DEFAULT_PERT_MIN_CELLS_EXCLUSIVE: int = 50



NCBI_CANDIDATE_NAMES: tuple[str, ...] = (
    "Homo_sapiens.gene_info",
    "Homo_sapiens.gene_info.gz",
)
GENE_INFO_CSV_NAMES: tuple[str, ...] = ("gene_info.csv",)
DEFAULT_BIOGRID_FILENAME = "BIOGRID.tsv"

_BIOGRID_METADATA_CACHE: dict[
    Path, tuple[frozenset[str], dict[str, str]]
] = {}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def default_annotation_dir() -> Path:
    return PROJECT_ROOT / "GPM" / "gene_embedding"


def default_gene2go_path() -> Path:
    return PROJECT_ROOT / "Data" / "gene2go_all.pkl"


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")  # type: ignore[return-value]
    return path.open("r", encoding="utf-8", errors="replace")


def _find_first_file(directory: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = directory / name
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class GeneNameMaps:
    ensembl_to_symbol: dict[str, str] = field(default_factory=dict)
    alias_to_symbol: dict[str, str] = field(default_factory=dict)
    official_symbols: frozenset[str] = frozenset()


def _register_alias(
    alias_map: dict[str, str], alias: str, symbol: str, *, overwrite: bool = False
) -> None:
    a = str(alias).strip()
    s = str(symbol).strip()
    if not a or not s:
        return
    if overwrite or a not in alias_map:
        alias_map[a] = s
    upper = a.upper()
    if overwrite or upper not in alias_map:
        alias_map[upper] = s


def _register_ensembl(ensembl_map: dict[str, str], ensembl_id: str, symbol: str) -> None:
    base = str(ensembl_id).strip().split(".")[0].upper()
    if not base.startswith("ENSG"):
        return
    sym = str(symbol).strip()
    if not sym:
        return
    if base not in ensembl_map:
        ensembl_map[base] = sym


def _split_pipe_field(raw: str) -> list[str]:
    return [p.strip() for p in str(raw).split("|") if str(p).strip()]


def _ensembl_ids_from_dbxrefs(dbxrefs: str) -> list[str]:
    out: list[str] = []
    for part in _split_pipe_field(dbxrefs):
        m = _ENSEMBL_TAG_RE.search(part)
        if m:
            out.append(m.group(1).split(".")[0].upper())
        elif part.upper().startswith("ENSG"):
            out.append(part.split(".")[0].upper())
    return out


def _load_gene_info_csv(path: Path, maps: GeneNameMaps) -> None:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        fields = {c.strip().lower(): c for c in reader.fieldnames if c}
        sym_col = (
            fields.get("gene_name")
            or fields.get("symbol")
            or fields.get("gene_symbol")
            or fields.get("feature_name")
        )
        ens_col = (
            fields.get("ensembl_id")
            or fields.get("ensembl_gene_id")
            or fields.get("ensembl")
            or fields.get("gene_id")
            or fields.get("feature_id")
        )
        for row in reader:
            symbol = str(row.get(sym_col, "")).strip() if sym_col else ""
            ens = str(row.get(ens_col, "")).strip() if ens_col else ""
            if not symbol and not ens:
                continue
            if symbol:
                _register_alias(maps.alias_to_symbol, symbol, symbol)
            if ens and symbol:
                _register_ensembl(maps.ensembl_to_symbol, ens, symbol)


def _load_ncbi_gene_info(path: Path, maps: GeneNameMaps) -> None:
    header: Optional[list[str]] = None
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                cols = line.lstrip("#").split("\t")
                if "GeneID" in cols and "Symbol" in cols:
                    header = cols
                continue
            if header is None:
                continue
            row = dict(zip(header, line.split("\t")))
            symbol = str(row.get("Symbol", "")).strip()
            if not symbol or symbol == "-":
                continue
            _register_alias(maps.alias_to_symbol, symbol, symbol)
            for syn in _split_pipe_field(str(row.get("Synonyms", ""))):
                _register_alias(maps.alias_to_symbol, syn, symbol)
            ensembl_col = str(row.get("Ensembl_gene_id", "")).strip()
            if ensembl_col and ensembl_col != "-":
                _register_ensembl(maps.ensembl_to_symbol, ensembl_col, symbol)
            for ens in _ensembl_ids_from_dbxrefs(str(row.get("dbXrefs", ""))):
                _register_ensembl(maps.ensembl_to_symbol, ens, symbol)


def resolve_biogrid_path(
    path: Optional[Union[str, Path]] = None,
    *,
    annotation_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve ``BIOGRID.tsv`` from an explicit path or the default annotation directory."""
    candidate = (
        Path(path)
        if path is not None
        else (Path(annotation_dir) if annotation_dir else default_annotation_dir())
        / DEFAULT_BIOGRID_FILENAME
    )
    if not candidate.is_file():
        raise FileNotFoundError(f"BioGRID file not found: {candidate}")
    return candidate


def _biogrid_columns(fieldnames: Optional[Sequence[str]]) -> dict[str, str]:
    fields = {
        str(name).lstrip("#").strip().lower(): str(name)
        for name in (fieldnames or ())
    }
    required = (
        "official symbol interactor a",
        "official symbol interactor b",
        "synonyms interactor a",
        "synonyms interactor b",
    )
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"BIOGRID.tsv is missing required columns: {missing}")
    return fields


def _read_biogrid_symbols_and_aliases(
    path: Path,
) -> tuple[frozenset[str], dict[str, str]]:
    """Read official symbols and unambiguous BioGRID aliases."""
    resolved = path.resolve()
    cached = _BIOGRID_METADATA_CACHE.get(resolved)
    if cached is not None:
        return cached

    official: set[str] = set()
    candidates: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = _biogrid_columns(reader.fieldnames)
        for row in reader:
            for side in ("a", "b"):
                symbol = str(row.get(cols[f"official symbol interactor {side}"], "")).strip()
                if not symbol or symbol == "-":
                    continue
                official.add(symbol)
                synonyms = str(row.get(cols[f"synonyms interactor {side}"], ""))
                for alias in _split_pipe_field(synonyms):
                    if alias != "-":
                        candidates.setdefault(alias.upper(), set()).add(symbol)

    if not official:
        raise ValueError(f"No official gene symbols were found in the BioGRID file: {path}")

    official_by_upper = {symbol.upper(): symbol for symbol in official}
    aliases = {
        alias: next(iter(targets))
        for alias, targets in candidates.items()
        if len(targets) == 1 and alias not in official_by_upper
    }
    aliases.update(official_by_upper)
    result = (frozenset(official), aliases)
    _BIOGRID_METADATA_CACHE[resolved] = result
    return result


def load_biogrid_gene_set(
    *,
    biogrid_path: Optional[Union[str, Path]] = None,
    annotation_dir: Optional[Union[str, Path]] = None,
) -> frozenset[str]:
    """Return the union of official symbols across both BioGRID interaction endpoints."""
    path = resolve_biogrid_path(biogrid_path, annotation_dir=annotation_dir).resolve()
    genes, _ = _read_biogrid_symbols_and_aliases(path)
    print(f"[annotation] Found {len(genes)} official BioGRID gene symbols.")
    return genes


class GeneNameMapper:
    """Map Ensembl IDs, former names, and aliases to canonical gene symbols."""

    def __init__(self, maps: GeneNameMaps) -> None:
        self.maps = maps

    @classmethod
    def from_paths(
        cls,
        *,
        gene_info_path: Optional[Union[str, Path]] = None,
        ncbi_path: Optional[Union[str, Path]] = None,
        biogrid_path: Optional[Union[str, Path]] = None,
        annotation_dir: Optional[Union[str, Path]] = None,
    ) -> GeneNameMapper:
        directory = Path(annotation_dir) if annotation_dir else default_annotation_dir()
        maps = GeneNameMaps()

        csv_path = Path(gene_info_path) if gene_info_path else _find_first_file(
            directory, GENE_INFO_CSV_NAMES
        )
        if csv_path is not None:
            _load_gene_info_csv(csv_path, maps)

        ncbi = Path(ncbi_path) if ncbi_path else _find_first_file(
            directory, NCBI_CANDIDATE_NAMES
        )
        if ncbi is not None:
            _load_ncbi_gene_info(ncbi, maps)

        # BioGRID official symbols take precedence over auxiliary annotations.
        bg_path = resolve_biogrid_path(biogrid_path, annotation_dir=directory)
        official, aliases = _read_biogrid_symbols_and_aliases(bg_path)
        for alias, symbol in aliases.items():
            _register_alias(maps.alias_to_symbol, alias, symbol, overwrite=True)
        maps.official_symbols = official

        return cls(maps)

    @classmethod
    def from_default_paths(cls) -> GeneNameMapper:
        """Build a mapper from BioGRID and the auxiliary files in the default annotation directory."""
        return cls.from_paths()

    def _lookup_alias(self, key: str) -> Optional[str]:
        k = str(key).strip()
        if not k:
            return None
        if k in self.maps.alias_to_symbol:
            return self.maps.alias_to_symbol[k]
        return self.maps.alias_to_symbol.get(k.upper())

    def resolve(self, gene: str) -> str:
        raw = str(gene).strip()
        if not raw or raw.lower() in {"nan", "none", "na"}:
            return raw

        if not _ENSG_RE.match(raw):
            hit = self._lookup_alias(raw)
            if hit:
                return hit
            return raw

        base = raw.split(".")[0].upper()
        sym = self.maps.ensembl_to_symbol.get(base)
        if sym:
            return self._lookup_alias(sym) or sym
        return raw

    def map_var_names(self, names: Sequence[str]) -> list[str]:
        return [self.resolve(g) for g in names]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


_GENE_SYMBOL_VAR_CANDIDATES: tuple[str, ...] = (
    "gene_name",
    "gene_names",
    "symbol",
    "Symbol",
    "feature_name",
)


def _gene_symbol_column_values(adata: sc.AnnData) -> Optional[np.ndarray]:
    for col in _GENE_SYMBOL_VAR_CANDIDATES:
        if col in adata.var.columns:
            return adata.var[col].astype(str).to_numpy()
    return None


def prepare_var_names_from_gene_symbols(
    adata: sc.AnnData, *, copy_matrix: bool = True
) -> sc.AnnData:
    """Copy the gene symbol column into ``var_names`` for GEARS-style inputs."""
    symbols = _gene_symbol_column_values(adata)
    if symbols is None:
        return adata

    current = adata.var_names.astype(str).to_numpy()
    if len(symbols) != len(current):
        return adata

    n_match = sum(
        str(a).strip() == str(b).strip()
        for a, b in zip(current, symbols)
    )
    if n_match == len(current):
        return adata

    ad = adata.copy() if copy_matrix else adata
    ad.var_names = pd.Index(symbols)
    # Retain duplicate symbols for aggregation after canonical mapping.
    ad.var["gene_name"] = np.asarray(symbols, dtype=object)
    ad.var.index.name = None
    print(
        f"[annotation] Set var_names from the GEARS gene symbol column "
        f"({n_match}/{len(current)} original names matched, {ad.n_vars} genes total)."
    )
    return ad


def standardize_var_names(
    adata: sc.AnnData,
    mapper: GeneNameMapper,
    *,
    duplicate_reducer: str = "sum",
    copy_matrix: bool = True,
) -> sc.AnnData:
    """Standardize gene symbols and merge duplicate expression columns.

    Use ``sum`` for raw counts and ``mean`` for normalized expression.
    """
    if duplicate_reducer not in {"sum", "mean"}:
        raise ValueError("duplicate_reducer must be either 'sum' or 'mean'.")

    ad = prepare_var_names_from_gene_symbols(adata, copy_matrix=copy_matrix)
    symbols = mapper.map_var_names(ad.var_names.astype(str))
    order: list[str] = []
    symbol_to_idx: dict[str, int] = {}
    group_index = np.empty(len(symbols), dtype=np.int64)
    first_indices: list[int] = []
    for old_idx, symbol in enumerate(symbols):
        clean = str(symbol).strip()
        if clean not in symbol_to_idx:
            symbol_to_idx[clean] = len(order)
            order.append(clean)
            first_indices.append(old_idx)
        group_index[old_idx] = symbol_to_idx[clean]

    counts = np.bincount(group_index, minlength=len(order)).astype(np.float32)
    n_duplicates = int(len(symbols) - len(order))
    if n_duplicates:
        import scipy.sparse as sps

        grouping = sps.csr_matrix(
            (
                np.ones(len(symbols), dtype=np.float32),
                (np.arange(len(symbols)), group_index),
            ),
            shape=(len(symbols), len(order)),
        )
        if sps.issparse(ad.X):
            collapsed = ad.X @ grouping
            if duplicate_reducer == "mean":
                collapsed = collapsed.multiply(1.0 / counts)
            # Convert to CSR for compatibility with AnnData slicing.
            collapsed = collapsed.tocsr()
        else:
            collapsed = (grouping.T @ np.asarray(ad.X).T).T
            if duplicate_reducer == "mean":
                collapsed = collapsed / counts.reshape(1, -1)

        var = ad.var.iloc[first_indices].copy()
        var.index = pd.Index(order)
        out = sc.AnnData(X=collapsed, obs=ad.obs.copy(), var=var)
        for key, value in ad.obsm.items():
            out.obsm[key] = value.copy()
        out.uns = copy.deepcopy(ad.uns)
        ad = out
    else:
        if copy_matrix and ad is adata:
            ad = ad.copy()
        ad.var_names = pd.Index(order)

    ad.var["gene_name"] = np.asarray(order, dtype=object)
    ad.var["source_feature_count"] = counts.astype(np.int32)
    ad.var.index.name = None
    ad.uns["gene_name_standardization"] = {
        "namespace": "BIOGRID Official Symbol",
        "n_features_before": len(symbols),
        "n_genes_after": len(order),
        "n_genes_in_biogrid": sum(
            symbol in mapper.maps.official_symbols for symbol in order
        ),
        "n_genes_not_in_biogrid": sum(
            symbol not in mapper.maps.official_symbols for symbol in order
        ),
        "n_duplicate_features_collapsed": n_duplicates,
        "duplicate_reducer": duplicate_reducer,
    }
    return ad


def subset_to_biogrid_genes(
    adata: sc.AnnData,
    biogrid_genes: Union[frozenset[str], set[str]],
) -> tuple[sc.AnnData, dict[str, Any]]:
    """Restrict the expression matrix to official BioGRID symbols."""
    import scipy.sparse as sps

    # Convert to CSR for compatibility with AnnData slicing.
    source = adata
    if sps.issparse(adata.X) and not sps.isspmatrix_csr(adata.X):
        source = adata.copy()
        source.X = adata.X.tocsr()

    names = source.var_names.astype(str).to_numpy()
    keep = np.fromiter((g in biogrid_genes for g in names), dtype=bool, count=len(names))
    n_before = int(source.n_vars)
    n_after = int(keep.sum())
    if n_after == 0:
        raise ValueError(
            "No expression features match official BioGRID gene symbols. Check the Ensembl and alias annotation files."
        )
    ad = source[:, keep].copy()
    print(f"[annotation] Retained {n_after} of {n_before} genes after BioGRID filtering.")
    report = {
        "n_genes_before": n_before,
        "n_genes_after": n_after,
        "n_genes_dropped": n_before - n_after,
    }
    return ad, report


def standardize_condition_gene_names(
    adata: sc.AnnData,
    mapper: GeneNameMapper,
    *,
    condition_col: str = "condition",
    control_token: str = "ctrl",
    copy_matrix: bool = True,
) -> tuple[sc.AnnData, dict[str, int]]:
    """Standardize perturbation targets with the same mapper used for expression features."""
    if condition_col not in adata.obs.columns:
        raise ValueError(f"AnnData is missing the obs[{condition_col!r}] column.")

    original = adata.obs[condition_col].astype(str)
    condition_map: dict[str, str] = {}
    unresolved_targets: set[str] = set()
    for condition in original.unique():
        if condition == control_token:
            condition_map[condition] = control_token
            continue
        targets = [mapper.resolve(g) for g in split_pert_target_genes(condition)]
        unresolved_targets.update(
            g for g in targets if g not in mapper.maps.official_symbols
        )
        targets = sorted(dict.fromkeys(g for g in targets if g), key=str.upper)
        if not targets:
            condition_map[condition] = control_token
        elif len(targets) == 1:
            condition_map[condition] = f"{targets[0]}+{control_token}"
        else:
            condition_map[condition] = "+".join(targets)

    out = adata.copy() if copy_matrix else adata
    canonical = original.map(condition_map)
    changed = int((original.to_numpy() != canonical.to_numpy()).sum())
    if changed and "condition_original" not in out.obs.columns:
        out.obs["condition_original"] = original.to_numpy(dtype=object)
    out.obs[condition_col] = canonical.to_numpy(dtype=object)
    return out, {
        "n_conditions": len(set(condition_map.values())),
        "n_cells_renamed": changed,
        "n_unresolved_targets": len(unresolved_targets),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def load_gene2go(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    p = Path(path) if path is not None else default_gene2go_path()
    if not p.is_file():
        raise FileNotFoundError(f"GO annotation file not found: {p}")
    with p.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a dict in {p}, got {type(raw)}.")
    return raw


def get_genes_from_perts(perts: Union[Sequence[str], str]) -> list[str]:
    """Extract target genes from control, single-target, and multi-target condition labels."""
    if isinstance(perts, str):
        perts = [perts]
    gene_list = [p.split("+") for p in np.unique(list(perts))]
    flat = [item for sublist in gene_list for item in sublist]
    return list(np.unique([g for g in flat if g and g != "ctrl"]))


def build_pert_names(
    gene2go_all: Mapping[str, Any],
    adata: sc.AnnData,
    *,
    condition_col: str = "condition",
) -> np.ndarray:
    """Return the eligible GO-annotated target genes using the GEARS convention."""
    panel = set(adata.var_names.astype(str))
    if "gene_name" in adata.var.columns:
        panel.update(adata.var["gene_name"].astype(str))
    pert_genes = get_genes_from_perts(adata.obs[condition_col].astype(str).tolist())
    essential = list(panel) + pert_genes
    gene2go = {g: gene2go_all[g] for g in essential if g in gene2go_all}
    return np.unique(list(gene2go.keys()))


def filter_pert_in_go(condition: str, pert_names: Sequence[str]) -> bool:
    """Return whether every target in the condition is eligible under the GEARS GO filter."""
    c = str(condition).strip()
    if c == "ctrl":
        return True
    perts_set = set(pert_names)
    parts = [p.strip() for p in c.split("+") if p.strip()]
    if len(parts) != 2:
        return all(p in perts_set for p in parts if p != "ctrl")
    cond1, cond2 = parts[0], parts[1]
    num_ctrl = int(cond1 == "ctrl") + int(cond2 == "ctrl")
    num_in_perts = int(cond1 in perts_set) + int(cond2 in perts_set)
    return num_ctrl + num_in_perts == 2


def split_pert_target_genes(condition: str) -> list[str]:
    """Extract non-control target genes from a perturbation condition."""
    c = str(condition).strip()
    if not c or c == "ctrl":
        return []
    return [p.strip() for p in c.split("+") if p.strip() and p.strip() != "ctrl"]


def _panel_gene_names(adata: sc.AnnData) -> set[str]:
    names: set[str] = set(map(str, adata.var_names))
    for col in ("gene_name", "symbol", "gene_symbols"):
        if col in adata.var.columns:
            names.update(
                str(x).strip()
                for x in adata.var[col].tolist()
                if str(x).strip() and str(x).strip().lower() not in {"nan", "none"}
            )
    return names


def mask_obs_pert_targets_in_panel(
    adata: sc.AnnData,
    *,
    condition_col: str = "condition",
    control_token: str = "ctrl",
) -> np.ndarray:
    """Keep controls and cells whose perturbation targets all occur in the panel."""
    panel = _panel_gene_names(adata)
    out = np.zeros(adata.n_obs, dtype=bool)
    cond = adata.obs[condition_col].astype(str).to_numpy()
    for i, c in enumerate(cond):
        if c == control_token:
            out[i] = True
            continue
        genes = split_pert_target_genes(c)
        out[i] = bool(genes) and all(g in panel for g in genes)
    return out


def filter_perturbations_by_min_cells(
    adata: sc.AnnData,
    *,
    condition_col: str = "condition",
    control_token: str = "ctrl",
    min_cells_exclusive: int = DEFAULT_PERT_MIN_CELLS_EXCLUSIVE,
    copy_result: bool = True,
) -> tuple[sc.AnnData, dict[str, Any]]:
    """Keep controls and perturbations with strictly more cells than the threshold."""
    cond_s = adata.obs[condition_col].astype(str)
    vc = cond_s.value_counts()
    n_ctrl_cells = int((cond_s == control_token).sum())
    perts_before = sorted(str(x) for x in vc.index if str(x) != control_token)
    kept: set[str] = {control_token}
    dropped: list[dict[str, Any]] = []
    for c, n in vc.items():
        cs = str(c)
        if cs == control_token:
            continue
        if int(n) > min_cells_exclusive:
            kept.add(cs)
        else:
            dropped.append({"condition": cs, "n_cells": int(n)})

    mask = cond_s.isin(kept).to_numpy()
    if bool(mask.all()):
        out = adata.copy() if copy_result else adata
    else:
        out = adata[mask].copy() if copy_result else adata[mask]
    cond_after = out.obs[condition_col].astype(str)
    perts_after = sorted(
        str(x) for x in cond_after.value_counts().index if str(x) != control_token
    )
    stats: dict[str, Any] = {
        "min_cells_exclusive_threshold": int(min_cells_exclusive),
        "n_cells_before": int(adata.n_obs),
        "n_cells_after": int(out.n_obs),
        "n_perturbations_before": len(perts_before),
        "n_perturbations_after": len(perts_after),
        "n_control_cells_before": n_ctrl_cells,
        "dropped_perturbations": sorted(
            dropped, key=lambda x: (-int(x["n_cells"]), str(x["condition"]))
        ),
        "n_dropped_perturbations": len(dropped),
    }
    if dropped:
        print(
            f"[annotation] Perturbation conditions removed for having at most {min_cells_exclusive} cells: "
            f"{len(dropped)}"
        )
    return out, stats


def filter_perturbations_in_go(
    adata: sc.AnnData,
    pert_names: Sequence[str],
    *,
    condition_col: str = "condition",
    control_token: str = "ctrl",
    copy_result: bool = True,
) -> tuple[sc.AnnData, dict[str, Any]]:
    """Drop conditions with targets outside the eligible GO gene set."""
    cond_s = adata.obs[condition_col].astype(str)
    vc = cond_s.value_counts()
    kept: set[str] = {control_token}
    dropped: list[dict[str, Any]] = []
    for c, n in vc.items():
        cs = str(c)
        if cs == control_token:
            continue
        if filter_pert_in_go(cs, pert_names):
            kept.add(cs)
        else:
            dropped.append({"condition": cs, "n_cells": int(n)})

    mask = cond_s.isin(kept).to_numpy()
    if bool(mask.all()):
        out = adata.copy() if copy_result else adata
    else:
        out = adata[mask].copy() if copy_result else adata[mask]
    cond_after = out.obs[condition_col].astype(str)
    perts_after = sorted(
        str(x) for x in cond_after.value_counts().index if str(x) != control_token
    )
    stats: dict[str, Any] = {
        "n_perturbations_before": int((vc.index != control_token).sum()),
        "n_perturbations_after": len(perts_after),
        "dropped_perturbations": sorted(
            dropped, key=lambda x: (-int(x["n_cells"]), str(x["condition"]))
        ),
        "n_dropped_perturbations": len(dropped),
        "n_cells_before": int(adata.n_obs),
        "n_cells_after": int(out.n_obs),
    }
    if dropped:
        print(
            f"[annotation] Removed {len(dropped)} perturbation conditions with targets absent from the GO graph."
        )
    return out, stats


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def is_log1p_normalized(adata: sc.AnnData) -> bool:
    """Read explicit preprocessing metadata to detect log-normalized expression."""
    gpm_norm = adata.uns.get("gpm_normalization", {})
    return (
        isinstance(gpm_norm, Mapping)
        and bool(gpm_norm.get("log1p", False))
    ) or "log1p" in adata.uns


def normalize_total_log1p(
    adata: sc.AnnData,
    *,
    target_sum: float = 1e4,
    copy: bool = True,
) -> sc.AnnData:
    """Apply library-size normalization and ``log1p`` across all genes before panel selection."""
    ad = adata.copy() if copy else adata
    sc.pp.normalize_total(ad, target_sum=float(target_sum))
    sc.pp.log1p(ad)
    ad.raw = None
    ad.uns["gpm_normalization"] = {
        "method": "normalize_total_log1p",
        "target_sum": float(target_sum),
        "log1p": True,
    }
    return ad


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _sanitize_x_for_hvg(adata: sc.AnnData) -> None:
    import scipy.sparse as sps

    x = adata.X
    if sps.issparse(x):
        if x.data.size and not np.isfinite(x.data).all():
            x = x.copy()
            x.data = np.nan_to_num(x.data, nan=0.0, posinf=0.0, neginf=0.0)
            adata.X = x
    else:
        xa = np.asarray(x, dtype=np.float64)
        if xa.size and not np.isfinite(xa).all():
            adata.X = np.nan_to_num(xa, nan=0.0, posinf=0.0, neginf=0.0)


def _hvg_fallback_top_variance(adata: sc.AnnData, n_top: int) -> None:
    import scipy.sparse as sps

    x = adata.X
    n = int(adata.n_vars)
    if n == 0:
        return
    if int(adata.n_obs) == 0:
        adata.var["highly_variable"] = False
        return
    if sps.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        var = np.maximum(mean_sq - mean**2, 0.0)
    else:
        var = np.var(np.asarray(x, dtype=np.float64), axis=0).ravel()
    n_pick = min(int(n_top), n)
    top_idx = np.argsort(-var, kind="stable")[:n_pick]
    sel = np.zeros(n, dtype=bool)
    sel[top_idx] = True
    adata.var["highly_variable"] = sel


def apply_hvg_panel(
    adata: sc.AnnData,
    *,
    n_top_genes: int = DEFAULT_N_TOP_HVG,
    condition_col: str = "condition",
    normalize_and_log1p: bool = False,
) -> sc.AnnData:
    """Select highly variable genes from controls and retain all perturbation targets.

    Set ``normalize_and_log1p`` for raw count inputs.
    """
    ad = adata.copy()
    if normalize_and_log1p:
        ad = normalize_total_log1p(ad, copy=False)

    if ad.n_vars == 0:
        raise ValueError("apply_hvg_panel: AnnData has zero genes.")

    control_mask = ad.obs[condition_col].astype(str).to_numpy() == "ctrl"
    if not bool(control_mask.any()):
        raise ValueError("apply_hvg_panel: no control cells for target-aware HVG selection.")
    fit = ad[control_mask].copy()
    _sanitize_x_for_hvg(fit)
    n_top = min(int(n_top_genes), max(1, ad.n_vars))

    if fit.n_vars <= 2:
        fit.var["highly_variable"] = True
    else:
        try:
            sc.pp.highly_variable_genes(fit, n_top_genes=n_top, subset=False)
        except ValueError as err:
            msg = str(err).lower()
            if "bin edges must be unique" in msg or "duplicates" in msg:
                try:
                    sc.pp.highly_variable_genes(
                        fit, n_top_genes=n_top, subset=False, flavor="cell_ranger"
                    )
                except (ValueError, TypeError, KeyError):
                    _hvg_fallback_top_variance(fit, n_top)
            else:
                raise

    target_tokens: set[str] = set()
    for c in ad.obs[condition_col].astype(str).unique():
        if c == "ctrl":
            continue
        target_tokens.update(split_pert_target_genes(c))

    var_names_str = ad.var_names.astype(str).to_numpy()
    gene_in_any_pert = np.fromiter(
        (str(v) in target_tokens for v in var_names_str),
        dtype=bool,
        count=len(var_names_str),
    )
    ad.var["highly_variable"] = fit.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
    hv_series = ad.var["highly_variable"]
    hvg = hv_series.fillna(False).to_numpy(dtype=bool)
    keep_genes = hvg | gene_in_any_pert
    ad.uns["gpm_hvg_subset"] = {
        "n_top_genes_requested": int(n_top_genes),
        "n_genes_hvg": int(hvg.sum()),
        "n_genes_pert_union_extra": int((gene_in_any_pert & ~hvg).sum()),
        "n_genes_after_union": int(keep_genes.sum()),
        "normalize_and_log1p": bool(normalize_and_log1p),
        "condition_col": condition_col,
        "hvg_fit_scope": "target_control_cells_only",
        "n_control_cells_used": int(control_mask.sum()),
        "n_perturbed_cells_used": 0,
    }
    ad = ad[:, keep_genes].copy()
    ad.raw = None
    print(
        f"[annotation] Expression panel: {int(hvg.sum())} HVGs + "
        f"{int((gene_in_any_pert & ~hvg).sum())} additional perturbation targets "
        f"= {int(keep_genes.sum())} genes."
    )
    return ad


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def print_dataset_cell_summary(
    adata: sc.AnnData,
    *,
    condition_col: str = "condition",
    control_token: str = "ctrl",
    label: str = "",
) -> dict[str, int]:
    """Print and return counts of controls, perturbed cells, and perturbation conditions."""
    cond = adata.obs[condition_col].astype(str)
    is_ctrl = cond == control_token
    n_ctrl = int(is_ctrl.sum())
    n_pert_cells = int((~is_ctrl).sum())
    n_pert_types = int(cond[~is_ctrl].nunique()) if n_pert_cells else 0
    prefix = f"[annotation]{f' {label}' if label else ''}"
    print(f"{prefix} Control cells: {n_ctrl}")
    print(f"{prefix} Perturbed cells: {n_pert_cells}")
    print(f"{prefix} Perturbation conditions excluding controls: {n_pert_types}")
    return {
        "n_ctrl_cells": n_ctrl,
        "n_pert_cells": n_pert_cells,
        "n_perturbation_types": n_pert_types,
    }


def preprocess_gpm_adata(
    adata: sc.AnnData,
    *,
    mapper: Optional[GeneNameMapper] = None,
    biogrid_genes: Optional[Union[frozenset[str], set[str]]] = None,
    biogrid_path: Optional[Union[str, Path]] = None,
    n_top_hvg: int = DEFAULT_N_TOP_HVG,
    min_pert_cells_exclusive: int = DEFAULT_PERT_MIN_CELLS_EXCLUSIVE,
    condition_col: str = "condition",
    control_token: str = "ctrl",
    normalize_log1p: bool = True,
) -> tuple[sc.AnnData, dict[str, Any]]:
    """Standardize and preprocess a single dataset for export.

    Normalize over all measured genes before filtering conditions and selecting the
    expression panel.
    """
    report: dict[str, Any] = {}
    mapper = mapper or GeneNameMapper.from_paths(biogrid_path=biogrid_path)

    already_normalized = is_log1p_normalized(adata) or not normalize_log1p
    reducer = "mean" if already_normalized else "sum"
    ad = standardize_var_names(adata, mapper, duplicate_reducer=reducer)
    ad, condition_report = standardize_condition_gene_names(
        ad, mapper, condition_col=condition_col, control_token=control_token
    )
    report["gene_name_standardization"] = dict(
        ad.uns.get("gene_name_standardization", {})
    )
    report["condition_name_standardization"] = condition_report
    if not already_normalized:
        ad = normalize_total_log1p(ad)
        report["normalize_log1p"] = {
            "applied": True,
            "stage": "before_biogrid_filter",
            "target_sum": 1e4,
            "n_genes_at_normalize": int(ad.n_vars),
        }
    else:
        if "gpm_normalization" not in ad.uns:
            ad.uns["gpm_normalization"] = {
                "method": "upstream_preprocessed",
                "log1p": True,
            }
        report["normalize_log1p"] = {
            "applied": False,
            "stage": "skipped_already_normalized",
        }

    report["biogrid_filter"] = {
        "enabled": False,
        "policy": "graph_prior_only_no_gene_or_perturbation_filtering",
    }

    keep = mask_obs_pert_targets_in_panel(ad, condition_col=condition_col)
    n_drop_panel = int(ad.n_obs) - int(keep.sum())
    if n_drop_panel:
        print(f"[annotation] Removed {n_drop_panel} cells with perturbation targets absent from the measured gene set.")
    ad = ad[np.asarray(keep, dtype=bool)].copy()
    if ad.n_obs == 0:
        raise ValueError("No cells remain after filtering for measured perturbation targets. Check the perturbation labels and gene symbols.")

    ad, pert_cell_stats = filter_perturbations_by_min_cells(
        ad,
        condition_col=condition_col,
        control_token=control_token,
        min_cells_exclusive=min_pert_cells_exclusive,
    )
    report["perturbation_cell_count_filter"] = pert_cell_stats

    ad = apply_hvg_panel(
        ad,
        n_top_genes=n_top_hvg,
        condition_col=condition_col,
        normalize_and_log1p=False,
    )
    report["hvg_subset"] = ad.uns.get("gpm_hvg_subset")
    report["cell_summary"] = print_dataset_cell_summary(
        ad, condition_col=condition_col, control_token=control_token, label="Final"
    )
    report["n_genes"] = int(ad.n_vars)
    report["n_cells"] = int(ad.n_obs)
    return ad, report


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class GeneAlignmentMaps:
    """Compatibility shim for legacy gene encoder imports."""


def align_gene_name(gene: str, store=None, maps: Optional[GeneAlignmentMaps] = None) -> str:
    return str(gene).strip()


def align_gene_names(
    gene_names: Sequence[str],
    store=None,
    maps: Optional[GeneAlignmentMaps] = None,
) -> list[str]:
    return [align_gene_name(g, store, maps) for g in gene_names]


def get_alignment_maps(
    annotation_dir: Optional[Union[str, Path]] = None,
    store=None,
    **kwargs: Any,
) -> GeneAlignmentMaps:
    return GeneAlignmentMaps()


def print_panel_scgpt_hit_rates(
    gene_names: Sequence[str],
    store,
    *,
    annotation_dir: Optional[Union[str, Path]] = None,
    maps: Optional[GeneAlignmentMaps] = None,
) -> GeneAlignmentMaps:
    names = [str(g) for g in gene_names]
    n = len(names)
    hits = sum(1 for g in names if store.vocab_index(g) is not None)
    pct = 100.0 * hits / n if n else 0.0
    print(f"Gene embedding coverage: {pct:.2f}% ({hits}/{n})")
    return GeneAlignmentMaps()
