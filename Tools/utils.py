from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import scipy.sparse as sp

from Data.data_tools import split_gpm_pert_target_genes
from GPM.conditions import is_control_condition, normalize_tech_name
from GPM.utils import GPMSample


class AnnDataExpressionStore:
    """Shared float32 CPU expression cache for a dataset's training, validation, and test rows."""

    __slots__ = ("_matrix", "_n_genes", "_moment_cache")

    def __init__(self, adata: Any) -> None:
        self._n_genes = int(adata.n_vars)
        source = adata.X
        if hasattr(source, "to_memory"):
            value = source.to_memory()
        else:
            value = source[:] if hasattr(source, "__getitem__") else source
        if sp.issparse(value):
            value = value.toarray()
        matrix = np.ascontiguousarray(value, dtype=np.float32)
        if matrix.shape != (int(adata.n_obs), self._n_genes):
            raise ValueError(
                "Invalid AnnData expression cache shape. "
                f"Expected {(int(adata.n_obs), self._n_genes)}, got {matrix.shape}."
            )
        self._matrix = matrix
        self._moment_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def num_genes(self) -> int:
        return self._n_genes

    def read_rows(self, rows: np.ndarray) -> np.ndarray:
        indices = np.asarray(rows, dtype=np.int64).ravel()
        if indices.size == 0:
            return np.empty((0, self._n_genes), dtype=np.float32)
        return self._matrix[indices]

    def control_moments(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return cached control-only moments without depending on minibatches."""
        indices = np.asarray(rows, dtype=np.int64).ravel()
        if indices.size == 0:
            raise ValueError("Control moments require at least one control cell.")
        key = indices.tobytes()
        cached = self._moment_cache.get(key)
        if cached is None:
            total = np.zeros(self._n_genes, dtype=np.float64)
            squared = np.zeros(self._n_genes, dtype=np.float64)
            for start in range(0, indices.size, 4096):
                values = self._matrix[indices[start : start + 4096]]
                total += values.sum(axis=0, dtype=np.float64)
                squared += np.square(values, dtype=np.float64).sum(axis=0)
            mean64 = total / indices.size
            variance64 = np.maximum(
                squared / indices.size - np.square(mean64), 0.0
            )
            mean = mean64.astype(np.float32)
            variance = variance64.astype(np.float32)
            cached = (mean, variance)
            self._moment_cache[key] = cached
        return cached

def panel_gene_names(adata: Any) -> list[str]:
    return [str(g) for g in adata.var_names]


def split_indices(
    adata: Any,
    split_name: str,
    *,
    split_column: str = "split",
    controls_only: bool = False,
    perturb_only: bool = False,
) -> np.ndarray:
    mask = adata.obs[split_column].astype(str) == str(split_name)
    conditions = adata.obs["condition"].astype(str)
    if controls_only:
        mask &= conditions == "ctrl"
    if perturb_only:
        mask &= conditions != "ctrl"
    return np.where(mask.to_numpy())[0]


def ctrl_pool_indices(adata: Any) -> np.ndarray:
    result = np.where((adata.obs["condition"].astype(str) == "ctrl").to_numpy())[0]
    if result.size == 0:
        raise RuntimeError("Control pool is empty")
    return result


def _obs_text(row: Any, candidates: tuple[str, ...], default: str) -> str:
    for column in candidates:
        if column in row.index:
            value = str(row[column]).strip()
            if value and value.casefold() not in {"nan", "none"}:
                return value
    return default


def _cell_line(row: Any, default: str) -> str:
    return _obs_text(row, ("gpm_cell_line", "cell_line", "cell_type"), default)


def _batch(row: Any) -> str:
    return _obs_text(row, ("gpm_batch", "batch", "batch_id", "replicate", "library"), "batch0")


def _technology(row: Any, default: str) -> str:
    return _obs_text(
        row, ("gpm_perturbation_tech", "perturbation_tech", "tech_name"), default
    )


def _paired_controls(
    adata: Any,
    perturbed: np.ndarray,
    controls: np.ndarray,
    *,
    seed: int,
    default_cell_line: str,
) -> tuple[dict[int, int], dict[int, str]]:
    exact: dict[tuple[str, str, str], list[int]] = {}
    fallback: dict[tuple[str, str], list[int]] = {}
    for index in controls:
        row = adata.obs.iloc[int(index)]
        dataset = _obs_text(row, ("gpm_dataset",), "dataset")
        cell_line = _cell_line(row, default_cell_line)
        exact.setdefault((dataset, cell_line, _batch(row)), []).append(int(index))
        fallback.setdefault((dataset, cell_line), []).append(int(index))
    rng = np.random.default_rng(seed)
    pairs: dict[int, int] = {}
    levels: dict[int, str] = {}
    for index in sorted(map(int, perturbed)):
        row = adata.obs.iloc[index]
        dataset = _obs_text(row, ("gpm_dataset",), "dataset")
        cell_line = _cell_line(row, default_cell_line)
        candidates = exact.get((dataset, cell_line, _batch(row)))
        level = "dataset_cell_batch"
        if not candidates:
            candidates = fallback.get((dataset, cell_line))
            level = "dataset_cell_fallback"
        if not candidates:
            raise RuntimeError(
                f"No matching control for dataset={dataset}, cell_line={cell_line}, batch={_batch(row)}"
            )
        pairs[index] = candidates[int(rng.integers(0, len(candidates)))]
        levels[index] = level
    return pairs, levels


def build_gpm_samples(
    adata: Any,
    *,
    cell_indices: np.ndarray,
    control_pool_indices: np.ndarray,
    gene_names: list[str],
    sample_seed: int,
    default_tech: str,
    default_cell_line: str,
    panel_key: Optional[str] = None,
    expression_store: Optional[AnnDataExpressionStore] = None,
    technology_to_id: Optional[Mapping[str, int]] = None,
) -> list[GPMSample]:
    indices = np.asarray(cell_indices, dtype=np.int64)
    if indices.size == 0:
        return []
    perturbed = np.asarray([
        i for i in indices if not is_control_condition(str(adata.obs.iloc[int(i)]["condition"]))
    ], dtype=np.int64)
    pairs, match_levels = _paired_controls(
        adata, perturbed, np.asarray(control_pool_indices, dtype=np.int64),
        seed=sample_seed, default_cell_line=default_cell_line,
    )
    store = expression_store or AnnDataExpressionStore(adata)
    if store.num_genes != len(gene_names):
        raise ValueError("The number of genes in the expression store does not match the fixed gene panel.")
    panel_genes = tuple(str(g) for g in gene_names)
    gene_lookup = {gene.upper(): idx for idx, gene in enumerate(panel_genes)}
    tech_lookup = {
        normalize_tech_name(name): int(idx)
        for name, idx in (technology_to_id or {}).items()
    }
    names = adata.obs_names.astype(str)
    # Compute control moments within each dataset and cell line.
    controls_by_context: dict[tuple[str, str], list[int]] = {}
    for control_index in np.asarray(control_pool_indices, dtype=np.int64):
        control_row = adata.obs.iloc[int(control_index)]
        context_key = (
            _obs_text(control_row, ("gpm_dataset",), "dataset"),
            _cell_line(control_row, default_cell_line),
        )
        controls_by_context.setdefault(context_key, []).append(int(control_index))
    context_moments = {
        key: store.control_moments(np.asarray(rows, dtype=np.int64))
        for key, rows in controls_by_context.items()
    }
    samples: list[GPMSample] = []
    for index in indices:
        index = int(index)
        row = adata.obs.iloc[index]
        condition = str(row["condition"])
        is_control = is_control_condition(condition)
        control_index = index if is_control else pairs[index]
        target_genes = [] if is_control else split_gpm_pert_target_genes(condition)
        target_indices: list[int] = []
        seen_targets: set[int] = set()
        for target in target_genes:
            target_idx = gene_lookup.get(str(target).strip().upper())
            if target_idx is None:
                raise ValueError(f"Perturbation target {target!r} is absent from the fixed gene panel.")
            if target_idx not in seen_targets:
                target_indices.append(target_idx)
                seen_targets.add(target_idx)
        if not is_control and not target_indices:
            raise ValueError(f"Perturbed sample {names[index]!r} has no valid target genes.")

        tech_name = normalize_tech_name(_technology(row, default_tech))
        cell_line_name = _cell_line(row, default_cell_line)
        context_key = (
            _obs_text(row, ("gpm_dataset",), "dataset"),
            cell_line_name,
        )
        if context_key not in context_moments:
            raise RuntimeError(
                f"No control expression moments match the dataset and cell line for sample {names[index]!r}."
            )
        context_mean, context_variance = context_moments[context_key]
        if is_control:
            tech_id = 0
        elif tech_name in tech_lookup:
            tech_id = tech_lookup[tech_name]
        else:
            raise ValueError(
                f"Perturbation technology {tech_name!r} is absent from the configured vocabulary {tuple(tech_lookup)}."
            )
        samples.append(GPMSample(
            expression=None,
            target_expression=None,
            gene_names=panel_genes,
            pert_target_genes=tuple(str(g) for g in target_genes),
            pert_tech=tech_name,
            condition=condition,
            cell_line=cell_line_name,
            batch_id=_batch(row),
            is_control=is_control,
            sample_id=names[index],
            panel_key=panel_key,
            control_match_level="identity" if is_control else match_levels[index],
            expression_store=store,
            expression_row=control_index,
            target_expression_row=index,
            precomputed_target_gene_indices=tuple(target_indices),
            precomputed_pert_tech_id=tech_id,
            precomputed_perturbation_group_key=(
                str(panel_key or "__default__"),
                cell_line_name.upper(),
                condition.strip(),
                tech_name,
            ),
            control_context_mean=context_mean,
            control_context_variance=context_variance,
        ))
    return samples
