"""Reproducible, stratified sampling of control inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

CONTROL_INPUT_PAIR_SEED_OFFSET: int = 1_000_003


def control_input_pair_seed(base_seed: int) -> int:
    """Derive a separate seed for control-to-control pairing."""
    return int(base_seed) + CONTROL_INPUT_PAIR_SEED_OFFSET


@dataclass(frozen=True)
class CellDescriptor:
    """Cell identifier and metadata used to assign pairing strata."""

    cell_id: str
    is_control: bool
    cell_type_id: Optional[int] = None
    cell_line: Optional[str] = None
    batch_id: Optional[str] = None
    metadata: Optional[Mapping[str, Hashable]] = None


def _stable_sort_cells(cells: Sequence[CellDescriptor]) -> list[CellDescriptor]:
    """Sort cells by ID so pairings are independent of input order."""
    return sorted(cells, key=lambda c: str(c.cell_id))


def _stratum_key(cell: CellDescriptor, *, key_field: str) -> str:
    """Resolve a built-in or metadata field into a stratum key."""
    if key_field == "cell_id":
        return str(cell.cell_id)
    if key_field == "cell_type_id":
        if cell.cell_type_id is None:
            return "__none__"
        return str(int(cell.cell_type_id))
    if key_field == "cell_line":
        if cell.cell_line is None:
            return "__none__"
        return str(cell.cell_line)
    if key_field == "batch_id":
        if cell.batch_id is None:
            return "__none__"
        return str(cell.batch_id)
    if cell.metadata and key_field in cell.metadata:
        return str(cell.metadata[key_field])
    return "__none__"


@dataclass(frozen=True)
class CellPair:
    """One perturbed cell and the matched control input."""

    control_cell_id: str
    perturbed_cell_id: str
    control_index: int
    perturbed_index: int
    stratum: str = ""


@dataclass(frozen=True)
class ControlInputPair:
    """One control query and the control sampled as its input."""

    query_cell_id: str
    input_control_cell_id: str
    query_index: int
    input_control_index: int
    stratum: str = ""


@dataclass
class PairingResult:
    """Control-to-perturbed pairings, ordered identifiers, and a reproducibility digest."""

    pairs: list[CellPair]
    random_seed: int
    # IDs are sorted in the order used by the stored indices.
    control_ids: list[str]
    perturbed_ids: list[str]
    stratum_key_field: str
    with_replacement: bool
    digest: str = ""

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def control_index_for_perturbed(self) -> np.ndarray:
        """Return each matched control's index in ``control_ids``."""
        return np.asarray([p.control_index for p in self.pairs], dtype=np.int64)

    def perturbed_indices(self) -> np.ndarray:
        """Return the matched perturbed row indices in ascending order."""
        return np.asarray([p.perturbed_index for p in self.pairs], dtype=np.int64)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the pairings."""
        return {
            "random_seed": self.random_seed,
            "stratum_key_field": self.stratum_key_field,
            "with_replacement": self.with_replacement,
            "digest": self.digest,
            "control_ids": self.control_ids,
            "perturbed_ids": self.perturbed_ids,
            "pairs": [
                {
                    "control_cell_id": p.control_cell_id,
                    "perturbed_cell_id": p.perturbed_cell_id,
                    "control_index": p.control_index,
                    "perturbed_index": p.perturbed_index,
                    "stratum": p.stratum,
                }
                for p in self.pairs
            ],
        }


@dataclass
class ControlInputPairingResult:
    """Control-to-control pairings and their reproducibility digest."""

    pairs: list[ControlInputPair]
    random_seed: int
    query_ids: list[str]
    pool_control_ids: list[str]
    stratum_key_field: str
    with_replacement: bool
    digest: str = ""

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def input_control_id_for_query(self) -> dict[str, str]:
        """Map each query control ID to its sampled input control ID."""
        return {p.query_cell_id: p.input_control_cell_id for p in self.pairs}

    def to_dict(self) -> dict:
        return {
            "random_seed": self.random_seed,
            "stratum_key_field": self.stratum_key_field,
            "with_replacement": self.with_replacement,
            "digest": self.digest,
            "query_ids": self.query_ids,
            "pool_control_ids": self.pool_control_ids,
            "pairs": [
                {
                    "query_cell_id": p.query_cell_id,
                    "input_control_cell_id": p.input_control_cell_id,
                    "query_index": p.query_index,
                    "input_control_index": p.input_control_index,
                    "stratum": p.stratum,
                }
                for p in self.pairs
            ],
        }


def _compute_pairing_digest(
    *,
    random_seed: int,
    stratum_key_field: str,
    with_replacement: bool,
    control_ids: Sequence[str],
    perturbed_ids: Sequence[str],
    pairs: Sequence[CellPair],
) -> str:
    """Compute a short digest for comparing pairing results across models."""
    payload = {
        "random_seed": int(random_seed),
        "stratum_key_field": stratum_key_field,
        "with_replacement": bool(with_replacement),
        "control_ids": list(control_ids),
        "perturbed_ids": list(perturbed_ids),
        "pairs": [
            (p.control_cell_id, p.perturbed_cell_id, p.stratum) for p in pairs
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _compute_control_input_digest(
    *,
    random_seed: int,
    stratum_key_field: str,
    with_replacement: bool,
    query_ids: Sequence[str],
    pool_control_ids: Sequence[str],
    pairs: Sequence[ControlInputPair],
) -> str:
    payload = {
        "kind": "control_input",
        "random_seed": int(random_seed),
        "stratum_key_field": stratum_key_field,
        "with_replacement": bool(with_replacement),
        "query_ids": list(query_ids),
        "pool_control_ids": list(pool_control_ids),
        "pairs": [
            (p.query_cell_id, p.input_control_cell_id, p.stratum) for p in pairs
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def pair_control_perturbed(
    controls: Sequence[CellDescriptor],
    perturbed: Sequence[CellDescriptor],
    random_seed: int,
    *,
    stratum_key_field: str = "cell_line",
    with_replacement: bool = True,
) -> PairingResult:
    """Sample one control per perturbed cell within the requested stratum.

    Use a fixed seed for reproducibility. Sampling without replacement requires at least
    as many controls as perturbed cells in each stratum.
    """
    ctrl_sorted = _stable_sort_cells(list(controls))
    pert_sorted = _stable_sort_cells(list(perturbed))

    if not ctrl_sorted:
        raise ValueError("At least one control cell is required for pairing.")
    if not pert_sorted:
        return PairingResult(
            pairs=[],
            random_seed=int(random_seed),
            control_ids=[c.cell_id for c in ctrl_sorted],
            perturbed_ids=[],
            stratum_key_field=stratum_key_field,
            with_replacement=with_replacement,
            digest=_compute_pairing_digest(
                random_seed=random_seed,
                stratum_key_field=stratum_key_field,
                with_replacement=with_replacement,
                control_ids=[c.cell_id for c in ctrl_sorted],
                perturbed_ids=[],
                pairs=[],
            ),
        )

    control_ids = [c.cell_id for c in ctrl_sorted]
    perturbed_ids = [c.cell_id for c in pert_sorted]
    ctrl_by_stratum: dict[str, list[int]] = {}
    for i, c in enumerate(ctrl_sorted):
        s = _stratum_key(c, key_field=stratum_key_field)
        ctrl_by_stratum.setdefault(s, []).append(i)

    pert_by_stratum: dict[str, list[int]] = {}
    for i, p in enumerate(pert_sorted):
        s = _stratum_key(p, key_field=stratum_key_field)
        pert_by_stratum.setdefault(s, []).append(i)

    rng = np.random.default_rng(int(random_seed))
    pairs: list[CellPair] = []

    # Sort strata to make the sampling order deterministic.
    for stratum in sorted(pert_by_stratum.keys()):
        pert_indices = pert_by_stratum[stratum]
        ctrl_indices = ctrl_by_stratum.get(stratum)

        if not ctrl_indices:
            raise ValueError(
                f"Stratum {stratum!r} contains {len(pert_indices)} perturbed cells and no controls. "
                f"Pairing requires controls in the same stratum (stratum_key_field={stratum_key_field!r})."
            )

        n_ctrl = len(ctrl_indices)
        n_pert = len(pert_indices)

        if not with_replacement and n_ctrl < n_pert:
            raise ValueError(
                f"Pairing without replacement requires at least as many controls as perturbed cells in stratum {stratum!r}. "
                f"Found {n_ctrl} controls and {n_pert} perturbed cells."
            )

        if with_replacement:
            choices = rng.integers(0, n_ctrl, size=n_pert, endpoint=False)
        else:
            perm = rng.permutation(n_ctrl)
            choices = perm[:n_pert]

        for local_pert_i, pert_i in enumerate(pert_indices):
            ctrl_i = ctrl_indices[int(choices[local_pert_i])]
            pairs.append(
                CellPair(
                    control_cell_id=control_ids[ctrl_i],
                    perturbed_cell_id=perturbed_ids[pert_i],
                    control_index=ctrl_i,
                    perturbed_index=pert_i,
                    stratum=stratum,
                )
            )

    pairs.sort(key=lambda P: P.perturbed_cell_id)

    digest = _compute_pairing_digest(
        random_seed=random_seed,
        stratum_key_field=stratum_key_field,
        with_replacement=with_replacement,
        control_ids=control_ids,
        perturbed_ids=perturbed_ids,
        pairs=pairs,
    )

    return PairingResult(
        pairs=pairs,
        random_seed=int(random_seed),
        control_ids=control_ids,
        perturbed_ids=perturbed_ids,
        stratum_key_field=stratum_key_field,
        with_replacement=with_replacement,
        digest=digest,
    )


def pair_control_inputs(
    query_controls: Sequence[CellDescriptor],
    pool_controls: Sequence[CellDescriptor],
    random_seed: int,
    *,
    stratum_key_field: str = "cell_line",
    with_replacement: bool = True,
) -> ControlInputPairingResult:
    """Sample one control input per control query within its stratum.

    For control reconstruction, use the sampled cell as both the input and supervision
    target. Seed this operation with ``control_input_pair_seed``.
    """
    query_sorted = _stable_sort_cells(list(query_controls))
    pool_sorted = _stable_sort_cells(list(pool_controls))

    if not pool_sorted:
        raise ValueError("The control pool is empty. At least one control cell is required for control input pairing.")
    if not query_sorted:
        return ControlInputPairingResult(
            pairs=[],
            random_seed=int(random_seed),
            query_ids=[],
            pool_control_ids=[c.cell_id for c in pool_sorted],
            stratum_key_field=stratum_key_field,
            with_replacement=with_replacement,
            digest=_compute_control_input_digest(
                random_seed=random_seed,
                stratum_key_field=stratum_key_field,
                with_replacement=with_replacement,
                query_ids=[],
                pool_control_ids=[c.cell_id for c in pool_sorted],
                pairs=[],
            ),
        )

    query_ids = [c.cell_id for c in query_sorted]
    pool_ids = [c.cell_id for c in pool_sorted]

    pool_by_stratum: dict[str, list[int]] = {}
    for i, c in enumerate(pool_sorted):
        s = _stratum_key(c, key_field=stratum_key_field)
        pool_by_stratum.setdefault(s, []).append(i)

    query_by_stratum: dict[str, list[int]] = {}
    for i, q in enumerate(query_sorted):
        s = _stratum_key(q, key_field=stratum_key_field)
        query_by_stratum.setdefault(s, []).append(i)

    rng = np.random.default_rng(int(random_seed))
    pairs: list[ControlInputPair] = []

    for stratum in sorted(query_by_stratum.keys()):
        query_indices = query_by_stratum[stratum]
        pool_indices = pool_by_stratum.get(stratum)
        if not pool_indices:
            raise ValueError(
                f"Stratum {stratum!r} contains {len(query_indices)} control queries and no eligible input controls. "
                f"Pairing requires controls in the same stratum (stratum_key_field={stratum_key_field!r})."
            )

        n_pool = len(pool_indices)
        n_query = len(query_indices)

        if not with_replacement and n_pool < n_query:
            raise ValueError(
                f"Pairing without replacement requires at least as many input controls as control queries in stratum {stratum!r}. "
                f"Found {n_pool} input controls and {n_query} control queries."
            )

        if with_replacement:
            choices = rng.integers(0, n_pool, size=n_query, endpoint=False)
        else:
            perm = rng.permutation(n_pool)
            choices = perm[:n_query]

        for local_qi, qi in enumerate(query_indices):
            pi = pool_indices[int(choices[local_qi])]
            pairs.append(
                ControlInputPair(
                    query_cell_id=query_ids[qi],
                    input_control_cell_id=pool_ids[pi],
                    query_index=qi,
                    input_control_index=pi,
                    stratum=stratum,
                )
            )

    pairs.sort(key=lambda p: p.query_cell_id)

    digest = _compute_control_input_digest(
        random_seed=random_seed,
        stratum_key_field=stratum_key_field,
        with_replacement=with_replacement,
        query_ids=query_ids,
        pool_control_ids=pool_ids,
        pairs=pairs,
    )

    return ControlInputPairingResult(
        pairs=pairs,
        random_seed=int(random_seed),
        query_ids=query_ids,
        pool_control_ids=pool_ids,
        stratum_key_field=stratum_key_field,
        with_replacement=with_replacement,
        digest=digest,
    )
