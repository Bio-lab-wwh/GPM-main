# Build the fixed BioGRID interaction graph for a gene panel.

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch


DEFAULT_BIOGRID_FILENAME = "BIOGRID.tsv"
BASE_EVIDENCE_NAMES = ("physical", "genetic")
_TABLE_CACHE: dict[tuple[object, ...], "CandidateEdgeTable"] = {}


def _default_prior_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "gene_embedding" / filename


def resolve_biogrid_path(path: Optional[Union[str, Path]] = None) -> Path:
    candidate = Path(path) if path is not None else _default_prior_path(DEFAULT_BIOGRID_FILENAME)
    if not candidate.is_file():
        raise FileNotFoundError(f"BIOGRID prior not found: {candidate}")
    return candidate


def _normalized_columns(fieldnames: Optional[Sequence[str]]) -> dict[str, str]:
    return {
        str(name).lstrip("#").strip().casefold(): str(name)
        for name in (fieldnames or ())
    }


def _value(row: dict[str, str], columns: dict[str, str], name: str) -> str:
    column = columns.get(name.casefold())
    return str(row.get(column, "") or "").strip() if column else ""


@dataclass
class _PairEvidence:
    physical: bool = False
    genetic: bool = False

    def add_biogrid(self, relation: str) -> None:
        if relation == "physical":
            self.physical = True
        elif relation == "genetic":
            self.genetic = True


@dataclass(frozen=True)
class CandidateEdgeTable:
    gene_names: tuple[str, ...]
    edge_index: torch.Tensor  # (2, E), sender -> receiver
    evidence: torch.Tensor  # (E, 2), with physical and genetic interaction indicators
    evidence_names: tuple[str, ...]
    num_components: int = 1
    component_sizes: tuple[int, ...] = ()
    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

def _connected_components(
    num_nodes: int,
    senders: Sequence[int],
    receivers: Sequence[int],
) -> list[tuple[int, ...]]:
    adjacency: list[set[int]] = [set() for _ in range(int(num_nodes))]
    for source, target in zip(senders, receivers):
        adjacency[int(source)].add(int(target))
        adjacency[int(target)].add(int(source))

    unseen = set(range(int(num_nodes)))
    components: list[tuple[int, ...]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue: deque[int] = deque((start,))
        component: list[int] = []
        while queue:
            source = queue.popleft()
            component.append(source)
            for target in adjacency[source]:
                if target in unseen:
                    unseen.remove(target)
                    queue.append(target)
        components.append(tuple(sorted(component)))
    return components


def _read_biogrid(path: Path) -> dict[tuple[str, str], _PairEvidence]:
    pairs: dict[tuple[str, str], _PairEvidence] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = _normalized_columns(reader.fieldnames)
        required = (
            "official symbol interactor a",
            "official symbol interactor b",
            "experimental system type",
        )
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"BIOGRID.tsv is missing columns: {missing}")
        for row in reader:
            relation = _value(row, columns, "experimental system type").casefold()
            if relation not in {"physical", "genetic"}:
                continue
            left = _value(row, columns, "official symbol interactor a").upper()
            right = _value(row, columns, "official symbol interactor b").upper()
            if not left or not right or left == "-" or right == "-" or left == right:
                continue
            pair = tuple(sorted((left, right)))
            pairs.setdefault(pair, _PairEvidence()).add_biogrid(relation)
    return pairs


# The panel may include isolated genes and multiple connected components.
def build_candidate_edge_table(
    gene_names: Sequence[str],
    *,
    biogrid_path: Optional[Union[str, Path]] = None,
) -> CandidateEdgeTable:
    names = tuple(str(gene).strip() for gene in gene_names)
    lookup = {gene.upper(): index for index, gene in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError("The expression panel contains duplicate gene symbols.")

    biogrid_source = resolve_biogrid_path(biogrid_path).resolve()
    biogrid_stat = biogrid_source.stat()
    cache_key = (
        names,
        str(biogrid_source),
        biogrid_stat.st_size,
        biogrid_stat.st_mtime_ns,
    )
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pairs = _read_biogrid(biogrid_source)

    senders: list[int] = []
    receivers: list[int] = []
    base_features: list[list[float]] = []
    for left, right in sorted(pairs):
        if left not in lookup or right not in lookup:
            continue
        prior = pairs[(left, right)]
        binary_evidence = [float(prior.physical), float(prior.genetic)]
        # Store both directions of each undirected interaction.
        for source, target in ((left, right), (right, left)):
            senders.append(lookup[source])
            receivers.append(lookup[target])
            base_features.append(binary_evidence)
    components = _connected_components(len(names), senders, receivers)
    component_sizes = tuple(sorted(len(component) for component in components))
    evidence_names = BASE_EVIDENCE_NAMES
    edge_index = (
        torch.tensor([senders, receivers], dtype=torch.long)
        if senders
        else torch.empty((2, 0), dtype=torch.long)
    )
    evidence = (
        torch.tensor(base_features, dtype=torch.float32)
        if base_features
        else torch.empty((0, len(evidence_names)), dtype=torch.float32)
    )
    order = torch.argsort(edge_index[1] * len(names) + edge_index[0])
    table = CandidateEdgeTable(
        names,
        edge_index[:, order],
        evidence[order],
        tuple(evidence_names),
        len(components),
        component_sizes,
    )
    print(
        f"[candidate graph] genes={len(names)}, ppi_directed_edges={table.num_edges}, "
        f"components={table.num_components}, "
        f"largest_component={max(table.component_sizes, default=0)}, "
        f"isolated_nodes={sum(size == 1 for size in table.component_sizes)}, "
        "evidence=(physical, genetic), synthetic_edges=0"
    )
    _TABLE_CACHE[cache_key] = table
    return table
