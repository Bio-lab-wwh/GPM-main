"""Relation-aware contrastive pretraining of BioGRID gene embeddings."""

from __future__ import annotations

import csv
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PRIOR_EMBED_DIM = 128
DEFAULT_HIDDEN_DIM = 128
DEFAULT_TEMPERATURE = 0.1
DEFAULT_NUM_NEGATIVES = 16
DEFAULT_NEGATIVE_OVERSAMPLE = 2
DEFAULT_BIOGRID_FILENAME = "BIOGRID.tsv"
DEFAULT_OUTPUT_FILENAME = "gene_embedding.pt"
EMBEDDING_METHOD = "biogrid_contrastive_graph_encoder"
EVIDENCE_NAMES = ("physical", "genetic")


def _embedding_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "gene_embedding"


def resolve_biogrid_path(path: str | Path | None = None) -> Path:
    """Resolve the BioGRID file used for pretraining."""
    result = Path(path) if path is not None else _embedding_dir() / DEFAULT_BIOGRID_FILENAME
    if not result.is_file():
        raise FileNotFoundError(f"BioGRID interaction file not found: {result}")
    return result


def resolve_output_path(
    path: str | Path | None = None,
    *,
    biogrid_path: str | Path | None = None,
) -> Path:
    """Default to writing the embeddings alongside the BioGRID input file."""
    if path is not None:
        return Path(path)
    return resolve_biogrid_path(biogrid_path).with_name(DEFAULT_OUTPUT_FILENAME)


def _normalized_columns(fieldnames: Sequence[str] | None) -> dict[str, str]:
    return {
        str(name).lstrip("#").strip().casefold(): str(name)
        for name in (fieldnames or ())
    }


def load_biogrid_interactions(
    path: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read undirected interactions and retain both relation labels for each pair."""
    source = Path(path)
    pair_evidence: dict[tuple[str, str], list[bool]] = {}
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = _normalized_columns(reader.fieldnames)
        required = (
            "official symbol interactor a",
            "official symbol interactor b",
            "experimental system type",
        )
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"BIOGRID.tsv is missing required columns: {missing}")
        left_col = columns[required[0]]
        right_col = columns[required[1]]
        type_col = columns[required[2]]
        for row in reader:
            relation = str(row.get(type_col, "") or "").strip().casefold()
            if relation not in EVIDENCE_NAMES:
                continue
            left = str(row.get(left_col, "") or "").strip().upper()
            right = str(row.get(right_col, "") or "").strip().upper()
            if not left or not right or left == "-" or right == "-" or left == right:
                continue
            pair = tuple(sorted((left, right)))
            flags = pair_evidence.setdefault(pair, [False, False])
            flags[EVIDENCE_NAMES.index(relation)] = True
    if not pair_evidence:
        raise ValueError(f"No eligible physical or genetic interactions were found in the BioGRID file: {source}")

    ordered_pairs = sorted(pair_evidence)
    genes = sorted({gene for pair in ordered_pairs for gene in pair})
    lookup = {gene: index for index, gene in enumerate(genes)}
    edges = np.asarray(
        [(lookup[left], lookup[right]) for left, right in ordered_pairs],
        dtype=np.int64,
    )
    evidence = np.asarray(
        [pair_evidence[pair] for pair in ordered_pairs], dtype=np.float32
    )
    return genes, edges, evidence


class TypedLightweightGraphEncoder(nn.Module):
    """Encode genes with shared embeddings and separate physical and genetic aggregators."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        directed_edge_index: torch.Tensor,
        directed_edge_evidence: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.node_embedding = nn.Embedding(self.num_nodes, int(hidden_dim))
        self.self_projection = nn.Linear(int(hidden_dim), self.embedding_dim, bias=False)
        self.relation_projections = nn.ModuleList([
            nn.Linear(int(hidden_dim), self.embedding_dim, bias=False)
            for _ in EVIDENCE_NAMES
        ])
        self.refine = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
        )
        # Initialize both relation-specific scoring transforms at the same scale.
        self.relation_scale = nn.Parameter(
            torch.zeros(len(EVIDENCE_NAMES), self.embedding_dim)
        )
        nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.02)

        if directed_edge_index.dim() != 2 or directed_edge_index.shape[0] != 2:
            raise ValueError("directed_edge_index must have shape (2, E).")
        if directed_edge_evidence.shape != (
            directed_edge_index.shape[1], len(EVIDENCE_NAMES)
        ):
            raise ValueError("directed_edge_evidence has an incompatible shape.")

        for relation_id, relation_name in enumerate(EVIDENCE_NAMES):
            mask = directed_edge_evidence[:, relation_id] > 0
            relation_edges = directed_edge_index[:, mask].contiguous()
            degree = torch.bincount(
                relation_edges[1], minlength=self.num_nodes
            ).to(dtype=torch.float32)
            inverse_degree = degree.clamp_min(1.0).reciprocal()
            self.register_buffer(
                f"{relation_name}_edge_index", relation_edges, persistent=False
            )
            self.register_buffer(
                f"{relation_name}_inverse_degree",
                inverse_degree,
                persistent=False,
            )

    def forward(self) -> torch.Tensor:
        x = self.node_embedding.weight
        encoded = self.self_projection(x)
        for relation_id, relation_name in enumerate(EVIDENCE_NAMES):
            edge_index = getattr(self, f"{relation_name}_edge_index")
            inverse_degree = getattr(self, f"{relation_name}_inverse_degree")
            source, target = edge_index
            neighbour_sum = torch.zeros_like(x)
            neighbour_sum.index_add_(0, target, x.index_select(0, source))
            neighbour_mean = neighbour_sum * inverse_degree.to(x.dtype).unsqueeze(1)
            encoded = encoded + self.relation_projections[relation_id](neighbour_mean)
        encoded = encoded + self.refine(encoded)
        return F.normalize(encoded, p=2, dim=-1)

    def relation_condition(
        self, anchors: torch.Tensor, relation_ids: torch.Tensor
    ) -> torch.Tensor:
        scale = 1.0 + torch.tanh(self.relation_scale.index_select(0, relation_ids))
        return anchors * scale.to(dtype=anchors.dtype)


# Preserve the legacy class name for existing imports.
LightweightGraphEncoder = TypedLightweightGraphEncoder


@dataclass
class ContrastiveTrainHistory:
    loss: list[float]
    physical_loss: list[float]
    genetic_loss: list[float]


def _directed_graph_arrays(
    undirected_edges: np.ndarray,
    edge_evidence: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    both_directions = np.concatenate(
        (undirected_edges, undirected_edges[:, ::-1]), axis=0
    )
    directed_evidence = np.concatenate((edge_evidence, edge_evidence), axis=0)
    return (
        torch.as_tensor(both_directions.T, dtype=torch.long),
        torch.as_tensor(directed_evidence, dtype=torch.float32),
    )


def _typed_positive_arrays(
    undirected_edges: np.ndarray,
    edge_evidence: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand each undirected edge into directed positives for every relation it carries."""
    pair_rows, relation_ids = np.nonzero(edge_evidence > 0)
    typed_edges = undirected_edges[pair_rows]
    both_directions = np.concatenate((typed_edges, typed_edges[:, ::-1]), axis=0)
    both_relations = np.concatenate((relation_ids, relation_ids), axis=0)
    return (
        torch.as_tensor(both_directions, dtype=torch.long),
        torch.as_tensor(both_relations, dtype=torch.long),
    )


@torch.no_grad()
def _sample_negative_nodes(
    anchors: torch.Tensor,
    *,
    num_nodes: int,
    known_edge_keys: torch.Tensor,
    noise_weights: torch.Tensor,
    num_negatives: int,
    oversample_factor: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample negative genes while excluding known interactions of either relation."""
    batch_size = int(anchors.numel())
    count = int(num_negatives)
    draw_count = max(count + 8, count * max(1, int(oversample_factor)))
    for _ in range(4):
        candidates = torch.multinomial(
            noise_weights,
            batch_size * draw_count,
            replacement=True,
            generator=generator,
        ).view(batch_size, draw_count)
        candidate_keys = anchors.unsqueeze(1) * int(num_nodes) + candidates
        flat_keys = candidate_keys.reshape(-1)
        positions = torch.searchsorted(known_edge_keys, flat_keys)
        safe_positions = positions.clamp_max(known_edge_keys.numel() - 1)
        is_known = known_edge_keys.index_select(0, safe_positions).eq(flat_keys)
        is_known = is_known.view(batch_size, draw_count)
        valid = (~is_known) & candidates.ne(anchors.unsqueeze(1))
        if bool((valid.sum(dim=1) >= count).all()):
            order = torch.arange(draw_count, device=anchors.device).view(1, -1)
            priority = torch.where(valid, order, order + draw_count)
            chosen_positions = torch.topk(
                priority, k=count, dim=1, largest=False
            ).indices
            return candidates.gather(1, chosen_positions)
        draw_count *= 2
    raise RuntimeError(
        "Negative sampling could not find enough genes without known interactions for some high-degree anchors. "
        "Decrease --num-negatives or increase --negative-oversample."
    )


def _balanced_relation_batch(
    relation_rows: tuple[torch.Tensor, ...],
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample comparable numbers of positive edges from each relation."""
    active = [rows for rows in relation_rows if rows.numel() > 0]
    if not active:
        raise ValueError("No positive edges with physical or genetic interaction labels are available for training.")
    base = int(batch_size) // len(active)
    remainder = int(batch_size) % len(active)
    selected: list[torch.Tensor] = []
    for relation_index, rows in enumerate(active):
        count = base + int(relation_index < remainder)
        positions = torch.randint(
            rows.numel(),
            (count,),
            device=rows.device,
            generator=generator,
        )
        selected.append(rows.index_select(0, positions))
    result = torch.cat(selected)
    shuffle = torch.randperm(result.numel(), device=result.device, generator=generator)
    return result.index_select(0, shuffle)


def train_contrastive_graph_encoder(
    gene_names: Sequence[str],
    undirected_edges: np.ndarray,
    edge_evidence: np.ndarray,
    *,
    device: str | torch.device = "cuda",
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    embedding_dim: int = PRIOR_EMBED_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    positive_batch_size: int = 8_192,
    num_negatives: int = DEFAULT_NUM_NEGATIVES,
    negative_oversample: int = DEFAULT_NEGATIVE_OVERSAMPLE,
    temperature: float = DEFAULT_TEMPERATURE,
    use_amp: bool = True,
    seed: int = 42,
    print_every: int = 10,
) -> tuple[TypedLightweightGraphEncoder, ContrastiveTrainHistory, torch.Tensor]:
    """Fit the relation-aware graph encoder with balanced InfoNCE."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if int(num_negatives) < 1:
        raise ValueError("num_negatives must be positive.")
    if edge_evidence.shape != (undirected_edges.shape[0], len(EVIDENCE_NAMES)):
        raise ValueError("edge_evidence must have one row per undirected edge and two relation columns.")

    torch.manual_seed(int(seed))
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    generator = torch.Generator(device=dev)
    generator.manual_seed(int(seed) + 1)
    num_nodes = len(gene_names)

    directed_edge_index_cpu, directed_evidence_cpu = _directed_graph_arrays(
        undirected_edges, edge_evidence
    )
    typed_edges_cpu, typed_relations_cpu = _typed_positive_arrays(
        undirected_edges, edge_evidence
    )
    model = TypedLightweightGraphEncoder(
        num_nodes,
        embedding_dim,
        hidden_dim,
        directed_edge_index_cpu,
        directed_evidence_cpu,
    ).to(dev)
    typed_edges = typed_edges_cpu.to(dev)
    typed_relations = typed_relations_cpu.to(dev)
    relation_rows = tuple(
        torch.nonzero(typed_relations == relation_id, as_tuple=False).flatten()
        for relation_id in range(len(EVIDENCE_NAMES))
    )

    directed_edge_index = directed_edge_index_cpu.to(dev)
    known_edge_keys = torch.unique(
        directed_edge_index[0] * int(num_nodes) + directed_edge_index[1]
    ).sort().values
    degree = torch.bincount(
        directed_edge_index[1], minlength=num_nodes
    ).to(dtype=torch.float32)
    noise_weights = (degree + 1.0).pow(0.75)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    amp_enabled = bool(use_amp and dev.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = ContrastiveTrainHistory(loss=[], physical_loss=[], genetic_loss=[])
    batch_size = min(
        max(1, int(positive_batch_size)), int(typed_edges.shape[0])
    )
    relation_counts = edge_evidence.sum(axis=0).astype(np.int64)
    print(
        f"[typed-graph-contrastive] genes={num_nodes}, pairs={undirected_edges.shape[0]}, "
        f"physical={int(relation_counts[0])}, genetic={int(relation_counts[1])}, "
        f"batch={batch_size}, negatives/positive={num_negatives}",
        flush=True,
    )

    for epoch in range(1, int(epochs) + 1):
        chosen = _balanced_relation_batch(relation_rows, batch_size, generator)
        positives = typed_edges.index_select(0, chosen)
        relation_ids = typed_relations.index_select(0, chosen)
        negatives = _sample_negative_nodes(
            positives[:, 0],
            num_nodes=num_nodes,
            known_edge_keys=known_edge_keys,
            noise_weights=noise_weights,
            num_negatives=int(num_negatives),
            oversample_factor=int(negative_oversample),
            generator=generator,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with amp_context:
            encoded = model()
            anchors = model.relation_condition(
                encoded.index_select(0, positives[:, 0]), relation_ids
            )
            positive_vectors = encoded.index_select(0, positives[:, 1])
            negative_vectors = encoded[negatives]
            positive_logits = (anchors * positive_vectors).sum(dim=-1, keepdim=True)
            negative_logits = torch.einsum("bd,bkd->bk", anchors, negative_vectors)
            logits = torch.cat((positive_logits, negative_logits), dim=1) / float(temperature)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=dev)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")

            relation_losses: list[torch.Tensor] = []
            logged_relation_losses: list[float] = []
            for relation_id in range(len(EVIDENCE_NAMES)):
                relation_mask = relation_ids == relation_id
                if bool(relation_mask.any()):
                    relation_loss = per_sample_loss[relation_mask].mean()
                    relation_losses.append(relation_loss)
                    logged_relation_losses.append(float(relation_loss.detach().cpu()))
                else:
                    logged_relation_losses.append(float("nan"))
            loss = torch.stack(relation_losses).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        history.loss.append(float(loss.detach().cpu()))
        history.physical_loss.append(logged_relation_losses[0])
        history.genetic_loss.append(logged_relation_losses[1])
        if epoch == 1 or epoch % max(1, int(print_every)) == 0 or epoch == epochs:
            print(
                f"[typed-graph-contrastive] epoch {epoch:4d}/{epochs} "
                f"loss={history.loss[-1]:.6f} "
                f"physical={history.physical_loss[-1]:.6f} "
                f"genetic={history.genetic_loss[-1]:.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        embeddings = model().float().cpu()
    return model, history, embeddings


def build_gene_embedding_payload(
    gene_names: Sequence[str],
    embeddings: torch.Tensor,
    *,
    source_file: str | Path,
    train_config: dict[str, Any],
    history: ContrastiveTrainHistory,
    relation_condition_scale: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build an embedding payload indexed by canonical gene symbol."""
    names = [str(name).strip().upper() for name in gene_names]
    matrix = embeddings.detach().cpu().to(torch.float32).contiguous()
    if matrix.shape != (len(names), int(train_config["embedding_dim"])):
        raise ValueError("The gene embedding matrix shape does not match the gene list and configured embedding dimension.")
    payload = {
        "version": 4,
        "embedding_method": EMBEDDING_METHOD,
        "edge_types": list(EVIDENCE_NAMES),
        "gene_names": names,
        "gene_to_idx": {name: index for index, name in enumerate(names)},
        "embeddings": matrix,
        "embedding_dim": int(matrix.shape[1]),
        "n_genes": len(names),
        "source_file": Path(source_file).name,
        "contrastive_train_config": dict(train_config),
        "contrastive_final_loss": float(history.loss[-1]),
        "contrastive_final_relation_loss": {
            "physical": float(history.physical_loss[-1]),
            "genetic": float(history.genetic_loss[-1]),
        },
    }
    if relation_condition_scale is not None:
        scale = relation_condition_scale.detach().cpu().to(torch.float32).contiguous()
        if scale.shape != (len(EVIDENCE_NAMES), matrix.shape[1]):
            raise ValueError("Relation conditioning scales must have shape (2, embedding_dim).")
        # Store the relation scales for downstream analysis.
        payload["relation_condition_scale"] = scale
    return payload


def generate_gene_prior_embeddings(
    *,
    biogrid_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    embedding_dim: int = PRIOR_EMBED_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    positive_batch_size: int = 8_192,
    num_negatives: int = DEFAULT_NUM_NEGATIVES,
    negative_oversample: int = DEFAULT_NEGATIVE_OVERSAMPLE,
    temperature: float = DEFAULT_TEMPERATURE,
    use_amp: bool = True,
    seed: int = 42,
    print_every: int = 10,
) -> tuple[dict[str, Any], TypedLightweightGraphEncoder, ContrastiveTrainHistory]:
    source = resolve_biogrid_path(biogrid_path)
    genes, edges, evidence = load_biogrid_interactions(source)
    relation_counts = evidence.sum(axis=0).astype(np.int64)
    config = {
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "embedding_dim": int(embedding_dim),
        "hidden_dim": int(hidden_dim),
        "positive_batch_size": int(positive_batch_size),
        "num_negatives": int(num_negatives),
        "negative_oversample": int(negative_oversample),
        "temperature": float(temperature),
        "use_amp": bool(use_amp),
        "seed": int(seed),
        "edge_types": list(EVIDENCE_NAMES),
        "relation_pair_counts": {
            "physical": int(relation_counts[0]),
            "genetic": int(relation_counts[1]),
        },
        "negative_sampling": "device_vectorized_degree_weighted_non_edge",
        "relation_sampling": "balanced_physical_genetic",
    }
    model, history, embeddings = train_contrastive_graph_encoder(
        genes,
        edges,
        evidence,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        positive_batch_size=positive_batch_size,
        num_negatives=num_negatives,
        negative_oversample=negative_oversample,
        temperature=temperature,
        use_amp=use_amp,
        seed=seed,
        print_every=print_every,
    )
    payload = build_gene_embedding_payload(
        genes,
        embeddings,
        source_file=source,
        train_config=config,
        history=history,
        relation_condition_scale=(
            1.0 + torch.tanh(model.relation_scale.detach())
        ),
    )
    return payload, model, history


def save_gene_embedding_pt(payload: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_gene_embedding_pt(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"gene_names", "embeddings", "embedding_dim"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"The gene embedding file is missing required fields: {sorted(missing)}")
    return payload
