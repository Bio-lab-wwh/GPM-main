"""GPM architecture for cell-context-conditioned perturbation prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
import torch
import torch.nn as nn

from GPM.causal_inference import (
    CandidateEdgeTable,
    build_candidate_edge_table,
)
from GPM.gene_decoder import DirectPerturbationEffectDecoder
from GPM.paths import project_path
from GPM.perturbation import (
    GraphConditionedResponsePropagator,
    HybridGraphEncoder,
    PerturbationTechnologyEmbedding,
    StructuredPerturbationGenerator,
)


@dataclass
class GPMForwardOutput:
    pred_perturbed_expression: torch.Tensor
    pred_delta_expression: torch.Tensor
    is_control: torch.Tensor
    response_rms: torch.Tensor
    gene_background: Optional[torch.Tensor] = None
    gene_context: Optional[torch.Tensor] = None
    intervention: Optional[torch.Tensor] = None
    propagated_response: Optional[torch.Tensor] = None


class ControlContextSetEncoder(nn.Module):
    """Encode control statistics and pool a global context vector for each sample."""

    def __init__(self, feature_dim: int, context_dim: int) -> None:
        super().__init__()
        hidden = int(feature_dim)
        self.statistics_projection = nn.Linear(3, hidden, bias=False)
        self.attention = nn.Sequential(
            nn.Linear(hidden, max(16, hidden // 2), bias=False),
            nn.Tanh(),
            nn.Linear(max(16, hidden // 2), 1, bias=False),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, int(context_dim), bias=False),
            nn.SiLU(),
            nn.Linear(int(context_dim), int(context_dim), bias=False),
        )

    def forward(
        self,
        control_expression: torch.Tensor,
        control_mean: torch.Tensor,
        control_variance: torch.Tensor,
        gene_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if control_expression.dim() != 2:
            raise ValueError("Paired control expression must have shape (batch, gene).")
        if control_mean.shape != control_variance.shape or control_mean.dim() not in (1, 2):
            raise ValueError("Control context moments must have shape (gene,) or (batch, gene).")
        batch_size, num_genes = control_expression.shape
        if control_mean.dim() == 1:
            control_mean = control_mean.unsqueeze(0).expand(batch_size, -1)
            control_variance = control_variance.unsqueeze(0).expand(batch_size, -1)
        elif control_mean.shape != control_expression.shape:
            raise ValueError("Per-sample control moments must match the shape of the paired control expression.")
        if gene_features.shape[0] != num_genes:
            raise ValueError("The number of gene identities does not match the expression panel.")

        mean = control_mean.to(device=control_expression.device, dtype=torch.float32)
        variance = control_variance.to(
            device=control_expression.device, dtype=torch.float32
        ).clamp_min(0.0)
        expression = control_expression.float()
        gene_statistics = torch.stack(
            (mean, variance, expression - mean), dim=-1
        )
        nodes = self.statistics_projection(gene_statistics)
        nodes = nodes + gene_features.unsqueeze(0).to(dtype=nodes.dtype)
        weights = torch.softmax(self.attention(nodes).squeeze(-1), dim=1)
        global_context = self.output((weights.unsqueeze(-1) * nodes).sum(dim=1))
        return global_context, gene_statistics


class PerGeneBackgroundCrossAttention(nn.Module):
    """Use gene identities to attend to local statistics and global context tokens."""

    def __init__(
        self, gene_feature_dim: int, node_dim: int, attention_dim: int = 32
    ) -> None:
        super().__init__()
        self.attention_dim = int(attention_dim)
        self.query = nn.Linear(gene_feature_dim, self.attention_dim, bias=False)
        self.key = nn.Linear(node_dim, self.attention_dim, bias=False)
        self.value = nn.Linear(node_dim, self.attention_dim, bias=False)
        self.output = nn.Linear(self.attention_dim, node_dim, bias=False)

    def forward(
        self, gene_features: torch.Tensor, background_tokens: torch.Tensor
    ) -> torch.Tensor:
        if background_tokens.ndim != 4 or background_tokens.shape[2] != 2:
            raise ValueError("Each gene must have two tokens: one for control statistics and one for global context.")
        batch_size, num_genes = background_tokens.shape[:2]
        if gene_features.shape[0] != num_genes:
            raise ValueError("Gene identities and background tokens must cover the same number of genes.")

        query = self.query(gene_features).view(1, num_genes, 1, -1)
        query = query.expand(batch_size, -1, -1, -1)
        key = self.key(background_tokens)
        value = self.value(background_tokens)
        scores = (query * key).sum(dim=-1) / math.sqrt(self.attention_dim)
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * value).sum(dim=2)
        return self.output(context)


class GeneCenteredBackgroundEncoder(nn.Module):
    """Fuse local expression statistics and global cell context around each gene identity."""

    def __init__(
        self,
        gene_feature_dim: int,
        context_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        output = int(output_dim)
        self.output_dim = output
        self.identity_projection = nn.Linear(
            int(gene_feature_dim), output, bias=False
        )
        self.statistics_projection = nn.Linear(3, output, bias=False)
        self.context_projection = nn.Linear(int(context_dim), output, bias=False)
        self.token_type_embeddings = nn.Parameter(torch.empty(2, output))
        nn.init.normal_(self.token_type_embeddings, mean=0.0, std=0.02)
        self.token_norm = nn.LayerNorm(output)
        self.cross_attention = PerGeneBackgroundCrossAttention(
            int(gene_feature_dim), output
        )
        initial_gate_logit = math.log(0.1 / 0.9)
        self.fusion_gate_logit = nn.Parameter(torch.tensor(initial_gate_logit))
        self.normalization = nn.LayerNorm(output)
        self.refine = nn.Sequential(
            nn.Linear(output, output, bias=False),
            nn.SiLU(),
            nn.Linear(output, output, bias=False),
        )

    def forward(
        self,
        gene_features: torch.Tensor,
        gene_statistics: torch.Tensor,
        cell_context: torch.Tensor,
    ) -> torch.Tensor:
        if gene_statistics.ndim != 3 or gene_statistics.shape[-1] != 3:
            raise ValueError("Gene-level control statistics must have shape (batch, gene, 3).")
        batch_size, num_genes = gene_statistics.shape[:2]
        if cell_context.shape[0] != batch_size or gene_features.shape[0] != num_genes:
            raise ValueError("Gene statistics, global context, and gene identities must agree on the batch size or gene panel where applicable.")

        identity = self.identity_projection(gene_features).unsqueeze(0)
        statistics_token = self.statistics_projection(gene_statistics)
        cell_token = self.context_projection(cell_context).unsqueeze(1)
        cell_token = cell_token.expand(-1, num_genes, -1)
        tokens = torch.stack((statistics_token, cell_token), dim=2)
        tokens = self.token_norm(
            tokens + self.token_type_embeddings.view(1, 1, 2, -1)
        )
        background_update = self.cross_attention(gene_features, tokens)
        gate = torch.sigmoid(self.fusion_gate_logit).to(
            dtype=background_update.dtype
        )
        # Initialize the gate near zero to preserve the gene identity signal early in training.
        background = self.normalization(
            identity.to(dtype=background_update.dtype) + gate * background_update
        )
        return background + self.refine(background)


class GPMModel(nn.Module):
    def __init__(
        self,
        gene_names: Sequence[str],
        gene_embedding: nn.Embedding,
        gene_embedding_source: str,
        cell_context_encoder: ControlContextSetEncoder,
        node_encoder: GeneCenteredBackgroundEncoder,
        graph_encoder: HybridGraphEncoder,
        technology_embedding: PerturbationTechnologyEmbedding,
        intervention_generator: StructuredPerturbationGenerator,
        propagator: GraphConditionedResponsePropagator,
        expression_decoder: DirectPerturbationEffectDecoder,
    ) -> None:
        super().__init__()
        names = tuple(str(gene).strip() for gene in gene_names)
        if not names or any(not name for name in names):
            raise ValueError("gene_names must be a non-empty fixed panel.")
        if len({name.upper() for name in names}) != len(names):
            raise ValueError("The gene panel contains case-insensitive duplicates.")
        self._panel_gene_names = names
        self.gene_embedding = gene_embedding
        self.gene_embedding_source = str(gene_embedding_source)
        self.cell_context_encoder = cell_context_encoder
        self.node_encoder = node_encoder
        self.graph_encoder = graph_encoder
        self.technology_embedding = technology_embedding
        self.intervention_generator = intervention_generator
        self.propagator = propagator
        self.expression_decoder = expression_decoder

    @property
    def technology_names(self) -> tuple[str, ...]:
        return self.technology_embedding.tech_names

    @property
    def gene_names(self) -> tuple[str, ...]:
        return self._panel_gene_names

    def set_gene_panel(self, gene_names: Sequence[str]) -> None:
        names = tuple(str(gene).strip() for gene in gene_names)
        if names != self._panel_gene_names:
            raise ValueError("The model gene panel is fixed at construction time.")

    def _validate_expression(
        self, expression: torch.Tensor, gene_names: Optional[Sequence[str]]
    ) -> torch.Tensor:
        if gene_names is not None:
            self.set_gene_panel(gene_names)
        if expression.dim() == 1:
            expression = expression.unsqueeze(0)
        if expression.dim() != 2 or expression.shape[1] != len(self._panel_gene_names):
            raise ValueError("control_expression must match the fixed gene panel.")
        return expression

    def _encode_control(
        self,
        control_expression: torch.Tensor,
        control_context_mean: torch.Tensor,
        control_context_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gene_ids = torch.arange(len(self._panel_gene_names), device=control_expression.device)
        gene_features = self.gene_embedding(gene_ids)
        cell_context, gene_statistics = self.cell_context_encoder(
            control_expression,
            control_context_mean,
            control_context_variance,
            gene_features,
        )
        if cell_context.shape[0] != control_expression.shape[0]:
            raise ValueError("Cell context batch does not match control expression batch.")
        gene_background = self.node_encoder(
            gene_features,
            gene_statistics,
            cell_context,
        )
        return gene_background, cell_context, gene_features, gene_statistics

    @torch.no_grad()
    def infer_causal_relations(
        self,
        control_expression: torch.Tensor,
        *,
        gene_names: Optional[Sequence[str]] = None,
    ) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("Call eval() before exporting causal relations.")
        control_expression = self._validate_expression(control_expression, gene_names)
        # This legacy interface exports fixed BioGRID interactions.
        return self.graph_encoder.export_relations()

    def forward(
        self,
        control_expression: torch.Tensor,
        pert_tech_ids: torch.Tensor,
        target_gene_indices: torch.Tensor,
        *,
        control_context_mean: torch.Tensor,
        control_context_variance: torch.Tensor,
        gene_names: Optional[Sequence[str]] = None,
        is_control: Optional[torch.Tensor] = None,
        target_groups: Optional[
            Sequence[tuple[tuple[int, ...], Sequence[int]]]
        ] = None,
        return_intermediates: bool = False,
    ) -> GPMForwardOutput:
        control_expression = self._validate_expression(control_expression, gene_names)
        batch_size = control_expression.shape[0]
        if pert_tech_ids.shape != (batch_size,) or target_gene_indices.dim() != 2:
            raise ValueError("Perturbation inputs have incompatible shapes.")
        if is_control is None:
            is_control = (target_gene_indices >= 0).sum(dim=1) == 0
        else:
            is_control = is_control.to(control_expression.device, dtype=torch.bool)
        gene_background, _, _, gene_statistics = self._encode_control(
            control_expression, control_context_mean, control_context_variance,
        )

        graph_enhanced_state = self.graph_encoder(gene_background)
        technology_features = self.technology_embedding(pert_tech_ids)
        intervention = self.intervention_generator(
            graph_enhanced_state,
            technology_features,
            target_gene_indices,
            is_control=is_control,
        )
        propagation = self.propagator(
            graph_enhanced_state,
            intervention.intervention,
            target_gene_indices,
            target_groups=target_groups,
        )
        pred_delta = self.expression_decoder(propagation.total_response)
        pred_delta = torch.where(is_control.unsqueeze(1), torch.zeros_like(pred_delta), pred_delta)
        # Anchor predicted expression to the control population mean.
        decoder_baseline = control_context_mean
        if decoder_baseline.dim() == 1:
            if decoder_baseline.shape[0] != control_expression.shape[1]:
                raise ValueError("control_context_mean does not match the gene panel.")
            decoder_baseline = decoder_baseline.unsqueeze(0)
        if decoder_baseline.dim() != 2 or decoder_baseline.shape[1] != control_expression.shape[1]:
            raise ValueError("control_context_mean must have shape (gene,) or (batch, gene).")
        if decoder_baseline.shape[0] == 1 and batch_size != 1:
            decoder_baseline = decoder_baseline.expand(batch_size, -1)
        elif decoder_baseline.shape[0] != batch_size:
            raise ValueError("Control context mean batch does not match control expression batch.")
        pred_perturbed = decoder_baseline.to(
            device=pred_delta.device, dtype=pred_delta.dtype
        ) + pred_delta

        response_rms = propagation.total_response.float().square().mean().sqrt().detach()
        return GPMForwardOutput(
            pred_perturbed_expression=pred_perturbed,
            pred_delta_expression=pred_delta,
            is_control=is_control,
            response_rms=response_rms,
            gene_background=graph_enhanced_state if return_intermediates else None,
            gene_context=gene_statistics if return_intermediates else None,
            intervention=intervention.intervention if return_intermediates else None,
            propagated_response=(
                propagation.total_response if return_intermediates else None
            ),
        )


def resolve_gene_embedding_path(
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the path to the pretrained gene identity embeddings."""
    return (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent / "gene_embedding" / "gene_embedding.pt"
    )


def _build_gene_identity_embedding(
    gene_names: Sequence[str],
    *,
    embedding_dim: int,
    path: Optional[Union[str, Path]],
) -> tuple[nn.Embedding, str]:
    """Initialize gene identities by symbol, with a trainable random fallback."""
    names = tuple(str(name).strip() for name in gene_names)
    embedding = nn.Embedding(len(names), int(embedding_dim))
    with torch.no_grad():
        nn.init.normal_(
            embedding.weight,
            mean=0.0,
            std=float(embedding_dim) ** -0.5,
        )
        embedding.weight.copy_(
            torch.nn.functional.normalize(embedding.weight, p=2, dim=-1)
        )

    source = resolve_gene_embedding_path(path)
    if not source.is_file():
        print(
            f"[gene identity] Pretrained embeddings not found at {source}. "
            f"All {len(names)} gene embeddings were randomly initialized and will be trained with the model.",
            flush=True,
        )
        return embedding, "random_initialization"

    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected_method = "biogrid_contrastive_graph_encoder"
    if payload.get("embedding_method") != expected_method:
        raise ValueError(
            f"The gene embeddings at {source} do not use the expected BioGRID contrastive graph encoder format. "
            "Regenerate the embeddings with GPM/pre_train/train_embedding.py."
        )
    matrix = torch.as_tensor(payload.get("embeddings"), dtype=torch.float32)
    stored_names = [str(name).strip().upper() for name in payload.get("gene_names", ())]
    if matrix.dim() != 2 or matrix.shape[0] != len(stored_names):
        raise ValueError(f"The pretrained embedding matrix shape does not match the gene list: {source}")
    if matrix.shape[1] != int(embedding_dim):
        raise ValueError(
            f"Pretrained embeddings have dimension {matrix.shape[1]}, "
            f"but gene_embedding_dim is {embedding_dim}: {source}"
        )
    lookup = {name: index for index, name in enumerate(stored_names)}
    if len(lookup) != len(stored_names):
        raise ValueError(f"The pretrained gene embeddings contain duplicate gene symbols: {source}")
    matched = 0
    with torch.no_grad():
        for row, name in enumerate(names):
            source_row = lookup.get(name.upper())
            if source_row is None:
                continue
            vector = matrix[source_row]
            if not torch.isfinite(vector).all():
                raise ValueError(f"The pretrained embedding for gene {name} contains NaN or Inf: {source}")
            embedding.weight[row].copy_(vector)
            matched += 1
    print(
        f"[gene identity] Loaded embeddings from {source} for {matched}/{len(names)} genes. "
        f"Randomly initialized the remaining {len(names) - matched} embeddings. All embeddings are trainable.",
        flush=True,
    )
    return embedding, project_path(source)


def build_gpm_model(
    *,
    gene_names: Sequence[str],
    gene_embedding_dim: int = 128,
    cell_context_dim: int = 256,
    enhanced_node_dim: int = 128,
    edge_chunk_size: int = 131072,
    technology_names: Sequence[str],
    technology_embedding_dim: int = 128,
    biogrid_path: Optional[Union[str, Path]] = None,
    gene_embedding_path: Optional[Union[str, Path]] = None,
    candidate_edge_table: Optional[CandidateEdgeTable] = None,
) -> GPMModel:
    names = tuple(str(gene).strip() for gene in gene_names)
    table = candidate_edge_table or build_candidate_edge_table(
        names, biogrid_path=biogrid_path
    )
    if table.gene_names != names:
        raise ValueError("Candidate edges and the model must use the same gene panel.")
    feature_dim = int(gene_embedding_dim)
    context_dim = int(cell_context_dim)
    node_dim = int(enhanced_node_dim)
    gene_embedding, embedding_source = _build_gene_identity_embedding(
        names,
        embedding_dim=feature_dim,
        path=gene_embedding_path,
    )
    context_encoder = ControlContextSetEncoder(feature_dim, context_dim)
    node_encoder = GeneCenteredBackgroundEncoder(
        gene_feature_dim=feature_dim,
        context_dim=context_dim,
        output_dim=node_dim,
    )
    graph_encoder = HybridGraphEncoder(table, node_dim)
    technology = PerturbationTechnologyEmbedding(
        technology_names, int(technology_embedding_dim)
    )
    intervention_generator = StructuredPerturbationGenerator(
        node_dim,
        technology.embedding_dim,
    )
    propagator = GraphConditionedResponsePropagator(
        node_dim=node_dim,
        receiver_chunk_size=edge_chunk_size,
    )
    decoder = DirectPerturbationEffectDecoder(node_dim)
    return GPMModel(
        names,
        gene_embedding,
        embedding_source,
        context_encoder,
        node_encoder,
        graph_encoder,
        technology,
        intervention_generator,
        propagator,
        decoder,
    )
