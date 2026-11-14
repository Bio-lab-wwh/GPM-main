from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from GPM.conditions import normalize_tech_name
from GPM.causal_inference import CandidateEdgeTable


class PerturbationTechnologyEmbedding(nn.Module):
    """Learn an embedding for each perturbation technology."""
    def __init__(self, tech_names: Sequence[str], embedding_dim: int) -> None:
        super().__init__()
        names = tuple(dict.fromkeys(normalize_tech_name(name) for name in tech_names))
        if not names:
            raise ValueError("tech_names cannot be empty.")
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.tech_names = names
        self.embedding_dim = int(embedding_dim)
        self._name_to_id = {name: idx for idx, name in enumerate(names)}
        self.embedding = nn.Embedding(len(names), self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def tech_name_to_id(self, tech: str) -> int:
        key = normalize_tech_name(tech)
        if key not in self._name_to_id:
            raise KeyError(f"Unknown perturbation technology {tech!r}: {self.tech_names}")
        return self._name_to_id[key]

    def forward(self, tech_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(tech_ids.long())


@dataclass
class PerturbationInterventionOutput:
    intervention: torch.Tensor


@dataclass
class PerturbationPropagationOutput:
    total_response: torch.Tensor


class StructuredPerturbationGenerator(nn.Module):
    """Generate intervention vectors at target genes from gene state and technology."""

    def __init__(self, node_dim: int, technology_dim: int) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.technology_dim = int(technology_dim)
        hidden_dim = 2 * self.node_dim
        self.gene_parent = nn.Linear(
            self.node_dim, hidden_dim, bias=False
        )
        self.technology_parent = nn.Linear(
            self.technology_dim, hidden_dim, bias=False
        )
        self.structural_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.intervention_output = nn.Linear(
            hidden_dim, self.node_dim, bias=False
        )

    def forward(
        self,
        graph_enhanced_state: torch.Tensor,
        technology_features: torch.Tensor,
        target_gene_indices: torch.Tensor,
        *,
        is_control: Optional[torch.Tensor] = None,
    ) -> PerturbationInterventionOutput:
        if graph_enhanced_state.dim() != 3 or target_gene_indices.dim() != 2:
            raise ValueError("Invalid shape for graph-enhanced gene states or perturbation target indices.")
        batch_size, num_genes, feature_dim = graph_enhanced_state.shape
        if feature_dim != self.node_dim:
            raise ValueError("The graph-enhanced gene state dimension does not match the intervention generator configuration.")
        if technology_features.shape != (batch_size, self.technology_dim):
            raise ValueError("Perturbation technology embeddings must match the batch size and configured embedding dimension.")

        device = graph_enhanced_state.device
        targets = target_gene_indices.to(device=device, dtype=torch.long)
        controls = (
            is_control.to(device=device, dtype=torch.bool)
            if is_control is not None
            else torch.zeros(batch_size, device=device, dtype=torch.bool)
        )
        valid = (targets >= 0) & (targets < num_genes) & (~controls.unsqueeze(1))
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(targets)
        rows, genes = rows[valid], targets[valid]
        if rows.numel() == 0:
            return PerturbationInterventionOutput(
                intervention=graph_enhanced_state.new_zeros(
                    batch_size, num_genes, self.node_dim
                )
            )

        structural_state = self.gene_parent(graph_enhanced_state[rows, genes])
        structural_state = structural_state + self.technology_parent(
            technology_features[rows]
        )
        structural_state = structural_state + self.structural_bias
        target_state = torch.tanh(
            self.intervention_output(F.silu(structural_state))
        )
        intervention = torch.zeros(
            batch_size,
            num_genes,
            self.node_dim,
            device=device,
            dtype=target_state.dtype,
        )
        flat_targets = rows * num_genes + genes
        intervention.view(-1, self.node_dim).index_add_(0, flat_targets, target_state)
        return PerturbationInterventionOutput(intervention=intervention)


class LinearGlobalAttention(nn.Module):
    """Apply single-head linear attention across genes."""

    def __init__(self, node_dim: int, attention_dim: int = 32) -> None:
        super().__init__()
        self.query = nn.Linear(node_dim, attention_dim, bias=False)
        self.key = nn.Linear(node_dim, attention_dim, bias=False)
        self.value = nn.Linear(node_dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, node_dim, bias=False)

    def forward(self, node_states: torch.Tensor) -> torch.Tensor:
        output_dtype = node_states.dtype
        # Positive ELU features allow computing K^T V before contracting with Q.
        query = F.elu(self.query(node_states)) + 1.0
        key = F.elu(self.key(node_states)) + 1.0
        value = self.value(node_states)
        # Accumulate over genes in FP32 to avoid overflow in mixed precision.
        with torch.autocast(device_type=node_states.device.type, enabled=False):
            query_f = query.float()
            key_f = key.float()
            value_f = value.float()
            key_value = torch.einsum("bgd,bge->bde", key_f, value_f)
            normalizer = torch.einsum(
                "bgd,bd->bg", query_f, key_f.sum(dim=1)
            ).clamp_min(1e-6)
            attended = torch.einsum(
                "bgd,bde->bge", query_f, key_value
            ) / normalizer.unsqueeze(-1)
        return self.output(attended.to(dtype=output_dtype))


class PerGeneRelationCrossAttention(nn.Module):
    """Let each gene attend to its own physical and genetic relation tokens."""

    def __init__(self, node_dim: int, attention_dim: int = 32) -> None:
        super().__init__()
        self.attention_dim = int(attention_dim)
        self.query = nn.Linear(node_dim, self.attention_dim, bias=False)
        self.key = nn.Linear(node_dim, self.attention_dim, bias=False)
        self.value = nn.Linear(node_dim, self.attention_dim, bias=False)
        self.output = nn.Linear(self.attention_dim, node_dim, bias=False)

    def forward(
        self,
        node_states: torch.Tensor,
        relation_tokens: torch.Tensor,
        relation_mask: torch.Tensor,
    ) -> torch.Tensor:
        if relation_tokens.shape[:3] != (*node_states.shape[:2], 2):
            raise ValueError("Each gene must have exactly two relation tokens, one for physical interactions and one for genetic interactions.")
        if relation_mask.shape != (node_states.shape[1], 2):
            raise ValueError("The relation token mask must have shape (num_genes, 2).")

        query = self.query(node_states).unsqueeze(2)
        key = self.key(relation_tokens)
        value = self.value(relation_tokens)
        scores = (query * key).sum(dim=-1) / math.sqrt(self.attention_dim)

        valid = relation_mask.to(device=scores.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * valid.to(dtype=scores.dtype)
        # Mask both relation tokens for isolated nodes so their local update stays zero.
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = (weights.unsqueeze(-1) * value).sum(dim=2)
        return self.output(context)


class HybridGraphEncoder(nn.Module):
    """Encode local interactions with relation tokens and model global dependencies with linear attention."""

    def __init__(self, table: CandidateEdgeTable, node_dim: int) -> None:
        super().__init__()
        if table.evidence.shape[-1] != 2:
            raise ValueError("The hybrid graph encoder requires two edge evidence types: physical and genetic.")
        self.num_genes = len(table.gene_names)
        self.num_layers = 1
        self.evidence_names = tuple(table.evidence_names)
        self.register_buffer("edge_index", table.edge_index.long(), persistent=False)
        self.register_buffer("edge_evidence", table.evidence.float(), persistent=False)

        self.relation_projections = nn.ModuleList(
            nn.Linear(node_dim, node_dim, bias=False) for _ in range(2)
        )
        self.relation_type_embeddings = nn.Parameter(torch.empty(2, node_dim))
        nn.init.normal_(self.relation_type_embeddings, mean=0.0, std=0.02)
        self.relation_token_norm = nn.LayerNorm(node_dim)
        self.relation_cross_attention = PerGeneRelationCrossAttention(node_dim)
        self.relation_fusion_norm = nn.LayerNorm(node_dim)
        self.global_attention = LinearGlobalAttention(node_dim)
        self.output_norm = nn.LayerNorm(node_dim)

        # Initialize both gates near zero to preserve the background state early in training.
        initial_gate_logit = math.log(0.1 / 0.9)
        self.relation_fusion_gate_logit = nn.Parameter(
            torch.tensor(initial_gate_logit)
        )
        self.global_gate_logit = nn.Parameter(torch.tensor(initial_gate_logit))

        self._adjacency_names = ("physical_adj", "genetic_adj")
        relation_masks = []
        for relation_id, name in enumerate(self._adjacency_names):
            mask = table.evidence[:, relation_id] > 0
            edge = table.edge_index[:, mask].long()
            sender, receiver = edge
            # Average incoming neighbors separately for each relation.
            raw_degree = torch.bincount(
                receiver, minlength=self.num_genes
            ).float()
            degree = raw_degree.clamp_min(1)
            values = degree.reciprocal().index_select(0, receiver)
            adjacency = torch.sparse_coo_tensor(
                torch.stack((receiver, sender)),
                values,
                (self.num_genes, self.num_genes),
            ).coalesce()
            self.register_buffer(name, adjacency, persistent=False)
            relation_masks.append(raw_degree > 0)
        self.register_buffer(
            "relation_token_mask",
            torch.stack(relation_masks, dim=-1),
            persistent=False,
        )

    @staticmethod
    def _aggregate(adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Aggregate one-hop neighbors for the entire batch with a sparse matrix product."""
        batch, genes, feature_dim = features.shape
        output_dtype = features.dtype
        # Use FP32 for CUDA sparse matrix multiplication under mixed precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            packed = features.float().permute(1, 0, 2).reshape(
                genes, batch * feature_dim
            )
            mixed = torch.sparse.mm(adjacency.float(), packed)
        return mixed.reshape(genes, batch, feature_dim).permute(1, 0, 2).to(
            dtype=output_dtype
        )

    def forward(self, gene_background: torch.Tensor) -> torch.Tensor:
        if gene_background.ndim != 3 or gene_background.shape[1] != self.num_genes:
            raise ValueError("The hybrid graph encoder input must have shape (batch, num_genes, node_dim).")

        local_updates = [
            self._aggregate(getattr(self, adjacency_name), projection(gene_background))
            for adjacency_name, projection in zip(
                self._adjacency_names, self.relation_projections
            )
        ]
        relation_tokens = torch.stack(local_updates, dim=2)
        relation_tokens = self.relation_token_norm(
            relation_tokens + self.relation_type_embeddings.view(1, 1, 2, -1)
        )
        relation_update = self.relation_cross_attention(
            gene_background,
            relation_tokens,
            self.relation_token_mask,
        )
        relation_gate = torch.sigmoid(self.relation_fusion_gate_logit).to(
            dtype=gene_background.dtype
        )
        graph_conditioned = self.relation_fusion_norm(
            gene_background + relation_gate * relation_update
        )

        global_update = self.global_attention(graph_conditioned)
        global_gate = torch.sigmoid(self.global_gate_logit).to(
            dtype=gene_background.dtype
        )
        # Add the global update to the original background state.
        return self.output_norm(gene_background + global_gate * global_update)

    def export_relations(self) -> dict[str, torch.Tensor]:
        return {
            "edge_index": self.edge_index,
            "evidence": self.edge_evidence,
            "num_layers": torch.tensor(self.num_layers, device=self.edge_index.device),
        }


class GraphConditionedResponsePropagator(nn.Module):
    """Propagate target interventions using the graph-enhanced states of receiving genes."""

    def __init__(self, *, node_dim: int,
                 receiver_chunk_size: int = 131072) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.receiver_chunk_size = max(1, int(receiver_chunk_size))
        self.receiver_modulation = nn.Linear(node_dim, node_dim, bias=False)
        self.target_interaction_query = nn.Linear(node_dim, node_dim, bias=False)
        self.target_interaction_message = nn.Linear(node_dim, node_dim, bias=False)
        self.target_interaction_update = nn.Sequential(
            nn.Linear(node_dim, node_dim, bias=False), nn.SiLU(),
            nn.Linear(node_dim, node_dim, bias=False),
        )
        self.target_interaction_bias = nn.Parameter(torch.tensor(-1.0))
        self.target_interaction_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _target_groups_from_cpu(target_gene_indices: torch.Tensor, num_genes: int):
        if target_gene_indices.device.type != "cpu":
            raise ValueError("Provide CPU target_groups when target_gene_indices are on the GPU.")
        groups: dict[tuple[int, ...], list[int]] = {}
        for row, raw_targets in enumerate(target_gene_indices.tolist()):
            targets = tuple(sorted({int(g) for g in raw_targets if 0 <= int(g) < num_genes}))
            groups.setdefault(targets, []).append(row)
        return groups

    def _broadcast_intervention(
        self,
        receiver_gain: torch.Tensor,
        intervention: torch.Tensor,
        target: int,
    ) -> torch.Tensor:
        """Broadcast a target intervention with gene-specific modulation."""
        batch_size, num_genes, _ = receiver_gain.shape
        chunks = []
        genes_per_chunk = max(1, self.receiver_chunk_size // max(1, batch_size))
        for start in range(0, num_genes, genes_per_chunk):
            end = min(start + genes_per_chunk, num_genes)
            response = intervention.unsqueeze(1) * receiver_gain[:, start:end]
            # The direct intervention already accounts for the target's own response.
            if start <= target < end:
                indicator = torch.ones(
                    end - start, device=response.device, dtype=response.dtype
                )
                indicator[target - start] = 0.0
                response = response * indicator.view(1, -1, 1)
            chunks.append(response)
        return torch.cat(chunks, dim=1)

    def _interact_target_interventions(
        self, target_interventions: torch.Tensor
    ) -> torch.Tensor:
        """Update each target intervention with symmetric pairwise interactions."""
        num_targets = target_interventions.shape[1]
        if num_targets <= 1:
            return target_interventions
        projected = self.target_interaction_query(target_interventions)
        candidates = self.target_interaction_message(target_interventions)
        received = [[] for _ in range(num_targets)]
        for left in range(num_targets):
            for right in range(left + 1, num_targets):
                score = (projected[:, left] * projected[:, right]).sum(-1) / math.sqrt(self.node_dim)
                gate = torch.sigmoid(score + self.target_interaction_bias)
                received[left].append(gate.unsqueeze(-1) * candidates[:, right])
                received[right].append(gate.unsqueeze(-1) * candidates[:, left])
        messages = torch.stack([torch.stack(x, 0).mean(0) for x in received], dim=1)
        delta_values = self.target_interaction_scale * torch.tanh(
            self.target_interaction_update(messages))
        return target_interventions + delta_values

    def forward(self, graph_enhanced_state: torch.Tensor, initial_intervention: torch.Tensor,
                target_gene_indices: torch.Tensor,
                target_groups: Optional[Sequence[tuple[tuple[int, ...], Sequence[int]]]] = None
                ) -> PerturbationPropagationOutput:
        if graph_enhanced_state.shape != initial_intervention.shape:
            raise ValueError("Graph-enhanced gene states and initial interventions must have the same shape.")
        if target_gene_indices.ndim != 2 or target_gene_indices.shape[0] != graph_enhanced_state.shape[0]:
            raise ValueError("Target indices must have shape (batch, targets).")

        total_response = torch.zeros_like(graph_enhanced_state)
        grouped_rows = (tuple(self._target_groups_from_cpu(
            target_gene_indices, graph_enhanced_state.shape[1]).items())
            if target_groups is None else target_groups)
        for targets, rows in grouped_rows:
            if not targets:
                continue
            row_indices = torch.as_tensor(
                rows, device=graph_enhanced_state.device, dtype=torch.long
            )
            group_background = graph_enhanced_state.index_select(0, row_indices)
            group_intervention = initial_intervention.index_select(0, row_indices)
            receiver_gain = 1.0 + torch.tanh(self.receiver_modulation(group_background))
            target_interventions = torch.stack(
                [group_intervention[:, int(target)] for target in targets], dim=1
            )
            interacted_interventions = self._interact_target_interventions(
                target_interventions
            )
            group_response = group_intervention.clone()
            for pos, target in enumerate(targets):
                propagated = self._broadcast_intervention(
                    receiver_gain, interacted_interventions[:, pos], int(target)
                )
                group_response = group_response + propagated.to(group_response.dtype)
            # Match the destination dtype for index_copy under mixed precision.
            total_response = total_response.index_copy(
                0, row_indices, group_response.to(total_response.dtype)
            )
        return PerturbationPropagationOutput(total_response=total_response)
