"""Sample records, perturbation grouping, and batch preparation for GPM."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator, Optional, Sequence, Union
import numpy as np
import torch

from GPM.conditions import normalize_tech_name
from Tools.Loss import perturbation_group_size


@dataclass
class GPMSample:
    """Store one sample and its references into the shared expression cache.

    The paired control is the model input. The observed perturbed expression is used for
    supervision.
    """

    expression: Optional[Union[np.ndarray, Sequence[float]]]
    gene_names: Sequence[str]
    pert_target_genes: Sequence[str]
    pert_tech: str
    condition: str
    cell_line: Optional[str] = None
    is_control: bool = False
    target_expression: Optional[Union[np.ndarray, Sequence[float]]] = None
    sample_id: Optional[str] = None
    panel_key: Optional[str] = None
    batch_id: Optional[str] = None
    control_match_level: Optional[str] = None
    expression_store: Optional[Any] = None
    expression_row: Optional[int] = None
    target_expression_row: Optional[int] = None
    precomputed_target_gene_indices: Optional[tuple[int, ...]] = None
    precomputed_pert_tech_id: Optional[int] = None
    precomputed_perturbation_group_key: Optional[tuple[str, str, str, str]] = None
    control_context_mean: Optional[np.ndarray] = None
    control_context_variance: Optional[np.ndarray] = None

    def resolved_panel_key(self) -> str:
        """Return the expression panel key, defaulting to ``__default__``."""
        if self.panel_key is not None and str(self.panel_key).strip():
            return str(self.panel_key).strip()
        return "__default__"

    def resolved_cell_line(self) -> str:
        """Return a case-insensitive cell line key with a sentinel for missing labels."""
        token = str(self.cell_line or "").strip()
        return token.upper() if token else "__UNKNOWN_CELL_LINE__"

    def resolved_batch_id(self) -> str:
        token = str(self.batch_id or "").strip()
        return token.upper() if token else "__UNKNOWN_BATCH__"

    def perturbation_group_key(self) -> tuple[str, str, str, str]:
        """Return the panel, cell line, condition, and technology key for OT grouping."""
        if self.precomputed_perturbation_group_key is not None:
            return self.precomputed_perturbation_group_key
        return (
            self.resolved_panel_key(),
            self.resolved_cell_line(),
            str(self.condition).strip(),
            normalize_tech_name(self.pert_tech),
        )


@dataclass
class GPMBatch:
    """Model tensors and metadata for a CPU or device batch."""

    expression: torch.Tensor  # (B, G), paired control expression used as model input
    target_expression: torch.Tensor  # (B, G), observed perturbed expression used for supervision
    pert_tech_ids: torch.Tensor  # (B,)
    target_gene_indices: torch.Tensor  # (B, K)
    is_control: torch.Tensor  # (B,) bool
    control_context_mean: torch.Tensor  # (B, G), cell-line control mean
    control_context_variance: torch.Tensor  # (B, G), cell-line control variance
    # Precompute target groups on the CPU to avoid device synchronization during forward.
    target_groups: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    gene_names: Sequence[str]
    panel_key: str
    # Half-open row slices for the OT groups in this micro-batch.
    pert_group_slices: list[tuple[int, int]] = field(default_factory=list)
    any_control: bool = False


@dataclass
class GPMInferenceResult:
    """Single-cell inference result.

    ``pred_expression`` is anchored to the model's context mean. ``pred_delta`` uses the
    dataset control mean as the evaluation reference.
    """

    pred_expression: np.ndarray
    pred_delta: np.ndarray
    sample_id: Optional[str]
    forward_output: Optional[Any] = None


def _bucket_samples_by_group_key(
    samples: Sequence[GPMSample],
) -> dict[tuple[str, str, str, str], list[GPMSample]]:
    """Group perturbed samples for supervision."""
    buckets: dict[tuple[str, str, str, str], list[GPMSample]] = {}
    for s in samples:
        if s.is_control:
            continue
        buckets.setdefault(s.perturbation_group_key(), []).append(s)
    return buckets


def count_perturbation_groups(
    samples: Sequence[GPMSample],
    group_size: Optional[int] = None,
) -> int:
    """Count complete perturbation groups of the requested size."""
    size = max(1, int(group_size or perturbation_group_size()))
    total = 0
    for items in _bucket_samples_by_group_key(samples).values():
        total += len(items) // size
    return total


@dataclass(frozen=True)
class PackedTrainingMicroBatch:
    """Store a panel-specific micro-batch and its complete OT group slices."""

    samples: tuple[GPMSample, ...]
    pert_group_slices: tuple[tuple[int, int], ...]


def _perturbation_groups_by_panel(
    samples: Sequence[GPMSample],
    group_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> dict[str, list[list[GPMSample]]]:
    """Shuffle samples within each perturbation group and build complete OT groups."""
    size = max(1, int(group_size))
    buckets = _bucket_samples_by_group_key(samples)
    by_panel: dict[str, list[list[GPMSample]]] = {}
    rng = np.random.default_rng(int(seed))

    for key, items in buckets.items():
        panel = key[0]
        if len(items) < size:
            continue
        # Shuffle before grouping so the excluded tail can vary across epochs.
        shuffled_items = list(items)
        if shuffle:
            rng.shuffle(shuffled_items)
        panel_groups = by_panel.setdefault(panel, [])
        for start in range(0, len(shuffled_items) - size + 1, size):
            panel_groups.append(shuffled_items[start : start + size])

    if shuffle:
        for panel in by_panel:
            rng.shuffle(by_panel[panel])

    return by_panel


def iter_packed_training_micro_batches(
    samples: Sequence[GPMSample],
    *,
    group_size: Optional[int] = None,
    micro_batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[PackedTrainingMicroBatch]:
    """Pack complete perturbation groups into panel-specific micro-batches.

    ``pert_group_slices`` records where each group begins and ends in the batch.
    """
    gs = max(1, int(group_size or perturbation_group_size()))
    micro_bs = max(1, int(micro_batch_size))
    if micro_bs < gs:
        raise ValueError(
            f"micro_batch_size={micro_bs} is smaller than pert_group_size={gs}. "
            "Each micro-batch must accommodate at least one complete OT group."
        )

    by_panel = _perturbation_groups_by_panel(
        samples,
        gs,
        shuffle=shuffle,
        seed=seed,
    )

    packed_batches: list[PackedTrainingMicroBatch] = []
    groups_per_batch = max(1, micro_bs // gs)
    for panel in sorted(by_panel):
        groups = by_panel[panel]
        for start in range(0, len(groups), groups_per_batch):
            selected_groups = groups[start : start + groups_per_batch]
            mb_samples: list[GPMSample] = []
            slices: list[tuple[int, int]] = []
            for group in selected_groups:
                group_start = len(mb_samples)
                mb_samples.extend(group)
                slices.append((group_start, len(mb_samples)))
            packed_batches.append(PackedTrainingMicroBatch(
                samples=tuple(mb_samples),
                pert_group_slices=tuple(slices),
            ))

    if shuffle:
        schedule_rng = np.random.default_rng(int(seed) + 209759)
        schedule_rng.shuffle(packed_batches)
    yield from packed_batches


def count_packed_training_micro_batches(
    samples: Sequence[GPMSample],
    *,
    group_size: Optional[int] = None,
    micro_batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> int:
    """Count packed training micro-batches."""
    return sum(
        1
        for _ in iter_packed_training_micro_batches(
            samples,
            group_size=group_size,
            micro_batch_size=micro_batch_size,
            shuffle=shuffle,
            seed=seed,
        )
    )


def iter_sample_batches(
    samples: Sequence[GPMSample],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterator[list[GPMSample]]:
    """Yield panel-specific micro-batches for inference."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    by_panel: dict[str, list[GPMSample]] = {}
    for s in samples:
        by_panel.setdefault(s.resolved_panel_key(), []).append(s)

    panel_keys = sorted(by_panel.keys())
    if shuffle:
        rng = np.random.default_rng(int(seed))
        for pk in panel_keys:
            rng.shuffle(by_panel[pk])

    for pk in panel_keys:
        items = by_panel[pk]
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]


def materialize_sample_expression(
    samples: Sequence[GPMSample],
    *,
    target: bool,
) -> np.ndarray:
    """Gather expression rows by shared store and restore the original sample order."""
    if not samples:
        return np.empty((0, 0), dtype=np.float32)
    num_genes = len(samples[0].gene_names)
    output = np.empty((len(samples), num_genes), dtype=np.float32)
    stored_groups: dict[int, tuple[Any, list[int], list[int]]] = {}

    for out_idx, sample in enumerate(samples):
        store = sample.expression_store
        row = sample.target_expression_row if target else sample.expression_row
        if store is not None and row is not None:
            key = id(store)
            if key not in stored_groups:
                stored_groups[key] = (store, [], [])
            stored_groups[key][1].append(out_idx)
            stored_groups[key][2].append(int(row))
            continue

        value = sample.target_expression if target else sample.expression
        if value is None:
            raise ValueError(f"Sample {sample.sample_id!r} has neither expression values nor an expression store row index.")
        row_value = np.asarray(value, dtype=np.float32).ravel()
        if row_value.shape[0] != num_genes:
            raise ValueError(f"The expression vector for sample {sample.sample_id!r} must contain {num_genes} genes.")
        output[out_idx] = row_value

    for store, output_indices, row_indices in stored_groups.values():
        matrix = store.read_rows(np.asarray(row_indices, dtype=np.int64))
        output[np.asarray(output_indices, dtype=np.int64)] = matrix
    return output


def _materialize_micro_batch(
    samples: Sequence[GPMSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Read paired control and target rows together from the expression cache."""
    first_store = samples[0].expression_store
    if first_store is not None and all(
        sample.expression_store is first_store
        and sample.expression_row is not None
        and sample.target_expression_row is not None
        for sample in samples
    ):
        batch_size = len(samples)
        rows = np.asarray(
            [int(sample.expression_row) for sample in samples]
            + [int(sample.target_expression_row) for sample in samples],
            dtype=np.int64,
        )
        matrix = first_store.read_rows(rows)
        return matrix[:batch_size], matrix[batch_size:]
    return (
        materialize_sample_expression(samples, target=False),
        materialize_sample_expression(samples, target=True),
    )


class GPMCPUStagingBuffer:
    """Reusable CPU staging buffer that grows to fit incoming batches."""

    def __init__(self, *, pin_memory: bool) -> None:
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.capacity = 0
        self.num_genes = 0
        self.max_targets = 0
        self.expression = torch.empty(0)
        self.target_expression = torch.empty(0)
        self.pert_tech_ids = torch.empty(0, dtype=torch.long)
        self.target_gene_indices = torch.empty((0, 0), dtype=torch.long)
        self.is_control = torch.empty(0, dtype=torch.bool)
        self.control_context_mean = torch.empty(0)
        self.control_context_variance = torch.empty(0)

    def _ensure_capacity(self, batch_size: int, num_genes: int, max_targets: int) -> None:
        needs_resize = (
            batch_size > self.capacity
            or num_genes != self.num_genes
            or max_targets > self.max_targets
        )
        if not needs_resize:
            return
        self.capacity = max(batch_size, self.capacity)
        self.num_genes = int(num_genes)
        self.max_targets = max(max_targets, self.max_targets, 1)
        options = {"device": "cpu", "pin_memory": self.pin_memory}
        self.expression = torch.empty(
            (self.capacity, self.num_genes), dtype=torch.float32, **options
        )
        self.target_expression = torch.empty(
            (self.capacity, self.num_genes), dtype=torch.float32, **options
        )
        self.pert_tech_ids = torch.empty(
            self.capacity, dtype=torch.long, **options
        )
        self.target_gene_indices = torch.empty(
            (self.capacity, self.max_targets), dtype=torch.long, **options
        )
        self.is_control = torch.empty(
            self.capacity, dtype=torch.bool, **options
        )
        self.control_context_mean = torch.empty(
            (self.capacity, self.num_genes), dtype=torch.float32, **options
        )
        self.control_context_variance = torch.empty(
            (self.capacity, self.num_genes), dtype=torch.float32, **options
        )

    def stage(
        self,
        expression: np.ndarray,
        target_expression: np.ndarray,
        tech_ids: Sequence[int],
        target_indices: Sequence[Sequence[int]],
        control_flags: Sequence[bool],
        control_context_mean: np.ndarray,
        control_context_variance: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, num_genes = expression.shape
        if control_context_mean.shape != expression.shape:
            raise ValueError("The per-sample control mean matrix must match the expression batch shape.")
        if control_context_variance.shape != expression.shape:
            raise ValueError("The per-sample control variance matrix must match the expression batch shape.")
        max_targets = max((len(row) for row in target_indices), default=0) or 1
        self._ensure_capacity(batch_size, num_genes, max_targets)
        self.expression[:batch_size].copy_(torch.from_numpy(expression))
        self.target_expression[:batch_size].copy_(torch.from_numpy(target_expression))
        self.pert_tech_ids[:batch_size].copy_(torch.as_tensor(tech_ids, dtype=torch.long))
        self.is_control[:batch_size].copy_(torch.as_tensor(control_flags, dtype=torch.bool))
        self.control_context_mean[:batch_size].copy_(
            torch.from_numpy(control_context_mean)
        )
        self.control_context_variance[:batch_size].copy_(
            torch.from_numpy(control_context_variance)
        )
        target_view = self.target_gene_indices[:batch_size, :max_targets]
        target_view.fill_(-1)
        for row_idx, indices in enumerate(target_indices):
            if indices:
                target_view[row_idx, : len(indices)] = torch.as_tensor(
                    indices, dtype=torch.long
                )
        return (
            self.expression[:batch_size],
            self.target_expression[:batch_size],
            self.pert_tech_ids[:batch_size],
            target_view,
            self.is_control[:batch_size],
            self.control_context_mean[:batch_size],
            self.control_context_variance[:batch_size],
        )


def collate_gpm_samples(
    samples: Sequence[GPMSample],
    *,
    tech_encoder,
    device: Optional[torch.device] = None,
    pert_group_slices: Optional[Sequence[tuple[int, int]]] = None,
    staging_buffer: Optional[GPMCPUStagingBuffer] = None,
) -> GPMBatch:
    """Gather sample expressions and assemble a model batch."""
    if not samples:
        raise ValueError("Collation requires a nonempty sample list.")
    panel_key = samples[0].resolved_panel_key()
    gene_names = samples[0].gene_names
    n_genes = len(gene_names)
    if any(sample.resolved_panel_key() != panel_key for sample in samples[1:]):
        raise ValueError("All samples in a micro-batch must share the same panel_key.")
    expression_np, target_expression_np = _materialize_micro_batch(samples)
    if expression_np.shape != (len(samples), n_genes):
        raise ValueError("The retrieved expression matrix does not match the fixed gene panel.")

    def sample_context_matrix(attribute: str) -> np.ndarray:
        vectors = []
        for sample in samples:
            raw = getattr(sample, attribute)
            if raw is None:
                raise ValueError(f"Sample {sample.sample_id!r} is missing {attribute}.")
            vector = np.asarray(raw, dtype=np.float32).ravel()
            if vector.shape != (n_genes,):
                raise ValueError(f"{attribute} does not match the gene panel.")
            vectors.append(vector)
        return np.ascontiguousarray(np.stack(vectors, axis=0), dtype=np.float32)

    context_mean = sample_context_matrix("control_context_mean")
    context_variance = sample_context_matrix("control_context_variance")

    tech_ids: list[int] = []
    target_indices: list[tuple[int, ...]] = []
    control_flags: list[bool] = []
    for s in samples:
        control_flags.append(bool(s.is_control))
        cached_targets = s.precomputed_target_gene_indices
        if cached_targets is None:
            raise ValueError(f"Sample {s.sample_id!r} is missing precomputed target gene indices.")
        target_indices.append(cached_targets)
        tech_ids.append(
            int(s.precomputed_pert_tech_id)
            if s.precomputed_pert_tech_id is not None
            else (0 if s.is_control else tech_encoder.tech_name_to_id(s.pert_tech))
        )

    grouped_rows: dict[tuple[int, ...], list[int]] = {}
    for row, raw_targets in enumerate(target_indices):
        targets = tuple(sorted({
            int(gene) for gene in raw_targets if 0 <= int(gene) < n_genes
        }))
        grouped_rows.setdefault(targets, []).append(row)
    target_groups = tuple(
        (targets, tuple(rows)) for targets, rows in grouped_rows.items()
    )

    staging = staging_buffer or GPMCPUStagingBuffer(
        pin_memory=device is not None and device.type == "cuda"
    )
    (
        expression,
        target_expression,
        pert_tech_ids,
        target_gene_indices,
        is_control_t,
        control_context_mean_t,
        control_context_variance_t,
    ) = staging.stage(
        expression_np,
        target_expression_np,
        tech_ids,
        target_indices,
        control_flags,
        context_mean,
        context_variance,
    )
    cpu_batch = GPMBatch(
        expression=expression,
        target_expression=target_expression,
        pert_tech_ids=pert_tech_ids,
        target_gene_indices=target_gene_indices,
        is_control=is_control_t,
        control_context_mean=control_context_mean_t,
        control_context_variance=control_context_variance_t,
        target_groups=target_groups,
        gene_names=gene_names,
        panel_key=panel_key,
        pert_group_slices=[(int(start), int(end)) for start, end in pert_group_slices or ()],
        any_control=any(control_flags),
    )
    return move_gpm_batch_to_device(cpu_batch, device) if device is not None else cpu_batch


def move_gpm_batch_to_device(
    batch: GPMBatch,
    device: Optional[torch.device],
    *,
    non_blocking: bool = False,
) -> GPMBatch:
    """Move model tensors to the device while keeping batch metadata on the CPU."""
    if device is None or device.type == "cpu":
        return batch
    return GPMBatch(
        expression=batch.expression.to(device, non_blocking=non_blocking),
        target_expression=batch.target_expression.to(device, non_blocking=non_blocking),
        pert_tech_ids=batch.pert_tech_ids.to(device, non_blocking=non_blocking),
        target_gene_indices=batch.target_gene_indices.to(device, non_blocking=non_blocking),
        is_control=batch.is_control.to(device, non_blocking=non_blocking),
        control_context_mean=batch.control_context_mean.to(
            device, non_blocking=non_blocking
        ),
        control_context_variance=batch.control_context_variance.to(
            device, non_blocking=non_blocking
        ),
        target_groups=batch.target_groups,
        gene_names=batch.gene_names,
        panel_key=batch.panel_key,
        pert_group_slices=batch.pert_group_slices,
        any_control=batch.any_control,
    )


def iter_prefetched_gpm_batches(
    packed_batches: Sequence[PackedTrainingMicroBatch],
    *,
    tech_encoder: Any,
    device: torch.device,
) -> Iterator[tuple[PackedTrainingMicroBatch, GPMBatch]]:
    """Prefetch the next batch into a second CPU staging buffer.

    CUDA events protect pinned buffers until their asynchronous transfers finish.
    """
    if not packed_batches:
        return
    use_cuda = device.type == "cuda"
    slots = [
        GPMCPUStagingBuffer(pin_memory=use_cuda),
        GPMCPUStagingBuffer(pin_memory=use_cuda),
    ]
    transfer_events: list[Optional[torch.cuda.Event]] = [
        torch.cuda.Event() if use_cuda else None,
        torch.cuda.Event() if use_cuda else None,
    ]
    event_recorded = [False, False]

    def load(slot_idx: int, packed: PackedTrainingMicroBatch) -> GPMBatch:
        event = transfer_events[slot_idx]
        if event is not None and event_recorded[slot_idx]:
            event.synchronize()
        return collate_gpm_samples(
            packed.samples,
            tech_encoder=tech_encoder,
            device=None,
            pert_group_slices=packed.pert_group_slices,
            staging_buffer=slots[slot_idx],
        )

    # Initialize the first batch on the main thread before starting background prefetch.
    cpu_batch = load(0, packed_batches[0])
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpm-prefetch") as executor:
        for batch_idx, packed in enumerate(packed_batches):
            slot_idx = batch_idx % 2
            gpu_batch = move_gpm_batch_to_device(
                cpu_batch,
                device,
                non_blocking=use_cuda,
            )
            if use_cuda:
                event = transfer_events[slot_idx]
                assert event is not None
                event.record(torch.cuda.current_stream(device))
                event_recorded[slot_idx] = True
            future: Optional[Future[GPMBatch]] = None
            if batch_idx + 1 < len(packed_batches):
                next_slot = (batch_idx + 1) % 2
                future = executor.submit(load, next_slot, packed_batches[batch_idx + 1])
            yield packed, gpu_batch
            if future is not None:
                cpu_batch = future.result()


def tensor_to_float32_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array in float32 precision."""
    return t.detach().float().cpu().numpy()
