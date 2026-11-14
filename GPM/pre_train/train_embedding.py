"""Train BioGRID structural gene embeddings from the command line."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pre_train import (  # noqa: E402
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NEGATIVE_OVERSAMPLE,
    DEFAULT_NUM_NEGATIVES,
    DEFAULT_TEMPERATURE,
    PRIOR_EMBED_DIM,
    generate_gene_prior_embeddings,
    load_gene_embedding_pt,
    resolve_biogrid_path,
    resolve_output_path,
    save_gene_embedding_pt,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a relation-aware BioGRID graph encoder for physical and genetic interactions.")
    parser.add_argument("--biogrid", type=Path, default=None, help="Path to BIOGRID.tsv.")
    parser.add_argument("--output", type=Path, default=None, help="Output path for gene_embedding.pt.")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=PRIOR_EMBED_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--positive-batch-size", type=int, default=8_192)
    parser.add_argument("--num-negatives", type=int, default=DEFAULT_NUM_NEGATIVES)
    parser.add_argument("--negative-oversample",type=int,default=DEFAULT_NEGATIVE_OVERSAMPLE,help="Negative sampling candidate multiplier. Increase it for denser graphs.",)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no-amp",action="store_true",help="Disable mixed precision, which is enabled by default on CUDA.",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    import torch
    if str(requested).lower().startswith("cuda") and not torch.cuda.is_available():
        print("[train_embedding] CUDA is unavailable. Using the CPU.")
        return "cpu"
    return str(requested)


def main() -> None:
    args = _parse_args()
    source = resolve_biogrid_path(args.biogrid)
    output = resolve_output_path(args.output, biogrid_path=source)
    device = _resolve_device(args.device)
    print(f"[train_embedding] BioGRID: {source}")
    print(f"[train_embedding] Output: {output}")
    print(f"[train_embedding] Device: {device}")
    started = time.perf_counter()
    payload, _model, history = generate_gene_prior_embeddings(
        biogrid_path=source,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        positive_batch_size=args.positive_batch_size,
        num_negatives=args.num_negatives,
        negative_oversample=args.negative_oversample,
        temperature=args.temperature,
        use_amp=not args.no_amp,
        seed=args.seed,
        print_every=args.print_every,
    )
    save_gene_embedding_pt(payload, output)
    print(
        f"[train_embedding] Saved {payload['n_genes']} x {payload['embedding_dim']} "
        f"embeddings to {output} | final_loss={history.loss[-1]:.6f} | "
        f"physical_loss={history.physical_loss[-1]:.6f} | "
        f"genetic_loss={history.genetic_loss[-1]:.6f} | "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    if args.verify:
        loaded = load_gene_embedding_pt(output)
        matrix = loaded["embeddings"]
        print(
            f"[verify] method={loaded.get('embedding_method')} "
            f"edge_types={loaded.get('edge_types')} "
            f"shape={tuple(matrix.shape)} finite={bool(matrix.isfinite().all())}"
        )


if __name__ == "__main__":
    main()
