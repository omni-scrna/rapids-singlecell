#!/usr/bin/env python3
"""
PCA module (rapids-singlecell-backed) for omnibenchmark.

Output
------
File: {output_dir}/{name}_pcas.tsv

Tab-separated, with header row:
  PC1  PC2  ...  PC{n_components}

One row per cell, prefixed by cell barcode (so R's read.table(..., header=TRUE)
auto-promotes column 1 to row.names). Values are float64.

Implementation notes
--------------------
- Genes are always centered/scaled before PCA (rsc.pp.scale, zero_center=True).
  Mirrors the scanpy module's invariant. If alternative scaling is needed
  later, expose it as a new --pca_type variant rather than as an independent
  flag (see src/cli.py for the rapids-* solver-token convention).
- ``--solver`` is a single opaque token ``rapids`` here. cuML's internal
  svd_solver knob (auto/full/jacobi) is left at its default. To compare
  cuML algorithms head-to-head, add new solver tokens (rapids-jacobi,
  rapids-full, rapids-truncated) in src/cli.py — do not expose svd_solver
  as a free-form flag.
- RMM is reinitialized with a non-managed, non-pooled allocator. This makes
  GPU memory accounting predictable for benchmark runs (a pool allocator
  would mask the true working-set cost).
"""

import sys
from pathlib import Path

import numpy as np
import rapids_singlecell as rsc

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_pca_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import load_matrix  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402


def run_pca(adata, args):
    """Run rapids-singlecell PCA on the GPU. Mutates adata; returns embedding (numpy)."""
    rsc.get.anndata_to_GPU(adata)

    rsc.pp.scale(adata, zero_center=True, max_value=None)

    rsc.pp.pca(
        adata,
        n_comps=args.n_components,
        zero_center=True,
        random_state=args.random_seed,
    )

    # Bring obsm/varm/uns back to host so downstream numpy code doesn't trip
    # on cupy arrays.
    rsc.get.anndata_to_CPU(adata, convert_all=True)

    return np.asarray(adata.obsm["X_pca"], dtype=np.float64)


def main():
    args = build_pca_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "input_h5", "solver", "n_components", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    setup_gpu()

    adata = load_matrix(args.input_h5)
    cell_ids = np.array(adata.obs_names)
    print(f"  matrix (cells x genes): {adata.shape}")

    embedding = run_pca(adata, args)
    print(f"  embedding: {embedding.shape}")

    col_names = [f"PC{i + 1}" for i in range(embedding.shape[1])]
    out = Path(args.output_dir) / f"{args.name}_pcas.tsv"
    write_embeddings(Embedding(embedding, list(cell_ids), col_names), out)
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
