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
- Genes are mean-centered (only) before PCA via rsc.pp.pca(zero_center=True).
  No per-gene variance scaling — mirrors the scanpy/scrapper invariant. If
  alternative scaling is needed later, expose it as a new --pca_type variant
  rather than as an independent flag (see src/cli.py for the rapids-* solver-
  token convention).
- ``--solver`` is an opaque token mapped to cuML's svd_solver knob via
  SOLVER_TO_SVD below: ``rapids`` -> default (auto), ``rapids-exact`` ->
  "full". To compare more cuML algorithms head-to-head, add new tokens
  (rapids-jacobi, rapids-truncated, ...) to SOLVER_TO_SVD and to
  src/cli.py's choices — do not expose svd_solver as a free-form flag.
- RMM is reinitialized with a non-managed, non-pooled allocator. This makes
  GPU memory accounting predictable for benchmark runs (a pool allocator
  would mask the true working-set cost).
"""

import sys
from pathlib import Path

import numpy as np
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_pca_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import load_matrix  # noqa: E402
from phases import phase  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402


SOLVER_TO_SVD = {
    "rapids":       None,    # cuML default (auto)
    "rapids-exact": "full",  # deterministic full SVD on GPU
}


def run_pca(adata, args):
    """GPU-only PCA. Pre/post: adata stays on GPU. Mutates in place."""
    kwargs = {
        "n_comps": args.n_components,
        "zero_center": True,
        "random_state": args.random_seed,
    }
    svd = SOLVER_TO_SVD[args.solver]
    if svd is not None:
        kwargs["svd_solver"] = svd
    rsc.pp.pca(adata, **kwargs)


def main():
    args = build_pca_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "input_h5", "solver", "n_components", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(args.output_dir)

    setup_gpu()

    with phase("load") as attrs:
        adata = load_matrix(args.input_h5)
        cell_ids = np.array(adata.obs_names)
        attrs["n_cells"], attrs["n_genes"] = adata.shape
        print(f"  matrix (cells x genes): {adata.shape}")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_pca(adata, args)
        attrs["n_components"] = args.n_components

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        embedding = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        col_names = [f"PC{i + 1}" for i in range(embedding.shape[1])]
        out = Path(args.output_dir) / f"{args.name}_pcas.tsv"
        write_embeddings(Embedding(embedding, list(cell_ids), col_names), out)
        attrs["path"] = str(out)
        print(f"  embedding: {embedding.shape}")
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
