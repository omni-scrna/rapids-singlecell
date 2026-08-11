#!/usr/bin/env python3
"""
PCA module (rapids-singlecell-backed) for omnibenchmark.

Output
------
Files: {output_dir}/{name}_pcas.tsv      (PCA stage output: pcas_tsv)
       {output_dir}/{name}_loadings.tsv  (PCA stage output: loadings_tsv)

Tab-separated, float64, one row per cell (pcas) or per gene (loadings). Same
layout as the scanpy module: header ``cell_id``/``gene_id`` then PC1..PCn.

Implementation notes
--------------------
- Genes are centered but not scaled to unit variance: rsc.pp.pca(zero_center=True)
  centers, and there is no rsc.pp.scale step.
- ``--solver`` maps to an rsc svd_solver via SOLVERS below, pinned to the input
  density it is valid for. rsc dispatches on density before the solver and
  substitutes silently, so run_pca raises on a mismatch rather than let a run
  report a solver it did not use.
- RMM is reinitialized with a non-managed, non-pooled allocator. This makes
  GPU memory accounting predictable for benchmark runs (a pool allocator
  would mask the true working-set cost).
"""

import argparse
import sys
from pathlib import Path

import cupyx.scipy.sparse as cusp
import numpy as np
import rapids_singlecell as rsc
import scipy.sparse as sp
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local helpers
from common import cli  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import load_matrix  # noqa: E402
from phases import phase  # noqa: E402
from writers import Embedding, write_embeddings, write_loadings  # noqa: E402


# --solver -> (rsc svd_solver, input density it is valid for). No "auto": the
# solver is always explicit, so a run is identifiable from its invocation line.
SOLVERS = {
    "covariance-eigh":  ("covariance_eigh", "sparse"),
    "lanczos":          ("lanczos",         "sparse"),
    "randomized-halko": ("randomized",      "sparse"),
    "full":             ("full",            "dense"),
    "jacobi":           ("jacobi",          "dense"),
}


def parse_args():
    # common/cli injects the synced contract; method params are hand-rolled.
    p = argparse.ArgumentParser(description="OmniBenchmark PCA module (rapids-singlecell)")
    cli.add_base_args(p)            # --output_dir, --name
    cli.add_stage_args(p, "PCA")    # --normalized_selected_h5
    p.add_argument("--solver", type=str, required=True, choices=sorted(SOLVERS),
                   help="PCA solver (see SOLVERS)")
    p.add_argument("--n_components", type=int, required=True,
                   help="Number of principal components to compute")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Seed for reproducibility")
    return p.parse_args()


def run_pca(adata, args):
    """GPU-only PCA, center-only. Pre/post: adata stays on GPU. Mutates in place."""
    svd_solver, wants = SOLVERS[args.solver]
    density = "sparse" if (cusp.issparse(adata.X) or sp.issparse(adata.X)) else "dense"
    print(f"  X: {type(adata.X).__name__} dtype={adata.X.dtype} "
          f"shape={adata.X.shape} density={density}")
    if density != wants:
        raise SystemExit(f"--solver {args.solver} is {wants}-only but X is {density}; "
                         "rsc would silently run a different solver")
    rsc.pp.pca(
        adata,
        n_comps=args.n_components,
        zero_center=True,
        svd_solver=svd_solver,
        random_state=args.random_seed,
    )


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "normalized_selected_h5", "solver", "n_components", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    setup_gpu()

    with phase("load") as attrs:
        adata = load_matrix(args.normalized_selected_h5)
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

        # Embedding is just a (matrix, row_ids) holder; write_loadings is what
        # stamps the gene_id header.
        loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)
        gene_ids = np.array(adata.var_names)
        loadings_out = Path(args.output_dir) / f"{args.name}_loadings.tsv"
        write_loadings(Embedding(loadings, list(gene_ids), col_names), loadings_out)

        attrs["path"] = str(out)
        attrs["loadings_path"] = str(loadings_out)
        print(f"  embedding: {embedding.shape}, loadings: {loadings.shape}")
        print(f"  wrote: {out}")
        print(f"  wrote: {loadings_out}")


if __name__ == "__main__":
    main()
