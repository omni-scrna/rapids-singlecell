#!/usr/bin/env python3
"""
PCA module using RAPIDS-singlecell.

Solvers:
    rapids-randomized: randomized SVD with configurable
        n_iter (power iterations) and n_oversamples (sampling vectors).
    rapids-exact: covariance matrix eigendecomposition.

PCA internally centers features without unit-variance scaling.
Both PC scores and gene loadings are exported as TSV files.
GPU synchronization is used around PCA computation for timing.
"""


import sys
import argparse
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli
from gpu import setup_gpu  # noqa: E402
from loaders import load_matrix  # noqa: E402
from phases import phase  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402

def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `PCA` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="PCA module (scanpy-backed)")
    cli.add_base_args(p)             # --output_dir, --name
    cli.add_stage_args(p, "PCA")     # --normalized_selected_h5
    p.add_argument("--solver", type=str, required=True,
                   choices=["rapids-randomized", "rapids-full"], help="PCA solver")
    p.add_argument("--dtype", type=str, default="input",
                   choices=["input", "float32", "float64"],
                   help="cast the matrix before PCA (input = leave as read)")
    p.add_argument("--n_components", type=int, required=True,
                   help="Number of principal components to compute")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Seed for randomized solvers (and for reproducibility)")
    p.add_argument("--n_iter", type=int, default=2)
    p.add_argument("--n_oversamples", type=int, default=10)
    return p.parse_args()


def run_pca(adata, args):
    """GPU randomized PCA without additional per-gene scaling."""
    if args.solver == "rapids-randomized":    
        rsc.pp.pca(adata, n_comps=args.n_components, zero_center=True, svd_solver="randomized", random_state=args.random_seed, n_iter=args.n_iter, n_oversamples=args.n_oversamples,)
    elif args.solver == "rapids-exact":
        rsc.pp.pca(adata, n_comps=args.n_components, zero_center=True, svd_solver="covariance_eigh",)
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "normalized_selected_h5", "solver", "n_components", "random_seed", "n_iter", "n_oversamples",):
        print(f"  {k}: {getattr(args, k)}")

    outdir = Path(args.output_dir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Could not create output directory {outdir}") from e

    assert outdir.exists() and outdir.is_dir()
    init_logger(str(outdir))

    setup_gpu()

    with phase("load") as attrs:
        adata = load_matrix(args.normalized_selected_h5)
        cell_ids = np.array(adata.obs_names)
        gene_ids = np.array(adata.var_names)
        attrs["n_cells"], attrs["n_genes"] = adata.shape
        print(f"  matrix (cells x genes): {adata.shape}")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)
    
    cp.cuda.runtime.deviceSynchronize()
    with phase("compute") as attrs:
        run_pca(adata, args)
        cp.cuda.runtime.deviceSynchronize() 

        attrs["n_components"] = args.n_components
        attrs["solver"] = args.solver
        attrs["random_seed"] = args.random_seed

        if args.solver == "rapids-randomized":
            attrs["n_iter"] = args.n_iter
            attrs["n_oversamples"] = args.n_oversamples

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        x_pca = adata.obsm["X_pca"]
        embedding = cp.asnumpy(x_pca).astype(np.float64) if isinstance(x_pca, cp.ndarray) else np.asarray(x_pca, dtype=np.float64)
        col_names = [f"PC{i + 1}" for i in range(embedding.shape[1])]
        out = Path(args.output_dir) / f"{args.name}_embedding.tsv"
        write_embeddings(Embedding(embedding, list(cell_ids), col_names), out)
        attrs["path"] = str(out)
        print(f"  embedding: {embedding.shape}")
        print(f"  wrote: {out}")

    with phase("write_loadings") as attrs:
        loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)
        out_loadings = Path(args.output_dir) / f"{args.name}_loadings.tsv"
        loadings_df = pd.DataFrame(loadings, index=gene_ids, columns=col_names)
        loadings_df.index.name = "gene_id"
        loadings_df.to_csv(out_loadings, sep="\t")
        attrs["path"] = str(out_loadings)
        print(f"  loadings: {loadings.shape}")
        print(f"  wrote: {out_loadings}")

if __name__ == "__main__":
    main()




