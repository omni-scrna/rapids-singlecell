#!/usr/bin/env python3
"""
PCA module (rapids-singlecell-backed) for omnibenchmark.

Output
------
Files: {output_dir}/{name}_pcas.tsv      (PCA stage output: pcas_tsv)
       {output_dir}/{name}_loadings.tsv  (PCA stage output: loadings_tsv)

Both are tab-separated with header row PC1 PC2 ... PC{n_components}. pcas has one
row per cell (prefixed by cell barcode); loadings has one row per gene (prefixed
by gene id) — so R's read.table(..., header=TRUE) auto-promotes column 1 to
row.names. Values are float64.

Implementation notes
--------------------
- Genes are centered but NOT scaled to unit variance: centering is done by
  rsc.pp.pca(zero_center=True) itself, so there is no rsc.pp.scale step. If
  unit-variance scaling is needed later, expose it as a new --solver token
  rather than as an independent flag (rapids-* solver-token convention —
  see --solver below).
- ``--solver`` is a single opaque token ``rapids`` here. cuML's internal
  svd_solver knob (auto/full/jacobi) is left at its default. To compare
  cuML algorithms head-to-head, add new solver tokens (rapids-jacobi,
  rapids-full, rapids-truncated) to the ``--solver`` choices — do not expose
  svd_solver as a free-form flag.
- RMM is reinitialized with a non-managed, non-pooled allocator. This makes
  GPU memory accounting predictable for benchmark runs (a pool allocator
  would mask the true working-set cost).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local helpers
from common import cli  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import load_matrix  # noqa: E402
from phases import phase  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402


def parse_args():
    # common/cli injects the shared contract (base args + PCA stage I/O from
    # common/schema); the rapids method params are hand-rolled below. See the
    # module docstring for the rapids-* solver-token extension scheme.
    p = argparse.ArgumentParser(description="OmniBenchmark PCA module (rapids-singlecell)")
    cli.add_base_args(p)            # --output_dir, --name
    cli.add_stage_args(p, "PCA")    # --normalized_selected_h5
    p.add_argument("--solver", type=str, required=True, choices=["rapids"],
                   help="PCA solver token (see module docstring for the rapids-* extension scheme)")
    p.add_argument("--n_components", type=int, required=True,
                   help="Number of principal components to compute")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Seed for reproducibility")
    return p.parse_args()


def run_pca(adata, args):
    """GPU-only PCA. Pre/post: adata stays on GPU. Mutates in place.

    Center-only: zero_center=True centers the genes; no unit-variance scaling.
    """
    # cuML dispatches on the container type/dtype of X (sparse vs dense picks a
    # different solver path), so log what the solver actually got.
    print(f"  X: {type(adata.X).__name__} dtype={adata.X.dtype} shape={adata.X.shape}")
    rsc.pp.pca(
        adata,
        n_comps=args.n_components,
        zero_center=True,
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

        # PCA stage also emits the gene loadings (varm["PCs"]); same TSV layout,
        # rows are genes. Embedding is just a (matrix, row_ids) holder — reuse it.
        loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)
        gene_ids = np.array(adata.var_names)
        loadings_out = Path(args.output_dir) / f"{args.name}_loadings.tsv"
        write_embeddings(Embedding(loadings, list(gene_ids), col_names), loadings_out)

        attrs["path"] = str(out)
        attrs["loadings_path"] = str(loadings_out)
        print(f"  embedding: {embedding.shape}, loadings: {loadings.shape}")
        print(f"  wrote: {out}")
        print(f"  wrote: {loadings_out}")


if __name__ == "__main__":
    main()
