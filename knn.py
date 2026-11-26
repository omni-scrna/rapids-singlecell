#!/usr/bin/env python3
"""kNN graph module (rapids-singlecell-backed) for omnibenchmark.

Input
-----
File: ``--embedding_tsv`` from any producer of the shared embedding_tsv output id.
Header = cell_id + dimension names (PC* or dim_*), each data row prefixed by cell barcode.

Output
------
File: {output_dir}/{name}_neighbors.h5  (NNG stage output: neighbors_h5)

  /distances/{data,indices,indptr,shape}        CSR sparse, n_cells x n_cells
  /connectivities/{data,indices,indptr,shape}   CSR sparse, n_cells x n_cells
  /cell_ids                                     1D string array of length n_cells

cell_ids is preserved on the output so downstream stages can join on
identity, not on row order. Both sparse matrices share that ordering.

Implementation notes
--------------------
- ``--flavor`` selects the ANN backend / precision. ``rapids`` leaves
  rsc.pp.neighbors at its cuVS default (currently CAGRA, FP32). Add new
  tokens (rapids-ivf-flat, rapids-brute, ...) to the ``--flavor`` choices
  to expose other backends.
- The synthetic ``X = zeros((n_cells, 1))`` is just a stand-in to give
  AnnData a well-formed obs axis; the actual neighbors computation runs
  on ``obsm["X_pca"]`` (use_rep="X_pca").
- ``--connectivities`` sets the edge weights written as /connectivities, on
  the same neighbours: ``native`` (rsc's UMAP weights, today's behaviour),
  ``umap`` (the same, named explicitly) or ``snn``: Seurat's shared-nearest-
  neighbour graph, |N_i & N_j| / |N_i | N_j| over neighbour sets with each cell
  itself included, edges below ``--snn_prune`` (default 1/15, Seurat's
  prune.SNN) removed, diagonal 1. Computed on the host from the neighbours, so
  it is the same graph the scanpy, Seurat and brisc kNN modules write for
  ``snn`` (rsc's own ``method="jaccard"`` is PhenoGraph's variant: kNN edges
  only, averaged, unpruned).
- ``random_seed`` is best-effort: only some ANN backends consult it
  (IVF training, for instance). For a fully deterministic graph, prefer
  brute-force once that token is added.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rapids_singlecell as rsc
import scipy.sparse as sp
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local helpers
from common import cli  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import embedding_to_adata  # noqa: E402
from phases import phase  # noqa: E402
from writers import NeighborGraph, read_embeddings, write_graph  # noqa: E402


def parse_args():
    # common/cli injects the synced contract; method params are hand-rolled.
    p = argparse.ArgumentParser(description="OmniBenchmark kNN module (rapids-singlecell)")
    cli.add_base_args(p)            # --output_dir, --name
    cli.add_stage_args(p, "NNG")    # --embedding_tsv
    p.add_argument("--n_neighbors", type=int, required=True,
                   help="Number of nearest neighbors")
    p.add_argument("--flavor", type=str, required=True,
                   choices=["rapids"],
                   help="kNN flavor token (see module docstring)")
    p.add_argument("--random_seed", type=int, required=True, help="Random seed")
    p.add_argument("--connectivities", type=str, default="native",
                   choices=["native", "umap", "snn"],
                   help="edge weights: native/umap (rsc's UMAP weights) or snn (Seurat's SNN); see module docstring")
    p.add_argument("--snn_prune", type=float, default=1 / 15,
                   help="--connectivities snn: drop edges below this Jaccard index (Seurat's prune.SNN)")
    return p.parse_args()


def snn_connectivities(distances, prune=1 / 15):
    """Seurat's SNN graph from a kNN graph's neighbour sets (its stored entries, each cell
    itself added): the Jaccard index |N_i & N_j| / |N_i | N_j| for every pair sharing a
    neighbour, entries below `prune` dropped. Symmetric, diagonal 1."""
    n = distances.shape[0]
    a = sp.csr_matrix((np.ones(distances.nnz, dtype=np.float32), distances.indices.copy(),
                       distances.indptr.copy()), shape=(n, n))
    a = ((a + sp.identity(n, dtype=np.float32, format="csr")) > 0).astype(np.float32)
    k = np.asarray(a.sum(axis=1)).ravel()
    shared = (a @ a.T).tocoo()
    jac = shared.data / (k[shared.row] + k[shared.col] - shared.data)
    keep = jac >= prune
    return sp.csr_matrix((jac[keep], (shared.row[keep], shared.col[keep])), shape=(n, n))


def run_knn(adata, args):
    """GPU-only neighbors. Pre/post: adata stays on GPU. Mutates in place."""
    rsc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        use_rep="X_pca",
        random_state=args.random_seed,
    )


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "embedding_tsv", "n_neighbors", "flavor", "random_seed", "connectivities"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    setup_gpu()

    with phase("load") as attrs:
        emb = read_embeddings(args.embedding_tsv)
        adata = embedding_to_adata(emb)
        attrs["n_cells"], attrs["n_components"] = emb.matrix.shape
        print(f"  embedding: {emb.matrix.shape}")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_knn(adata, args)
        attrs["n_neighbors"] = args.n_neighbors
        attrs["connectivities"] = args.connectivities

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    if args.connectivities == "snn":
        with phase("snn") as attrs:
            adata.obsp["connectivities"] = snn_connectivities(adata.obsp["distances"], args.snn_prune)
            attrs["snn_prune"] = args.snn_prune

    with phase("write") as attrs:
        out = cli.output_path(args, "neighbors_h5", "_neighbors.h5")
        graph = NeighborGraph(
            distances=adata.obsp["distances"],
            connectivities=adata.obsp["connectivities"],
            row_ids=list(emb.row_ids),
        )
        write_graph(graph, out)
        attrs["distances_nnz"] = int(graph.distances.nnz)
        attrs["connectivities_nnz"] = int(graph.connectivities.nnz)
        attrs["path"] = str(out)
        print(f"  distances nnz:      {attrs['distances_nnz']}")
        print(f"  connectivities nnz: {attrs['connectivities_nnz']}")
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
