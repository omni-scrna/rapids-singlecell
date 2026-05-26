#!/usr/bin/env python3
"""kNN graph module (rapids-singlecell-backed) for omnibenchmark.

Input
-----
File: ``--pcas.tsv`` produced by the pca entrypoint of this module
(or any module that emits the same TSV format: header = PC names,
each data row prefixed by cell barcode).

Output
------
File: {output_dir}/{name}_knn.h5

  /distances/{data,indices,indptr,shape}        CSR sparse, n_cells x n_cells
  /connectivities/{data,indices,indptr,shape}   CSR sparse, n_cells x n_cells
  /cell_ids                                     1D string array of length n_cells

cell_ids is preserved on the output so downstream stages can join on
identity, not on row order. Both sparse matrices share that ordering.

Implementation notes
--------------------
- ``--flavor`` selects the ANN backend / precision. ``rapids`` leaves
  rsc.pp.neighbors at its cuVS default (currently CAGRA, FP32);
  ``rapids-cagra-fp16`` casts the embedding to FP16 before indexing.
  Add new tokens (rapids-ivf-flat, rapids-brute, ...) in src/cli.py to
  expose other backends — see the docstring there.
- The synthetic ``X = zeros((n_cells, 1))`` is just a stand-in to give
  AnnData a well-formed obs axis; the actual neighbors computation runs
  on ``obsm["X_pca"]`` (use_rep="X_pca").
- ``random_seed`` is best-effort: only some ANN backends consult it
  (IVF training, for instance). For a fully deterministic graph, prefer
  brute-force once that token is added.
"""

import sys
from pathlib import Path

import cupy as cp
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_knn_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import embedding_to_adata  # noqa: E402
from phases import phase  # noqa: E402
from writers import NeighborGraph, read_embeddings, write_graph  # noqa: E402


def run_knn(adata, args):
    """GPU-only neighbors. Pre/post: adata stays on GPU. Mutates in place.

    Precision: the rsc default path runs in the dtype of ``obsm["X_pca"]``,
    which arrives as FP32 from the PCA stage. ``rapids-cagra-fp16`` casts
    that representation to FP16 before indexing — CAGRA accepts FP16
    datasets natively, so the graph is built end-to-end in half precision.
    """
    if args.flavor == "rapids-cagra-fp16":
        adata.obsm["X_pca"] = adata.obsm["X_pca"].astype(cp.float16)

    rsc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        use_rep="X_pca",
        random_state=args.random_seed,
    )


def main():
    args = build_knn_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "pcas_tsv", "n_neighbors", "flavor", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(args.output_dir)

    setup_gpu()

    with phase("load") as attrs:
        emb = read_embeddings(args.pcas_tsv)
        adata = embedding_to_adata(emb)
        attrs["n_cells"], attrs["n_components"] = emb.matrix.shape
        print(f"  embedding: {emb.matrix.shape}")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_knn(adata, args)
        attrs["n_neighbors"] = args.n_neighbors

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        out = Path(args.output_dir) / f"{args.name}_knn.h5"
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
