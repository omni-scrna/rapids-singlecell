#!/usr/bin/env python3
"""Clustering module (rapids-singlecell-backed) for omnibenchmark.

Input
-----
File: ``--knn.h5`` produced by the knn entrypoint (NeighborGraph layout:
two CSR matrices + cell_ids).

Output
------
File: {output_dir}/{name}_clusters.tsv

  Header row:    cluster
  Each data row: cell_barcode <TAB> cluster_label

Implementation notes
--------------------
- ``--method`` is an opaque token. Currently rapids-leiden and rapids-louvain
  are the only choices; both are graph-based community detection on the
  connectivities matrix. Embedding-based methods (kmeans / hdbscan / dbscan)
  would belong in a separate entrypoint that takes ``--pcas.tsv`` instead.
  See src/cli.py for the token convention.
- The synthetic ``X = zeros((n_cells, 1))`` is just a stand-in to give
  AnnData a well-formed obs axis; the actual computation runs on
  ``obsp["connectivities"]``. ``obsp["distances"]`` is loaded for
  round-trip fidelity but unused by the graph algorithms.
- ``random_seed`` is required for leiden. Louvain in rsc has no seed parameter
  and is non-deterministic; passing ``--random_seed`` for rapids-louvain is
  rejected to avoid the false impression that the run is seed-controlled.
"""

import sys
from pathlib import Path

import numpy as np
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_cluster_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import graph_to_adata  # noqa: E402
from options import ClusterOptions, build_cluster_opts  # noqa: E402
from phases import phase  # noqa: E402
from writers import Labels, read_graph, write_labels  # noqa: E402


def run_cluster(adata, opts: ClusterOptions):
    """GPU-only clustering. Pre/post: adata stays on GPU. Mutates in place.

    Stores cluster labels in ``adata.obs["cluster"]`` regardless of method,
    so downstream extraction is method-agnostic.
    """
    if opts.method == "rapids-leiden":
        rsc.tl.leiden(
            adata,
            resolution=opts.resolution,
            obsp="connectivities",
            random_state=opts.random_seed,
            key_added="cluster",
        )
    elif opts.method == "rapids-louvain":
        rsc.tl.louvain(
            adata,
            resolution=opts.resolution,
            obsp="connectivities",
            key_added="cluster",
        )
    else:
        raise ValueError(f"unknown method: {opts.method!r}")


def main():
    args = build_cluster_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "knn_h5", "method", "resolution", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    opts = build_cluster_opts(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(args.output_dir)

    setup_gpu()

    with phase("load") as attrs:
        graph = read_graph(args.knn_h5)
        adata = graph_to_adata(graph)
        attrs["n_cells"] = adata.n_obs
        attrs["distances_nnz"] = int(graph.distances.nnz)
        attrs["connectivities_nnz"] = int(graph.connectivities.nnz)
        print(f"  graph: {adata.n_obs} cells, "
              f"{graph.connectivities.nnz} connectivity edges")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_cluster(adata, opts)
        attrs["method"] = opts.method
        attrs["resolution"] = opts.resolution

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        labels_arr = adata.obs["cluster"].astype(str).to_numpy()
        out = Path(args.output_dir) / f"{args.name}_clusters.tsv"
        write_labels(
            Labels(values=labels_arr, row_ids=list(graph.row_ids)),
            out,
        )
        n_clusters = len(np.unique(labels_arr))
        attrs["n_clusters"] = int(n_clusters)
        attrs["path"] = str(out)
        print(f"  n_clusters: {n_clusters}")
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
