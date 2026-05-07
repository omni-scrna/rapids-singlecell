#!/usr/bin/env python3
"""Embedding-based clustering module (rapids-singlecell-backed) for omnibenchmark.

Input
-----
File: ``--pcas.tsv`` produced by the pca entrypoint (cell-id-indexed PC scores).

Output
------
File: {output_dir}/{name}_clusters.tsv

  Header row:    cluster
  Each data row: cell_barcode <TAB> cluster_label

Implementation notes
--------------------
- ``rapids-kmeans`` uses ``rsc.tl.kmeans`` on obsm["X_pca"]. Requires
  ``--n_clusters``. Labels are integers 0..k-1, deterministic given
  ``--random_seed``.
- ``rapids-hdbscan`` uses ``cuml.cluster.HDBSCAN`` directly on obsm["X_pca"]
  (rsc does not expose HDBSCAN as a tl function). Requires ``--min_samples``
  and ``--min_cluster_size``. Noise points receive label -1 and are preserved
  as-is in the output; downstream metric stages must handle them.
  ``--random_seed`` is accepted for CLI uniformity but ignored (HDBSCAN is
  deterministic).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cuml.cluster
import cupy as cp
import numpy as np
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_cluster_embedding_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import embedding_to_adata  # noqa: E402
from phases import phase  # noqa: E402
from writers import Labels, read_embeddings, write_labels  # noqa: E402


@dataclass
class ClusterEmbeddingOptions:
    method: str
    random_seed: int
    n_clusters: Optional[int] = None
    min_samples: Optional[int] = None
    min_cluster_size: Optional[int] = None


def _build_opts(args) -> ClusterEmbeddingOptions:
    if args.method == "rapids-kmeans" and args.n_clusters is None:
        raise ValueError("--n_clusters is required for rapids-kmeans")
    if args.method == "rapids-hdbscan":
        if args.min_samples is None:
            raise ValueError("--min_samples is required for rapids-hdbscan")
        if args.min_cluster_size is None:
            raise ValueError("--min_cluster_size is required for rapids-hdbscan")
    return ClusterEmbeddingOptions(
        method=args.method,
        random_seed=args.random_seed,
        n_clusters=args.n_clusters,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )


def run_cluster(adata, opts: ClusterEmbeddingOptions):
    """GPU-only clustering on obsm["X_pca"]. Mutates adata.obs["cluster"] in place."""
    if opts.method == "rapids-kmeans":
        rsc.tl.kmeans(
            adata,
            n_clusters=opts.n_clusters,
            use_rep="X_pca",
            random_state=opts.random_seed,
            key_added="cluster",
        )
    elif opts.method == "rapids-hdbscan":
        model = cuml.cluster.HDBSCAN(
            min_samples=opts.min_samples,
            min_cluster_size=opts.min_cluster_size,
        )
        labels = model.fit_predict(adata.obsm["X_pca"])
        adata.obs["cluster"] = cp.asnumpy(labels).astype(str)
    else:
        raise ValueError(f"unknown method: {opts.method!r}")


def main():
    args = build_cluster_embedding_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "pcas_tsv", "method",
              "n_clusters", "min_samples", "min_cluster_size", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    opts = _build_opts(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(args.output_dir)

    setup_gpu()

    with phase("load") as attrs:
        embedding = read_embeddings(args.pcas_tsv)
        adata = embedding_to_adata(embedding)
        attrs["n_cells"], attrs["n_dims"] = adata.obsm["X_pca"].shape
        print(f"  embedding: {adata.n_obs} cells x {adata.obsm['X_pca'].shape[1]} dims")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_cluster(adata, opts)
        attrs["method"] = opts.method

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        labels_arr = adata.obs["cluster"].astype(str).to_numpy()
        out = Path(args.output_dir) / f"{args.name}_clusters.tsv"
        write_labels(
            Labels(values=labels_arr, row_ids=list(embedding.row_ids)),
            out,
        )
        n_noise = int(np.sum(labels_arr == "-1"))
        n_clusters = len(set(labels_arr) - {"-1"})
        attrs["n_clusters"] = n_clusters
        attrs["n_noise"] = n_noise
        attrs["path"] = str(out)
        print(f"  n_clusters: {n_clusters}")
        if n_noise:
            print(f"  n_noise:    {n_noise}")
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
