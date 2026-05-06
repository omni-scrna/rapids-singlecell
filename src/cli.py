"""Argument parsers for omnibenchmark rapids-singlecell modules.

Reusable across stage entrypoints (pca.py, and future knn.py, cluster.py, ...).

Conventions:
- All arguments are required. No defaults — callers (omnibenchmark configs)
  must pass everything explicitly so runs are fully reproducible from the
  invocation line.
- argparse.parse_args() rejects unknown flags by default; we rely on that
  for strictness rather than parse_known_args().

Token conventions
-----------------
Both ``--solver`` (PCA) and ``--flavor`` (kNN) currently accept a single
opaque token: ``rapids``. This is deliberately *coarser* than cuML's
internal knob sets. Each benchmark cell should correspond to a single
labeled method, not a free-form combination of sub-knobs. To compare
cuML algorithms head-to-head later, extend ``choices=`` here with new
tokens rather than exposing sub-knob flags.

PCA solver tokens (cuML's ``svd_solver`` axis):
    rapids               -> cuML default (auto), zero-centered
    rapids-full          -> svd_solver="full"
    rapids-jacobi        -> svd_solver="jacobi"
    rapids-truncated     -> sparse truncated SVD path (zero_center=False)

kNN flavor tokens (ANN search backend axis):
    rapids               -> rsc.pp.neighbors default (currently CAGRA)
    rapids-cagra         -> CAGRA explicitly
    rapids-ivf-flat      -> IVF-Flat
    rapids-ivf-pq        -> IVF-PQ (lossy, faster on very large data)
    rapids-brute         -> brute-force (reference, slow)

The scanpy module uses the same pattern (``arpack`` / ``randomized`` for
PCA; ``umap`` / ``gauss`` for kNN).
"""

import argparse


def add_common_args(parser):
    """Args required by omnibenchmark for every module."""
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--name", type=str, required=True,
                        help="Module name/identifier")


def build_pca_parser():
    parser = argparse.ArgumentParser(description="OmniBenchmark PCA module (rapids-singlecell)")
    add_common_args(parser)

    parser.add_argument("--normalized_selected.h5", dest="input_h5",
                        type=str, required=True,
                        help="TENx-format HDF5 of normalized, selected expression (genes x cells)")

    parser.add_argument("--solver", type=str, required=True,
                        choices=["rapids"],
                        help="PCA solver token (see module docstring for the rapids-* extension scheme)")
    parser.add_argument("--n_components", type=int, required=True,
                        help="Number of principal components to compute")
    parser.add_argument("--random_seed", type=int, required=True,
                        help="Seed for reproducibility")

    return parser


def build_knn_parser():
    parser = argparse.ArgumentParser(description="OmniBenchmark kNN module (rapids-singlecell)")
    add_common_args(parser)

    parser.add_argument("--pcas.tsv", dest="pcas_tsv", type=str, required=True,
                        help="PCA TSV produced by the pca entrypoint (cell-id-indexed PC scores)")
    parser.add_argument("--n_neighbors", type=int, required=True,
                        help="Number of nearest neighbors")
    parser.add_argument("--flavor", type=str, required=True,
                        choices=["rapids"],
                        help="kNN flavor token (see module docstring for the rapids-* extension scheme)")
    parser.add_argument("--random_seed", type=int, required=True,
                        help="Random seed")

    return parser
