"""Argument parsers for omnibenchmark rapids-singlecell modules.

Reusable across stage entrypoints (pca.py, and future knn.py, cluster.py, ...).

Conventions:
- All arguments are required. No defaults — callers (omnibenchmark configs)
  must pass everything explicitly so runs are fully reproducible from the
  invocation line.
- argparse.parse_args() rejects unknown flags by default; we rely on that
  for strictness rather than parse_known_args().

Solver token convention
-----------------------
For PCA, ``--solver`` currently accepts a single opaque token: ``rapids``.

This is deliberately *coarser* than cuML's internal knob set
(``svd_solver`` ∈ {auto, full, jacobi}, plus a sparse truncated-SVD branch).
Each benchmark cell should correspond to a single labeled method, not a
free-form combination of sub-knobs. So when we want to compare cuML's
algorithms head-to-head later, we extend ``choices=`` here with new tokens
rather than exposing a sub-solver flag:

    rapids               -> cuML default (auto), zero-centered
    rapids-full          -> svd_solver="full"
    rapids-jacobi        -> svd_solver="jacobi"
    rapids-truncated     -> sparse truncated SVD path (zero_center=False)

The scanpy module uses the same pattern (``arpack`` / ``randomized``).
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
