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
- ``--flavor`` is a single opaque token ``rapids``. rsc.pp.neighbors'
  internal ANN backend (CAGRA / IVF-Flat / IVF-PQ / brute_force) is left
  at the cuVS default. Add new tokens (rapids-cagra, rapids-ivf-flat, ...)
  in src/cli.py to expose the choice — see the docstring there.
- The synthetic ``X = zeros((n_cells, 1))`` is just a stand-in to give
  AnnData a well-formed obs axis; the actual neighbors computation runs
  on ``obsm["X_pca"]`` (use_rep="X_pca").
- ``random_seed`` is best-effort: only some ANN backends consult it
  (IVF training, for instance). For a fully deterministic graph, prefer
  brute-force once that token is added.
"""

import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import rapids_singlecell as rsc

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_knn_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from writers import read_embeddings  # noqa: E402


def write_sparse(h5, name, m):
    m = m.tocsr()
    g = h5.create_group(name)
    g.create_dataset("data",    data=m.data)
    g.create_dataset("indices", data=m.indices)
    g.create_dataset("indptr",  data=m.indptr)
    g.create_dataset("shape",   data=np.array(m.shape))


def run_knn(emb, args):
    """Run rapids-singlecell neighbors on the GPU. Returns AnnData with obsp set."""
    adata = ad.AnnData(X=np.zeros((emb.matrix.shape[0], 1), dtype=np.float32))
    adata.obs_names = emb.row_ids
    adata.obsm["X_pca"] = emb.matrix.astype(np.float32)

    rsc.get.anndata_to_GPU(adata)
    rsc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        use_rep="X_pca",
        random_state=args.random_seed,
    )
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    return adata


def main():
    args = build_knn_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "pcas_tsv", "n_neighbors", "flavor", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    setup_gpu()

    emb = read_embeddings(args.pcas_tsv)
    print(f"  embedding: {emb.matrix.shape}")

    adata = run_knn(emb, args)
    print(f"  distances nnz:      {adata.obsp['distances'].nnz}")
    print(f"  connectivities nnz: {adata.obsp['connectivities'].nnz}")

    out = Path(args.output_dir) / f"{args.name}_knn.h5"
    with h5py.File(out, "w") as h5:
        write_sparse(h5, "distances",      adata.obsp["distances"])
        write_sparse(h5, "connectivities", adata.obsp["connectivities"])
        h5.create_dataset(
            "cell_ids",
            data=np.array(emb.row_ids, dtype=h5py.string_dtype()),
        )

    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
