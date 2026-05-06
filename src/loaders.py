"""Loaders for omnibenchmark rapids-singlecell modules.

The HDF5 schema is the omni-scrna intermediate layout (CSC, genes x cells):

    /matrix/data        (1D)  CSC values
    /matrix/indices     (1D)  CSC row indices (gene index per nonzero)
    /matrix/indptr      (1D)  CSC column pointers (one per cell + 1)
    /matrix/shape       (1D)  [n_genes, n_cells]
    /matrix/genes       (1D)  gene IDs (bytes)
    /matrix/barcodes    (1D)  cell barcodes (bytes)

Note this is *not* 10X-conformant (no /matrix/gene_names for v2, no
/matrix/features for v3), so scanpy.read_10x_h5 cannot ingest it. We read
the bare datasets directly. Transposed to cells x genes (CSR) on the way
out to match AnnData conventions.
"""

import anndata as ad
import h5py
import scipy.sparse as sp


def load_matrix(h5_path):
    with h5py.File(h5_path, "r") as h5:
        g = h5["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])
        gene_ids = g["genes"][:].astype(str)
        cell_ids = g["barcodes"][:].astype(str)

    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()  # cells x genes
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_ids
    adata.var_names = gene_ids
    return adata
