"""read_graph must accept the NNG stage output spec, not just our own writer."""

import h5py
import numpy as np

from writers import read_graph


def _write_csr(grp, m):
    grp.create_dataset("data", data=m.data)
    grp.create_dataset("indices", data=m.indices)
    grp.create_dataset("indptr", data=m.indptr)


def test_reads_spec_layout(tmp_path, knn_h5):
    """Spec layout: distances flat at the root, no `shape`, connectivities nested.

    This is what the scanpy module writes (the R metrics reader needs the
    distance CSR at the root); ours nests both and adds `shape`.
    """
    ours = read_graph(knn_h5)

    spec = tmp_path / "spec_neighbors.h5"
    with h5py.File(spec, "w") as h5:
        h5.create_dataset("cell_ids", data=np.array(ours.row_ids, dtype="S"))
        _write_csr(h5, ours.distances)
        _write_csr(h5.create_group("connectivities"), ours.connectivities)

    got = read_graph(spec)
    assert got.row_ids == ours.row_ids
    assert got.distances.shape == ours.distances.shape
    assert (got.distances != ours.distances).nnz == 0
    assert (got.connectivities != ours.connectivities).nnz == 0
