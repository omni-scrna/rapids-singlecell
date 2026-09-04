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


def test_writes_spec_layout(tmp_path, knn_h5):
    """write_graph must EMIT the spec layout, not just read it.

    The reader accepts both, so a nested-distances writer looks fine in a
    round-trip and only fails downstream, in the R metrics reader that reads
    /data directly.
    """
    from writers import write_graph

    out = tmp_path / "out_neighbors.h5"
    write_graph(read_graph(knn_h5), out)

    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"cell_ids", "connectivities",
                                  "data", "indices", "indptr"}
        assert "distances" not in h5          # flat at the root, not a group
        assert "shape" not in h5
        assert "shape" not in h5["connectivities"]

    got = read_graph(out)
    ours = read_graph(knn_h5)
    assert got.row_ids == ours.row_ids
    assert (got.distances != ours.distances).nnz == 0
    assert (got.connectivities != ours.connectivities).nnz == 0
