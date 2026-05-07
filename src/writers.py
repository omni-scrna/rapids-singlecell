"""Reusable writers for objects produced by this module, conforming to the omni-scrna benchmark spec."""

import csv
from dataclasses import dataclass, field

import h5py
import numpy as np
import scipy.sparse as sp

@dataclass
class Embedding:
    matrix: np.ndarray        # shape (n_cells, n_dims)
    row_ids: list             # cell barcodes, length n_cells
    col_names: list = field(default_factory=list)  # dim labels; auto-generated if empty


def _col_names(embedding):
    if embedding.col_names:
        return embedding.col_names
    return [f"dim_{i + 1}" for i in range(embedding.matrix.shape[1])]


def _row_iter(embedding):
    # Header has N cols (no row-name label); data rows have N+1 cols so that
    # read.table(f, header=TRUE) auto-promotes the first data column to row.names.
    yield _col_names(embedding)
    for cell_id, row in zip(embedding.row_ids, embedding.matrix):
        yield [cell_id] + row.tolist()


def _write_tsv(path, embedding):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(_row_iter(embedding))


def write_embeddings(obj, path, format="tsv"):
    if format == "tsv":
        _write_tsv(path, obj)
    else:
        raise ValueError(f"unsupported format: {format!r}")


def _read_tsv(path):
    rows = []
    row_ids = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        col_names = next(reader)
        for row in reader:
            row_ids.append(row[0])
            rows.append([float(x) for x in row[1:]])
    return Embedding(
        matrix=np.array(rows, dtype=np.float64),
        row_ids=row_ids,
        col_names=col_names,
    )


def read_embeddings(path, format="tsv"):
    """Inverse of write_embeddings. Round-trip-stable for the TSV format."""
    if format == "tsv":
        return _read_tsv(path)
    raise ValueError(f"unsupported format: {format!r}")


@dataclass
class NeighborGraph:
    distances: sp.csr_matrix       # n_cells x n_cells, sparse
    connectivities: sp.csr_matrix  # n_cells x n_cells, sparse
    row_ids: list                  # cell barcodes, length n_cells


def _write_h5_sparse(h5, name, m):
    m = m.tocsr()
    g = h5.create_group(name)
    g.create_dataset("data",    data=m.data)
    g.create_dataset("indices", data=m.indices)
    g.create_dataset("indptr",  data=m.indptr)
    g.create_dataset("shape",   data=np.array(m.shape))


def _write_h5_graph(path, graph):
    with h5py.File(path, "w") as h5:
        _write_h5_sparse(h5, "distances",      graph.distances)
        _write_h5_sparse(h5, "connectivities", graph.connectivities)
        h5.create_dataset(
            "cell_ids",
            data=np.array(graph.row_ids, dtype=h5py.string_dtype()),
        )


def write_graph(obj, path, format="h5"):
    if format == "h5":
        _write_h5_graph(path, obj)
    else:
        raise ValueError(f"unsupported format: {format!r}")


def _read_h5_sparse(g):
    return sp.csr_matrix(
        (g["data"][:], g["indices"][:], g["indptr"][:]),
        shape=tuple(g["shape"][:]),
    )


def read_graph(path, format="h5"):
    """Inverse of write_graph. Round-trip-stable for the HDF5 format."""
    if format != "h5":
        raise ValueError(f"unsupported format: {format!r}")
    with h5py.File(path, "r") as h5:
        return NeighborGraph(
            distances=_read_h5_sparse(h5["distances"]),
            connectivities=_read_h5_sparse(h5["connectivities"]),
            row_ids=list(h5["cell_ids"][:].astype(str)),
        )


@dataclass
class Labels:
    values: np.ndarray   # shape (n_cells,)
    row_ids: list        # length n_cells
    col_name: str = "cluster"


def _write_labels_tsv(path, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow([labels.col_name])
        for cid, val in zip(labels.row_ids, labels.values):
            w.writerow([cid, val])


def write_labels(obj, path, format="tsv"):
    if format == "tsv":
        _write_labels_tsv(path, obj)
    else:
        raise ValueError(f"unsupported format: {format!r}")


def _read_labels_tsv(path):
    row_ids = []
    values = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        col_name = next(reader)[0]
        for row in reader:
            row_ids.append(row[0])
            values.append(row[1])
    return Labels(
        values=np.array(values, dtype=object),
        row_ids=row_ids,
        col_name=col_name,
    )


def read_labels(path, format="tsv"):
    """Inverse of write_labels. Round-trip-stable for the TSV format."""
    if format == "tsv":
        return _read_labels_tsv(path)
    raise ValueError(f"unsupported format: {format!r}")
