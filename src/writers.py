"""Reusable writers for objects produced by this module, conforming to the omni-scrna benchmark spec."""

import csv
from dataclasses import dataclass, field

import numpy as np

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
