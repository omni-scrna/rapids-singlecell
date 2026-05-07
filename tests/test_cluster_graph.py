import sys

import pytest

from writers import read_graph, read_labels


def _run(monkeypatch, tmp_path, knn_h5, method, resolution=0.5, seed=42):
    monkeypatch.setattr(sys, "argv", [
        "cluster_graph.py",
        "--knn.h5", str(knn_h5),
        "--method", method,
        "--resolution", str(resolution),
        "--random_seed", str(seed),
        "--output_dir", str(tmp_path),
        "--name", "test",
    ])
    from cluster_graph import main
    main()
    return tmp_path / "test_clusters.tsv"


def _check(out, knn_h5):
    assert out.exists()
    labels = read_labels(out)
    graph = read_graph(knn_h5)
    assert labels.col_name == "cluster"
    assert labels.row_ids == graph.row_ids
    assert len(set(labels.values)) > 1


def test_leiden(monkeypatch, tmp_path, knn_h5):
    out = _run(monkeypatch, tmp_path, knn_h5, "rapids-leiden")
    _check(out, knn_h5)


def test_louvain(monkeypatch, tmp_path, knn_h5):
    out = _run(monkeypatch, tmp_path, knn_h5, "rapids-louvain")
    _check(out, knn_h5)


def test_leiden_resolution_affects_n_clusters(monkeypatch, tmp_path, knn_h5):
    out_lo = _run(monkeypatch, tmp_path / "lo", knn_h5, "rapids-leiden", resolution=0.2)
    out_hi = _run(monkeypatch, tmp_path / "hi", knn_h5, "rapids-leiden", resolution=1.5)
    n_lo = len(set(read_labels(out_lo).values))
    n_hi = len(set(read_labels(out_hi).values))
    assert n_hi > n_lo
