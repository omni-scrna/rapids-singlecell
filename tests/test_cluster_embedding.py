import sys

from writers import read_embeddings, read_labels


def _run(monkeypatch, tmp_path, pcas_tsv, method, extra_args=(), seed=42):
    monkeypatch.setattr(sys, "argv", [
        "cluster_embedding.py",
        "--pcas.tsv", str(pcas_tsv),
        "--method", method,
        "--random_seed", str(seed),
        "--output_dir", str(tmp_path),
        "--name", "test",
        *extra_args,
    ])
    from cluster_embedding import main
    main()
    return tmp_path / "test_clusters.tsv"


def _check(out, pcas_tsv):
    assert out.exists()
    labels = read_labels(out)
    emb = read_embeddings(pcas_tsv)
    assert labels.col_name == "cluster"
    assert labels.row_ids == emb.row_ids


def test_kmeans(monkeypatch, tmp_path, pcas_tsv):
    out = _run(monkeypatch, tmp_path, pcas_tsv, "rapids-kmeans",
               extra_args=("--n_clusters", "10"))
    _check(out, pcas_tsv)
    labels = read_labels(out)
    assert len(set(labels.values)) == 10


def test_hdbscan(monkeypatch, tmp_path, pcas_tsv):
    out = _run(monkeypatch, tmp_path, pcas_tsv, "rapids-hdbscan",
               extra_args=("--min_samples", "5", "--min_cluster_size", "30"))
    _check(out, pcas_tsv)
    labels = read_labels(out)
    assert len(set(labels.values) - {"-1"}) > 1


def test_kmeans_missing_n_clusters(monkeypatch, tmp_path, pcas_tsv):
    import pytest
    with pytest.raises(ValueError, match="--n_clusters"):
        _run(monkeypatch, tmp_path, pcas_tsv, "rapids-kmeans")
