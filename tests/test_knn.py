import sys

from writers import read_embeddings, read_graph


def _run(monkeypatch, tmp_path, pcas_tsv, n_neighbors=15, seed=42, extra=()):
    monkeypatch.setattr(sys, "argv", [
        "knn.py",
        "--embedding_tsv", str(pcas_tsv),
        "--n_neighbors", str(n_neighbors),
        "--flavor", "rapids",
        "--random_seed", str(seed),
        "--output_dir", str(tmp_path),
        "--name", "test",
        *extra,
    ])
    from knn import main
    main()
    return tmp_path / "test_neighbors.h5"


def test_knn_graph_shape(monkeypatch, tmp_path, pcas_tsv):
    n_neighbors = 15
    out = _run(monkeypatch, tmp_path, pcas_tsv, n_neighbors=n_neighbors)
    assert out.exists()
    graph = read_graph(out)
    emb = read_embeddings(pcas_tsv)
    n_cells = len(emb.row_ids)
    assert graph.distances.shape == (n_cells, n_cells)
    assert graph.connectivities.shape == (n_cells, n_cells)


def test_knn_distances_nnz(monkeypatch, tmp_path, pcas_tsv):
    n_neighbors = 15
    out = _run(monkeypatch, tmp_path, pcas_tsv, n_neighbors=n_neighbors)
    graph = read_graph(out)
    n_cells = graph.distances.shape[0]
    assert graph.distances.nnz == n_neighbors * n_cells


def test_knn_row_ids_match_embedding(monkeypatch, tmp_path, pcas_tsv):
    out = _run(monkeypatch, tmp_path, pcas_tsv)
    graph = read_graph(out)
    emb = read_embeddings(pcas_tsv)
    assert graph.row_ids == emb.row_ids


def test_snn_connectivities_is_seurats_jaccard():
    """snn: |N_i & N_j| / |N_i | N_j| over neighbour sets with each cell itself added,
    below-prune entries dropped, symmetric, diagonal 1 (Seurat's ComputeSNN)."""
    import numpy as np
    import scipy.sparse as sp
    from knn import snn_connectivities
    rng = np.random.default_rng(0)
    n, k = 40, 6
    rows = np.repeat(np.arange(n), k - 1)
    cols = np.concatenate([rng.choice(np.delete(np.arange(n), i), k - 1, replace=False) for i in range(n)])
    d = sp.csr_matrix((rng.random(len(rows)) + 0.1, (rows, cols)), shape=(n, n))
    sets = [set(d[i].indices) | {i} for i in range(n)]
    want = np.array([[len(a & b) / len(a | b) for b in sets] for a in sets])
    want[want < 1 / 15] = 0
    got = snn_connectivities(d).toarray()
    np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_allclose(got, got.T)
    assert (np.diag(got) == 1).all()


def test_connectivities_snn_changes_weights_not_neighbours(monkeypatch, tmp_path, pcas_tsv):
    import numpy as np
    native = read_graph(_run(monkeypatch, tmp_path / "n", pcas_tsv))
    snn = read_graph(_run(monkeypatch, tmp_path / "s", pcas_tsv, extra=("--connectivities", "snn")))
    np.testing.assert_array_equal(snn.distances.indices, native.distances.indices)
    c = snn.connectivities
    assert abs(c - c.T).max() < 1e-6
    np.testing.assert_allclose(c.diagonal(), 1)
