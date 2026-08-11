# rapids-singlecell

A omnibenchmark module to use rapids-singlecell

## Tests

```
pixi run tests
```

Tests live in `tests/`. The GPU is initialized once per session; individual tests do not reinitialize it.

### Fixtures (`tests/data/`)

| File | Description |
|------|-------------|
| `datasets_normalized_selected.h5` | Normalized, feature-selected expression matrix (input to PCA) |
| `datasets_pcas.tsv` | PCA embedding produced from the above (input to kNN) |
| `datasets_neighbors.h5` | kNN graph produced from the above (input to clustering) |

The three files form a chain: normalized → PCA → kNN → cluster. Adding a new stage fixture, or refreshing one after an output change, means running the previous stage's entrypoint and committing the output to `tests/data/`:

```
pixi run python pca.py --normalized_selected_h5 tests/data/datasets_normalized_selected.h5 \
  --solver covariance-eigh --n_components 50 --random_seed 42 --output_dir tests/data --name datasets
pixi run python knn.py --pcas_tsv tests/data/datasets_pcas.tsv \
  --n_neighbors 15 --flavor rapids --random_seed 42 --output_dir tests/data --name datasets
```
