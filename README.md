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
| `datasets_knn.h5` | kNN graph produced from the above (input to clustering) |

The three files form a chain: normalized → PCA → kNN → cluster. Adding a new stage fixture means running the previous stage's entrypoint and committing the output to `tests/data/`.
