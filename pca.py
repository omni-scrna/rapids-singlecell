from cuml.decomposition import PCA

# PCA stub

adata.obsm["X_pca"] = PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)
