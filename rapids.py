# Copied from
# https://github.com/scrna-bench/pipelines-py/blob/main/methods/rapids.py
# https://github.com/billila/pca_scwf_paper/blob/main/wfsc/1.3M/rapids_1.3M.py
# TODO: check attribution, license
# Copyright (c) 2026 Ilaria Billato

from time import time
import sys

import scanpy as sc
import cupy as cp
import rmm
import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
from search_res import binary_search


def run_rapids(
    adata: sc.AnnData,
    n_cluster: int,
    n_comp: int,
    n_neig: int,
    n_hvg: int,
    filter: str,
    timings: dict[str, None | float],
    clustering_info: dict[str, dict[str, None | float | int]],
) -> sc.AnnData:
    start_time = time()
    # push anndata to GPU ####
    rmm.reinitialize(
        managed_memory=False,
        pool_allocator=False,
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    rsc.get.anndata_to_GPU(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["gpu_load"] = time_elapsed

    # find mitocondrial genes ####
    start_time = time()
    rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix="MT-")
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["find_mit_gene"] = time_elapsed

    print("after loading:", adata.shape)

    # filter data ####
    start_time = time()
    if filter == "manual":
        qc = adata.uns["qc_thresholds"]
        min_genes = qc[qc["metric"] == "nFeature"]["min"].values[0]
        max_genes = qc[qc["metric"] == "nFeature"]["max"].values[0]
        max_counts = qc[qc["metric"] == "nCount"]["max"].values[0]
        max_mt = qc[qc["metric"] == "percent.mt"]["max"].values[0]

        rsc.pp.filter_cells(adata, min_genes=min_genes)
        sys.stderr.write("cells after filtering1: " + str(adata.shape) + "\n")
        rsc.pp.filter_cells(adata, max_genes=max_genes)
        sys.stderr.write("cells after filtering2: " + str(adata.shape) + "\n")
        rsc.pp.filter_cells(adata, max_counts=max_counts)
        sys.stderr.write("cells after filtering3: " + str(adata.shape) + "\n")
        rsc.pp.filter_genes(adata, min_cells=3)
        print("after filtering1:", adata.shape)

        # adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
        adata = adata[adata.obs.pct_counts_mt < max_mt, :]

    end_time = time()
    print("after filtering4:", adata.shape)
    sys.stderr.write("cells after filtering4: " + str(adata.shape) + "\n")
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["filter"] = time_elapsed

    # normalization ####
    start_time = time()
    adata.layers["counts"] = adata.X.copy()
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["normalization"] = time_elapsed

    # Identification of highly variable features (feature selection) ####
    start_time = time()
    rsc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="seurat_v3",
        layer="counts",
    )
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]]
    rsc.pp.regress_out(adata, keys=["total_counts", "pct_counts_mt"])
    end_time = time()
    print("'adata.raw' after HVGs:", adata.raw.shape)
    print("'adata' after HVGs:", adata.shape)
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["hvg"] = time_elapsed

    # Scaling the data ####
    start_time = time()
    rsc.pp.scale(adata, max_value=10)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["scaling"] = time_elapsed

    # PCA ####
    start_time = time()
    rsc.pp.pca(adata, n_comps=n_comp)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["pca"] = time_elapsed

    # t-sne ####
    start_time = time()
    rsc.tl.tsne(adata, n_pcs=n_comp)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["t_sne"] = time_elapsed

    # UMAP ####
    start_time = time()
    rsc.pp.neighbors(adata, n_neighbors=n_neig, n_pcs=n_comp)
    rsc.tl.umap(adata)
    end_time = time()
    time_elapsed = end_time - start_time
    print("Time Elapsed:", time_elapsed)
    timings["umap"] = time_elapsed

    # louvain ####
    start_time = time()
    _, res, num_runs = binary_search(adata, n_cluster, rsc.tl.louvain)
    end_time = time()
    time_elapsed = end_time - start_time
    avg_time_elapsed = time_elapsed / num_runs
    print(f"Louvain resolution: {res}")
    print(f"Louvain runs: {num_runs}")
    print("Total Search Time Elapsed:", time_elapsed)
    print("Average Time per Clustering Run:", avg_time_elapsed)
    timings["louvain"] = avg_time_elapsed
    clustering_info["resolutions"]["louvain"] = res
    clustering_info["num_runs"]["louvain"] = num_runs

    # leiden ####
    start_time = time()
    _, res, num_runs = binary_search(adata, n_cluster, rsc.tl.leiden)
    end_time = time()
    time_elapsed = end_time - start_time
    avg_time_elapsed = time_elapsed / num_runs
    print(f"Leiden resolution: {res}")
    print(f"Leiden runs: {num_runs}")
    print("Total Search Time Elapsed:", time_elapsed)
    print("Average Time per Clustering Run:", avg_time_elapsed)
    timings["leiden"] = avg_time_elapsed
    clustering_info["resolutions"]["leiden"] = res
    clustering_info["num_runs"]["leiden"] = num_runs

    rsc.get.anndata_to_CPU(adata, convert_all=True)

    return adata
