from dataclasses import dataclass
from typing import Optional


@dataclass
class ClusterOptions:
    method: str
    resolution: float
    random_seed: int


@dataclass
class ClusterEmbeddingOptions:
    method: str
    random_seed: int
    n_clusters: Optional[int] = None
    min_samples: Optional[int] = None
    min_cluster_size: Optional[int] = None


def build_cluster_embedding_opts(args) -> ClusterEmbeddingOptions:
    if args.method == "rapids-kmeans" and args.n_clusters is None:
        raise ValueError("--n_clusters is required for rapids-kmeans")
    if args.method == "rapids-hdbscan":
        if args.min_samples is None:
            raise ValueError("--min_samples is required for rapids-hdbscan")
        if args.min_cluster_size is None:
            raise ValueError("--min_cluster_size is required for rapids-hdbscan")
    return ClusterEmbeddingOptions(
        method=args.method,
        random_seed=args.random_seed,
        n_clusters=args.n_clusters,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )
