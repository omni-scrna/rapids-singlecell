"""GPU runtime setup shared across rapids-singlecell entrypoints.

Conservative RMM settings (no managed memory, no pool allocator) so that
peak GPU working-set is the *true* working-set, not a pre-warmed pool.
This matters when the GPU memory footprint is itself a benchmark dimension.
"""

import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator


def setup_gpu():
    rmm.reinitialize(managed_memory=False, pool_allocator=False)
    cp.cuda.set_allocator(rmm_cupy_allocator)
