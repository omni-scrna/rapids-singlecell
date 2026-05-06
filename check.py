#!/usr/bin/env python3
# test that all runtime dependencies import cleanly
import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
print("OK")
