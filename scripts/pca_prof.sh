#!/usr/bin/env bash
# denet wrapper for the pca-prof entrypoint.
#
# Omnibenchmark only dispatches an entrypoint via its shebang when the
# entrypoint string is a single file path — anything with spaces gets
# routed through `python3 <string>`. So this script encapsulates the
# whole denet invocation rather than expecting flags in the yaml.
#
# Thread pools are pinned to 1 so host-side BLAS/OpenMP noise doesn't
# drown out the GPU signal in the trace.
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

HERE=$(cd "$(dirname "$0")" && pwd)

# Pull --output_dir out of the forwarded args so denet's trace lands in
# the per-run output dir alongside obkit-events.jsonl. Without this it
# would write to the module's cwd and every concurrent rapids job would
# clobber the same file.
output_dir=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--output_dir" ]; then
    output_dir="$arg"
    break
  fi
  prev="$arg"
done
if [ -z "$output_dir" ]; then
  echo "pca_prof.sh: --output_dir not found in args" >&2
  exit 2
fi

exec denet --json --out "$output_dir/denet_pca.json" run -- python "$HERE/../pca.py" "$@"
