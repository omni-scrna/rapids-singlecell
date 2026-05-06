"""Phase boundary helper around obkit.logger.

TODO: promote to obkit. This context manager is generic and not specific
to rapids-singlecell — every omnibenchmark Python module that emits phase
events would benefit from it. Once upstreamed, this module can be deleted
and callers can ``from obkit.logger import phase`` directly.

Usage:

    init_logger(output_dir)               # do this once, before any phase()
    with phase("load") as attrs:
        adata = load_matrix(...)
        attrs["n_cells"] = adata.n_obs    # captured on end-event
        attrs["n_genes"] = adata.n_vars

The yielded ``attrs`` is a mutable dict scratched by the caller during the
block. We need this deferred-mutation pattern because obkit.emit() only
accepts attrs at call-time, but the metadata you want on the *end* event
is usually only known *after* the work runs (shapes, nnz, output paths,
etc.). So we open the block with an empty dict, let the caller fill it,
and flush whatever's in there on the closing emit(..., "end").

The ``finally`` ensures the end event is emitted even if the block raises,
which is useful because downstream profiler-alignment tooling can detect
truncated phases by absence of an end record. Empty attrs dicts collapse
to no ``attrs`` key in the JSONL record.
"""

from contextlib import contextmanager

from obkit.logger import emit


@contextmanager
def phase(name):
    attrs = {}  # caller mutates during the block; flushed on end
    emit(name, "start")
    try:
        yield attrs
    finally:
        emit(name, "end", attrs=attrs or None)
