"""OR-Tools' CP-SAT, loaded so it runs beside TensorFlow: import `cp_model` from here, not from OR-Tools."""

import os
import sys

# TensorFlow (2.20 on) and OR-Tools (9.12 on) ship native libraries whose symbols clash: with TensorFlow loaded first,
# as an hls4ml flow loads it, every CP-SAT solve segfaults. Loaded with RTLD_DEEPBIND, OR-Tools binds its own symbols
# before those already loaded. Only glibc has the flag; elsewhere the libraries load as usual.
if hasattr(os, 'RTLD_DEEPBIND'):
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(_flags | os.RTLD_DEEPBIND)
    try:
        from ortools.sat.python import cp_model
    finally:
        sys.setdlopenflags(_flags)
else:
    from ortools.sat.python import cp_model

__all__ = ['cp_model']
