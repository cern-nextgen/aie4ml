"""Cycle costs of AIE designs. `estimate(model)` gives a converted model's interval, first-inference latency and tile
use without compiling it: kernel costs from the part's calibrated cost model (`proxy`), composed along the design's
physical plan (`chain`, `design`). Calibration reads compiled kernels -- their identity (`specialization`) and the
schedule the compiler gave them (`listing`) -- and measures them (`probe`)."""

from .estimate import Estimate, estimate

__all__ = ['Estimate', 'estimate']
