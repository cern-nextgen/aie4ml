"""Rough cycle estimates the design search ranks designs by: a ranking, not a measure, which a cost model replaces.

A kernel call costs a fixed overhead and the longer of its multiply-accumulates at the core's peak and its loads; a
cascade adds a short hand-over per further stage. A leg costs nothing where the two kernels can share one buffer,
its bytes at the stream rate where a DMA copies it, and in latency twice that through a memory tile, which stores a
buffer before it forwards it.
"""

from __future__ import annotations

from math import prod
from typing import Tuple

from ..op_impls.utils.precision import element_bytes, resolve_exact_storage_dtype
from .transport.model import Connection

CALL_CYCLES = 30  # locks and loop set-up: a 16x16x16 int8 Dense takes 53 cc on AIE-ML for 16 cc of loads
LOAD_BYTES_PER_CYCLE = 32  # a batch-1 Dense reads its 32 KB of weights in about 1060 cc
CASCADE_HOP_CYCLES = 30  # a light conv took 29 cc longer per cascade stage added


def kernel_cycles(inst, device) -> int:
    """One call of the instance's kernels, from its first stage starting to its last finishing. Its parameters
    are taken as spread evenly over its tiles; its multiply-accumulates per tile are its variant's `work`."""
    node, config = inst.node, inst.config
    work = inst.variant.work(node, config)
    parallelism = config.parallelism
    widths = [element_bytes(inst.variant.input_precision(config, item.role)) for item in inst.inputs]
    loads = sum(prod(inst.port_views[item.tensor].tile) * width for item, width in zip(inst.inputs, widths))
    parameter_bytes = sum(
        prod(t.shape) * element_bytes(resolve_exact_storage_dtype(t.precision, namespace=t.name, layer_name=node.name))
        for t in node.inputs
        if t.is_parameter
    )
    loads += parameter_bytes // parallelism.tiles
    cycles = loads / LOAD_BYTES_PER_CYCLE
    if work is not None:
        cycles = max(cycles, work * widths[0] / device.int8_macs_per_cycle)  # the peak halves as operands widen
    return round(cycles) + CALL_CYCLES + (int(parallelism.cas_length) - 1) * CASCADE_HOP_CYCLES


def leg_cycles(leg: Connection, realization: str, shared: bool, execution, device) -> Tuple[int, int]:
    """(latency, interval) a leg adds: one port's bytes at the stream rate -- twice in latency through a memory
    tile, which stores the buffer before it forwards it -- and nothing where the kernels share the buffer."""
    if shared:
        return 0, 0
    endpoint = leg.producer if leg.consumer is None else leg.consumer
    inst = execution.get(endpoint.node.name)
    if leg.consumer is None:
        dtype = inst.variant.output_precision(inst.config)
    else:
        dtype = inst.variant.input_precision(inst.config, inst.input(endpoint.tensor).role)
    nbytes = prod(inst.port_views[endpoint.tensor].tile) * element_bytes(dtype)
    transfer = nbytes * 8 // device.stream_switch_width_bits
    return (2 if realization == 'memtile' else 1) * transfer, transfer
