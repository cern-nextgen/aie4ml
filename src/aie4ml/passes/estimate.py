"""Rough cycle estimates the design search ranks designs by: a ranking, not a measure, which a cost model replaces.

A kernel call costs a fixed overhead and the longer of its multiply-accumulates at the core's peak and its loads; a
cascade adds a short hand-over per further stage. A leg costs nothing where the two kernels can share one buffer,
its bytes at the stream rate where a DMA copies it, and in latency twice that through a memory tile, which stores a
buffer before it forwards it.
"""

from __future__ import annotations

from math import prod
from typing import Optional, Tuple

from ..op_impls.utils.precision import element_bytes, resolve_exact_storage_dtype
from .shared_buffer import port_problem
from .transport.model import Connection

CALL_CYCLES = 30  # locks and loop set-up: a 16x16x16 int8 Dense takes 53 cc on AIE-ML for 16 cc of loads
LOAD_BYTES_PER_CYCLE = 32  # a batch-1 Dense reads its 32 KB of weights in about 1060 cc
CASCADE_HOP_CYCLES = 30  # a light conv took 29 cc longer per cascade stage added


def kernel_cycles(inst, work: Optional[int], device) -> int:
    """One call of the instance's kernels, from its first stage starting to its last finishing. Its parameters
    are taken as spread evenly over its tiles; `work` (multiply-accumulates per tile) is None where the variant
    gives none."""
    node, config = inst.node, inst.config
    footprint = inst.variant.footprint(node, config)
    widths = [element_bytes(inst.variant.input_precision(config, item.role)) for item in inst.inputs]
    loads = sum(prod(inst.port_views[item.tensor].tile) * width for item, width in zip(inst.inputs, widths))
    parameters = [t for t in node.inputs if t.is_parameter]
    loads += sum(
        prod(t.shape) * element_bytes(resolve_exact_storage_dtype(t.precision, namespace=t.name, layer_name=node.name))
        for t in parameters
    ) // (footprint.width * footprint.height)
    cycles = loads / LOAD_BYTES_PER_CYCLE
    if work is not None:
        cycles = max(cycles, work * widths[0] / device.int8_macs_per_cycle)  # the peak halves as operands widen
    return round(cycles) + CALL_CYCLES + (footprint.width - 1) * CASCADE_HOP_CYCLES


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


def shareable(leg: Connection, realization: str, execution, readers: int) -> bool:
    """Whether a direct kernel-to-kernel leg can be one buffer both kernels reach, placed right: one port to one
    port, buffer ports bound to one kernel each, and the value its consumer's alone (`shared_buffer`)."""
    producer, consumer = leg.producer, leg.consumer
    if realization != 'direct' or producer.node is None or consumer is None:
        return False
    if producer.tensor != consumer.tensor or readers != 1 or producer.tensor in execution.graph_outputs:
        return False
    source, sink = execution.get(producer.node.name), execution.get(consumer.node.name)
    written = producer.selected_ports(source.ports.outputs[producer.tensor].count)
    read = consumer.selected_ports(sink.ports.inputs[consumer.tensor].count)
    return len(written) == len(read) and not any(
        port_problem(source, producer.group, port, 'outputs') for port in written
    ) and not any(port_problem(sink, consumer.group, port, 'inputs') for port in read)
