from __future__ import annotations

import copy

from ...ir import get_backend_context
from ..base import AIEPass


class LegalizeFanoutEntries(AIEPass):
    def __init__(self):
        self.name = 'legalize_fanout_entries'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        entries = ctx.ir.physical.entries

        changed = False
        rewritten = []

        for entry in entries:
            needs_consumer_split = len(entry.consumers) > 1
            needs_output_split = entry.graph_output and bool(entry.consumers)

            if not needs_consumer_split and not needs_output_split:
                rewritten.append(entry)
                continue

            changed = True

            if needs_output_split:
                output_only = copy.copy(entry)
                output_only.consumers = []
                output_only.graph_output = True
                rewritten.append(output_only)

            for conn in entry.consumers:
                replica = copy.copy(entry)
                replica.consumers = [conn]
                replica.graph_output = False
                if entry.producer.node is None:
                    # Each consumer of a graph input gets its own PLIO ports, sized for it alone.
                    inst = ctx.ir.execution.get(conn.consumer.node.name)
                    replica.producer_port_count = len(
                        conn.consumer.selected_ports(inst.ports.inputs[conn.consumer.tensor].count)
                    )
                rewritten.append(replica)

        ctx.ir.physical.entries = rewritten

        for entry in rewritten:
            if len(entry.consumers) > 1:
                raise RuntimeError(
                    f'{entry.logical_tensor}: unsupported split transport; '
                    f'fanout legalization left {len(entry.consumers)} consumers on one entry.'
                )
            if entry.consumers and entry.graph_output:
                raise RuntimeError(
                    f'{entry.logical_tensor}: unsupported boundary transport mix; '
                    'fanout legalization must separate graph-output legs from consumer legs.'
                )

        return changed
