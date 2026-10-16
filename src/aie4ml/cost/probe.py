"""What a compiled, simulated project shows about each kernel, from the compiler's and simulator's own reports:
its tile, cascade role and buffer placement, and the cycles and instructions each call of `run` took."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

_ROLE = re.compile(r'_(single|first|middle|last)I\d')
_STACK = re.compile(r'Core (\d+_\d+)\s+.*?Stack Addr : \[ startAddress = (0x[0-9a-f]+)', re.S)


@dataclass(frozen=True)
class CompiledKernel:
    tile: str  # "col_row"
    graph: str  # the kernel graph's instance name
    role: str  # its cascade role
    block: str  # its block instance in the compiler report


def compiled_kernels(report: dict) -> Iterator[CompiledKernel]:
    """Every kernel of a project's `Work/reports/compiler_report.json`, as the compiler placed and specialised it."""
    mapping = report['mapping']['blockInstanceMapping']
    for block_id, block in report['blockInstances'].items():
        core = mapping.get(block_id, {}).get('coreInfo')
        if core is None:
            continue
        function = report['blockTypes'][block['blockType']]['mangledName']
        role = _ROLE.search(function)
        if role is None:
            raise RuntimeError(f"{block['qualifiedName']} runs {function}, naming no cascade role.")
        tile = f"{core['column']}_{core['row']}"
        yield CompiledKernel(tile, block['qualifiedGraphName'].rsplit('.', 1)[1], role[1], block_id)


def placement(project: Path, report: dict, kernel: CompiledKernel, bank_bytes: int, tile_bytes: int) -> Dict:
    """Where the compiler put a kernel's buffers and stack, relative to its tile: per port, each buffer's
    [columns, rows, bank] away from the kernel; the stack's bank in the kernel's own memory."""
    column, row = (int(n) for n in kernel.tile.split('_'))
    ports = {}
    for port_id, port in report['portInstances'].items():
        if port.get('blockInstance') != kernel.block:
            continue
        buffers = report['mapping']['portInstanceMapping'][port_id].get('bufferInfo', [])
        ports[port['portName'].rsplit('.', 1)[-1]] = [
            [b['column'] - column, b['row'] - row, b['offset'] // bank_bytes] for b in buffers
        ]
    stacks = dict(_STACK.findall((project / 'Work' / 'reports' / 'report_stack.txt').read_text()))
    return {'ports': ports, 'stack_bank': int(stacks[kernel.tile], 16) % tile_bytes // bank_bytes}


def _instruction_profile(project: Path, tile: str) -> List[str]:
    """The simulator's instruction profile of the program the compiler built for `tile`, found by that program:
    the simulator names its profiles after tile coordinates of its own, which differ by generation."""
    program = Path('Work') / 'aie' / tile / 'Release' / tile
    found = []
    for path in sorted((project / 'aiesimulator_output').glob('profile_instr_*.txt')):
        lines = path.read_text(errors='replace').splitlines()
        simulated = next(
            (lines[i + 2].strip() for i, line in enumerate(lines) if line == 'Program being simulated:'), ''
        )
        if Path(simulated).parts[-len(program.parts) :] == program.parts:
            found.append(lines)
    if len(found) != 1:
        raise RuntimeError(f'{project}: {len(found)} instruction profiles simulate the program of {tile}, not one.')
    return found[0]


def run_profile(project: Path, tile: str) -> Dict[str, int]:
    """Calls of the kernel's `run` on `tile` and the instructions and cycles they took in all, from the simulator's
    instruction profile."""
    lines = _instruction_profile(project, tile)
    start = next(i for i, line in enumerate(lines) if line.startswith('Function detail: run '))
    totals = {}
    for i, line in enumerate(lines[start:], start):
        if line.strip().startswith(('Cycle-count', 'Instruction-count')):
            name, value = line.split(':')
            totals[name.strip()] = int(value.split()[0])
        elif line.strip().startswith('-----'):  # the bundle table: its fourth column counts executions
            executions = [match.span() for match in re.finditer(r'-+', line)][3]
            calls = int(lines[i + 1][slice(*executions)])  # of run's first bundle
            return {'calls': calls, 'instructions': totals['Instruction-count'], 'cycles': totals['Cycle-count']}
    raise RuntimeError(f'{project}: the instruction profile of {tile} holds no bundles of run.')
