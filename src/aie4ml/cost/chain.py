"""A cascade chain's steady interval and first-call end, from each kernel's timeline of active cycles and cascade
accesses: word j is read no sooner than `Link.latency` after its write, and written only once the successor has
read word j - `Link.depth`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .listing import Loop, LoopPlan, ScheduleUnavailable

Timeline = List[Tuple[float, str]]  # (active cycles before it, 'read' | 'write' | 'end')


@dataclass(frozen=True)
class Link:
    """A target's cascade between neighbouring kernels: cycles from a word's write issuing to the earliest read
    of it that does not stall, and the words in flight it holds."""

    latency: int
    depth: int


def timeline(plan: LoopPlan, pair_stall: float) -> Timeline:
    """One call of the kernel `plan` schedules, as active cycles between its cascade accesses."""
    plan.cycles()  # refuses a conditional plan
    events: Timeline = []
    active = 0.0

    def walk(loop: Loop, root: bool) -> None:
        nonlocal active
        if not root and not loop.moves_cascade():
            active += loop.cycles() + pair_stall * loop.pairs()
            return
        for _ in range(loop.trips):
            for item in loop.items:
                if isinstance(item, Loop):
                    walk(item, False)
                elif isinstance(item, int):
                    active += item
                else:
                    tags = item.split('+')
                    active += 1 + pair_stall * ('pair' in tags)
                    for kind in ('read', 'write'):
                        if kind in tags:
                            events.append((active, kind))
                            active = 0.0

    walk(plan.root, True)
    events.append((active, 'end'))
    return events


def interval(timelines: Sequence[Timeline], wrappers: Sequence[float], link: Link, calls: int = 8) -> float:
    """Steady cycles per call of a chain whose kernels, in chain order, run `timelines`, each call preceded by its
    wrapper's `wrappers` cycles."""
    ends = _solve(timelines, wrappers, link, calls)
    half = calls // 2
    return (ends[-1] - ends[half - 1]) / (calls - half)


def first_call(
    timelines: Sequence[Timeline], wrappers: Sequence[float], link: Link, starts: Optional[Sequence[float]] = None
) -> float:
    """When the chain's last kernel ends its first call, fill included, its kernels starting at `starts` (0 when
    not given): when their input buffers are ready."""
    return _solve(timelines, wrappers, link, 1, starts)[0]


def _solve(
    timelines: Sequence[Timeline], wrappers: Sequence[float], link: Link, calls: int, starts=None
) -> List[float]:
    """When the chain's last kernel ends each of `calls` calls."""
    for upstream, downstream in zip(timelines, timelines[1:]):
        if sum(kind == 'write' for _, kind in upstream) != sum(kind == 'read' for _, kind in downstream):
            raise ScheduleUnavailable('neighbouring kernels move different numbers of cascade words per call.')
    tiles = len(timelines)
    events = [
        [(wrappers[t] if i == 0 else 0.0) + active for i, (active, _) in enumerate(timelines[t])] for t in range(tiles)
    ]
    kinds = [[kind for _, kind in timelines[t]] for t in range(tiles)]
    reads: List[List[float]] = [[] for _ in range(tiles)]
    writes: List[List[float]] = [[] for _ in range(tiles)]
    ends: List[List[float]] = [[] for _ in range(tiles)]
    now = [0.0] * tiles if starts is None else [float(s) for s in starts]
    cursor = [0] * tiles  # the next event, counted over all calls
    total = [calls * len(timelines[t]) for t in range(tiles)]
    while any(cursor[t] < total[t] for t in range(tiles)):
        progressed = False
        for t in range(tiles):
            while cursor[t] < total[t]:
                call, i = divmod(cursor[t], len(timelines[t]))
                ready = now[t] + events[t][i]
                if kinds[t][i] == 'read':
                    j = len(reads[t])
                    if t > 0:
                        if j >= len(writes[t - 1]):
                            break  # its word is not written yet
                        ready = max(ready, writes[t - 1][j] + link.latency)
                    reads[t].append(ready)
                elif kinds[t][i] == 'write':
                    j = len(writes[t])
                    if t + 1 < tiles and j >= link.depth:
                        if j - link.depth >= len(reads[t + 1]):
                            break  # the link is full
                        ready = max(ready, reads[t + 1][j - link.depth] + 1)
                    writes[t].append(ready)
                else:
                    ends[t].append(ready)
                now[t] = ready
                cursor[t] += 1
                progressed = True
        if not progressed:
            raise ScheduleUnavailable('the chain deadlocks: its kernels wait on each other.')
    return ends[-1]
