"""How designs rank (AIEConfig `Optimize`), and a constraint program solved by that ranking: the design search ranks
designs by it, and placement ranks a design's placements by it, so the two agree on what is better.

'throughput' takes the lowest interval, then latency -- each to within INTERVAL_TOLERANCE, which the estimate cannot
tell apart -- then the fewest memory-tile legs, hand-overs copied by a DMA rather than shared, and tiles; 'latency'
the lowest latency, then interval, likewise; 'resource' the fewest tiles, then memory-tile legs and copies, then the
lowest latency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

# What each mode ranks designs by, most significant first: a stepped criterion ranks the values on one
# INTERVAL_TOLERANCE step alike (`step`), which the estimate cannot tell apart. 'copies' are the hand-overs a tile DMA
# copies rather than shares, so of designs alike otherwise the one sharing the most buffers ranks best.
RANKING = {
    'throughput': (
        ('interval', True),
        ('latency', True),
        ('memtile', False),
        ('copies', False),
        ('tiles', False),
        ('interval', False),
        ('latency', False),
    ),
    'latency': (
        ('latency', True),
        ('interval', True),
        ('memtile', False),
        ('copies', False),
        ('tiles', False),
        ('latency', False),
        ('interval', False),
    ),
    # the fewest tiles for the kernels resolution prefers; a fallback kernel only where none of those fits
    'resource': (
        ('fallbacks', False),
        ('tiles', False),
        ('memtile', False),
        ('copies', False),
        ('latency', False),
        ('interval', False),
    ),
}
MODES = tuple(RANKING)
# what of a design its placement decides (`PhysicalIR.estimate`), the rest of a ranking being its kernels' alone
PLACED = ('latency', 'interval', 'copies')
ALIASES = {'performance': 'throughput'}  # the mode's former name
INTERVAL_TOLERANCE = 0.1  # cycle estimates this close the search cannot tell apart
# The largest weight one solve ranks several criteria by at once (`solve_ranked`): beyond it, they take a solve each.
MAX_WEIGHT = 10**9


def optimize_mode(ctx) -> str:
    """The ranking AIEConfig `Optimize` asks for (`RANKING`)."""
    mode = ctx.aie_config.get('Optimize', 'resource')
    mode = ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f'AIEConfig Optimize={mode!r}; expected one of {list(MODES)}.')
    return mode


def step(cycles: int) -> int:
    """`cycles` on a geometric grid INTERVAL_TOLERANCE apart."""
    return int(math.log1p(cycles) / math.log1p(INTERVAL_TOLERANCE))


def rank(values: Mapping[str, int], order: Sequence[Tuple[str, bool]]) -> tuple:
    """`values` ranked by `order` (`RANKING`): the lower, the better."""
    return tuple(step(values[name]) if stepped else values[name] for name, stepped in order)


class Solution(NamedTuple):
    decisions: Dict[int, int]  # each decision's value, by its variable's index
    criteria: Dict[str, int]  # each criterion's value


def solve_ranked(
    model: cp_model.CpModel,
    solver: cp_model.CpSolver,
    criteria: Mapping[str, Tuple[Any, int]],
    order: Sequence[Tuple[str, bool]],
    decisions: Sequence[cp_model.IntVar],
    hint: Optional[Mapping[int, int]] = None,
    floors: Optional[Mapping[str, int]] = None,
) -> Tuple[Optional[Solution], bool]:
    """The best solution of `model` by `order`, criterion by criterion, and whether every solve proved its stage
    optimal; with no `order`, any solution. `criteria` maps each name to its expression and an upper bound; the
    solution holds the value of each of `decisions`, which also hint each solve from the last (the first from `hint`,
    by variable index, which may name some of them), and of each criterion. A stepped criterion is solved alone, then
    bounded to the end of its best value's step; the others several at once, weighted so they still rank in turn, as
    many as MAX_WEIGHT keeps apart. The bounds stay in `model`, so a later call ranks among its best solutions.
    `floors` holds a value no solution goes below for some criteria: a stage whose criteria all have one ends, proven,
    at the first solution reaching them, which spares the solver proving it.

    None where no solution was found: the model is infeasible where the second is True, else the first solve ran out
    of time. A later solve that runs out of time without a solution leaves the one before it, not proven."""
    criteria, floors = dict(criteria), floors or {}
    solution, proven = None, True
    for stage, stepped in _stages(order, criteria) if order else [((), False)]:
        objective, weight, floor = 0, 1, 0
        for name in reversed(stage):
            objective += weight * criteria[name][0]
            floor = None if floor is None or name not in floors else floor + weight * floors[name]
            weight *= criteria[name][1] + 1
        model.Minimize(objective)
        start = solution.decisions if solution is not None else hint
        model.ClearHints()
        for variable in decisions:
            if start is not None and variable.Index() in start:
                model.AddHint(variable, start[variable.Index()])
        reached = _Reached(floor) if stage and floor is not None else None
        status = solver.Solve(model, reached)
        if status == cp_model.INFEASIBLE:
            if solution is not None:  # each stage keeps the solution before it: a solver fault, not the model's
                raise RuntimeError('solve_ranked: a later stage refused the solution of the stage before it.')
            return None, True
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return solution, False
        proven = proven and (status == cp_model.OPTIMAL or (reached is not None and reached.reached))
        solution = Solution(
            {variable.Index(): solver.Value(variable) for variable in decisions},
            {name: solver.Value(expression) for name, (expression, _) in criteria.items()},
        )
        # Each criterion bounded by its value, which keeps exactly the solutions ranking alike on them, as one bound
        # on the weighted sum would, but which the solves after it propagate far better.
        for name in stage:
            expression, _ = criteria[name]
            best = value = solution.criteria[name]
            if stepped:  # every value on the best one's step ranks alike
                while step(value + 1) == step(best):
                    value += 1
            criteria[name] = (expression, value)
            model.Add(expression <= value)
    return solution, proven


class _Reached(cp_model.CpSolverSolutionCallback):
    """Stops a solve at its first solution whose objective reaches `floor`, below which none goes."""

    def __init__(self, floor: int):
        super().__init__()
        self.floor, self.reached = floor, False

    def on_solution_callback(self):
        if self.ObjectiveValue() <= self.floor:
            self.reached = True
            self.StopSearch()


def _stages(order, criteria):
    """The criteria each solve ranks by, taken in turn, so a stage's weights see the bounds the solves before it
    set."""
    group = []
    for name, stepped in order:
        weight = math.prod(criteria[other][1] + 1 for other in (*group, name))
        if group and (stepped or weight > MAX_WEIGHT):
            yield group, False
            group = []
        if stepped:
            yield [name], True
        else:
            group.append(name)
    if group:
        yield group, False
