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

from .cpsat import cp_model

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


def to_beat(values: Mapping[str, int], order: Sequence[Tuple[str, bool]], best: tuple) -> Tuple[tuple, bool]:
    """What the placed criteria (`PLACED`) of a design whose other criteria are `values` must rank, by the placed
    part of `order`, for the design to rank better than `best`: the leading placed values of `best`, below which the
    design ranks better, and whether equalling them is not enough (`strict`) -- as at the first criterion placement
    does not decide where the design differs from `best`: worse there, it must rank strictly below; better, equal
    suffices; equal everywhere, strictly below again."""
    placed = []
    for (name, stepped), value in zip(order, best):
        if name in PLACED:
            placed.append(value)
            continue
        own = step(values[name]) if stepped else values[name]
        if own != value:
            return tuple(placed), own > value
    return tuple(placed), True


def require(
    model: cp_model.CpModel,
    criteria: Mapping[str, Tuple[Any, int]],
    order: Sequence[Tuple[str, bool]],
    beat: Tuple[tuple, bool],
) -> None:
    """Keep in `model` the solutions whose `criteria`, ranked by `order`, rank below `beat` (`to_beat`): its values on
    the first criteria of `order`, which a solution betters at the first it differs in, or, unless `strict`, equals."""
    values, strict = beat
    ties, wins = [], []
    for (name, stepped), value in zip(order, values):
        expression = criteria[name][0]
        low, high = (_least(value), _least(value + 1) - 1) if stepped else (value, value)
        below, tie, win = model.NewBoolVar(''), model.NewBoolVar(''), model.NewBoolVar('')
        model.Add(expression <= low - 1).OnlyEnforceIf(below)
        model.Add(expression >= low).OnlyEnforceIf(tie)
        model.Add(expression <= high).OnlyEnforceIf(tie)
        model.AddBoolAnd([*ties, below]).OnlyEnforceIf(win)  # equal before it, below at it
        wins.append(win)
        ties.append(tie)
    if not strict:
        equal = model.NewBoolVar('')
        model.AddBoolAnd(ties).OnlyEnforceIf(equal)
        wins.append(equal)
    model.AddBoolOr(wins)


def _least(index: int) -> int:
    """The least cycles whose `step` is `index` or more."""
    cycles = max(0, math.ceil(math.expm1(index * math.log1p(INTERVAL_TOLERANCE))))
    while cycles > 0 and step(cycles - 1) >= index:
        cycles -= 1
    while step(cycles) < index:
        cycles += 1
    return cycles


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
    solution holds the value of each of `decisions` and of each criterion. `hint` (by variable index, which may name
    some of the decisions) hints the first solve, the whole solution before each later one, which its bounds keep. A
    stepped criterion is solved alone, then bounded to the end of its best value's step; the others several at once,
    weighted so they still rank in turn, as many as MAX_WEIGHT keeps apart. The bounds stay in `model`, so a later
    call ranks among its best solutions. `floors` holds a value no solution goes below for some criteria: a stage
    whose criteria all have one ends, proven, at the first solution reaching them, which spares the solver proving
    it, and takes no solve where the solution before reaches them.

    None where no solution was found: the model is infeasible where the second is True, else the first solve ran out
    of time. A later solve that runs out of time without a solution leaves the one before it, not proven."""
    criteria, floors = dict(criteria), floors or {}
    solution, assignment, proven = None, [], True  # assignment: every variable's value in the solution
    for stage, stepped in _stages(order, criteria) if order else [((), False)]:
        objective, weight, floor, value = 0, 1, 0, 0
        for name in reversed(stage):
            objective += weight * criteria[name][0]
            floor = None if floor is None or name not in floors else floor + weight * floors[name]
            value += 0 if solution is None else weight * solution.criteria[name]
            weight *= criteria[name][1] + 1
        # a solution before that reaches the floor is the stage's best: no solve
        if not (stage and solution is not None and floor is not None and value <= floor):
            model.Minimize(objective)
            model.ClearHints()
            if solution is None:
                for variable in decisions:
                    if hint is not None and variable.Index() in hint:
                        model.AddHint(variable, hint[variable.Index()])
            else:  # the whole solution before, which the bounds since keep: a complete hint, the solve's first solution
                model.Proto().solution_hint.vars.extend(range(len(assignment)))
                model.Proto().solution_hint.values.extend(assignment)
            reached = _Reached(floor) if stage and floor is not None else None
            status = solver.Solve(model, reached)
            if solution is not None and (
                status not in (cp_model.OPTIMAL, cp_model.FEASIBLE) or solver.ObjectiveValue() > value
            ):
                # Presolve drops solutions some other one dominates -- the hint, where it takes an alternative another
                # beats on this stage's objective -- and the stage then found none as good in its time. Solved again
                # keeping every feasible solution, a weaker presolve the hint survives as its first solution.
                solver.parameters.keep_all_feasible_solutions_in_presolve = True
                reached = _Reached(floor) if stage and floor is not None else None
                status = solver.Solve(model, reached)
                solver.parameters.keep_all_feasible_solutions_in_presolve = False
            if status == cp_model.INFEASIBLE:
                if solution is not None:  # each stage keeps the solution before it: a solver fault, not the model's
                    raise RuntimeError('solve_ranked: a later stage refused the solution of the stage before it.')
                return None, True
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return solution, False
            proven = proven and (status == cp_model.OPTIMAL or (reached is not None and reached.reached))
            assignment = list(solver.ResponseProto().solution)
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
