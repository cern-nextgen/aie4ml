from __future__ import annotations

import pytest
from aie4ml.cost.chain import Link, first_call, interval, timeline
from aie4ml.cost.listing import ScheduleUnavailable, loop_plan

from .test_listing import _listing

WORDS = 8
WRITER = [(1.0, 'write')] * WORDS + [(0.0, 'end')]
READER = [(1.0, 'read')] * WORDS + [(0.0, 'end')]


def test_a_call_is_active_cycles_between_cascade_accesses(tmp_path):
    lines = (
        'MOV.s10 lc, #4',
        '.loop_nesting 1',
        '.begin_of_loop',
        'VMOV MCD, bm0',
        'NOP',
        '.end_of_loop',
        '.loop_nesting 0',
        'MOV.s10 lc, #3',
        '.loop_nesting 1',
        '.begin_of_loop',
        'VLDA wr0, [p3], #32; VLDB wr1, [p3, cs4]',  # two loads of one buffer: they stall
        '.end_of_loop',
        '.loop_nesting 0',
        'RET lr',
    )
    plan = loop_plan(_listing(tmp_path, *lines), 'AIE')
    # The loop without cascade accesses adds its pairs' stalls; the one with them is walked bundle by bundle.
    assert timeline(plan, pair_stall=0.5) == [(2.0, 'write')] * 4 + [(1 + 1 + 3 * 1.5 + 1, 'end')]
    assert timeline(plan, pair_stall=0.0)[-1] == (1 + 1 + 3 + 1, 'end')


def test_the_slowest_kernel_sets_the_interval():
    writer, reader = [(10.0, 'write'), (0.0, 'end')], [(1.0, 'read'), (20.0, 'end')]
    assert interval([writer, reader], [0, 0], Link(7, 4)) == 21
    reader = [(1.0, 'read'), (3.0, 'end')]
    assert interval([writer, reader], [0, 0], Link(7, 4)) == 10
    assert interval([writer, reader], [5, 0], Link(7, 4)) == 15  # the wrapper runs every call
    middle = [(1.0, 'read'), (1.0, 'write')] * WORDS + [(0.0, 'end')]
    assert interval([WRITER, middle, READER], [0, 0, 0], Link(2, 4)) == 2 * WORDS


def test_the_first_call_waits_for_the_chain_to_fill():
    writer, reader = [(10.0, 'write'), (0.0, 'end')], [(1.0, 'read'), (20.0, 'end')]
    # The read waits for the write (10) plus the link (7), then 20 cycles of work follow.
    assert first_call([writer, reader], [0, 0], Link(7, 4)) == 10 + 7 + 20
    assert first_call([writer, reader], [5, 3], Link(7, 4)) == 5 + 10 + 7 + 20
    # A kernel whose input arrives late starts late: the reader starts at 30, after the word is there.
    assert first_call([writer, reader], [0, 0], Link(7, 4), starts=[0, 30]) == 30 + 1 + 20


def test_words_in_flight_bound_the_pace():
    # A write waits for the read `depth` words back, which waits `latency` after its write: (latency + 1) / depth.
    assert interval([WRITER, READER], [0, 0], Link(7, 4)) == (7 + 1) / 4 * WORDS
    assert interval([WRITER, READER], [0, 0], Link(3, 4)) == WORDS


def test_kernels_that_disagree_on_words_are_refused():
    with pytest.raises(ScheduleUnavailable, match='different numbers of cascade words'):
        interval([WRITER, READER[1:]], [0, 0], Link(7, 4))
