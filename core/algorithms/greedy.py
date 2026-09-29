
"""Greedy algorithms for MathLab AI.

This module contains classical greedy algorithms:
- Activity Selection
- Fractional Knapsack
- Greedy Coin Change
- Huffman Coding
- Job Sequencing
- Interval Scheduling

The functions are independent from the Streamlit interface and are
designed to be deterministic, reusable and easy to test.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Sequence
from numbers import Real
from typing import TypeVar


T = TypeVar("T")


# ============================================================
# VALIDATION HELPERS
# ============================================================


def _validate_sequence(data: Sequence[T], name: str) -> None:
    """Validate a sequence."""
    if isinstance(data, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of structured items.")

    if not isinstance(data, Sequence):
        raise TypeError(f"{name} must be a sequence.")


def _validate_numeric(value: Real, name: str) -> None:
    """Validate a real numeric value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")


def _validate_non_negative_number(value: Real, name: str) -> None:
    """Validate a non-negative real number."""
    _validate_numeric(value, name)

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_number(value: Real, name: str) -> None:
    """Validate a strictly positive real number."""
    _validate_numeric(value, name)

    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_interval(interval: Sequence[Real], name: str = "interval") -> None:
    """Validate an interval represented as (start, finish)."""
    if not isinstance(interval, (list, tuple)):
        raise TypeError(f"{name} must be a tuple or list.")

    if len(interval) != 2:
        raise ValueError(f"{name} must contain exactly two values.")

    start, finish = interval

    _validate_numeric(start, f"{name} start")
    _validate_numeric(finish, f"{name} finish")

    if finish < start:
        raise ValueError(
            f"{name} finish time must be greater than or equal to start time."
        )


# ============================================================
# ACTIVITY SELECTION
# ============================================================


def activity_selection(
    activities: Sequence[tuple[Real, Real]],
) -> list[tuple[Real, Real]]:
    """Select the maximum number of non-overlapping activities.

    Each activity is represented as:

        (start, finish)

    The classical greedy strategy sorts activities by increasing
    finish time and repeatedly selects the next compatible activity.

    Examples
    --------
    >>> activity_selection(
    ...     [(1, 2), (3, 4), (0, 6), (5, 7), (8, 9)]
    ... )
    [(1, 2), (3, 4), (5, 7), (8, 9)]
    """
    _validate_sequence(activities, "activities")

    validated: list[tuple[Real, Real]] = []

    for index, activity in enumerate(activities):
        _validate_interval(activity, f"activities[{index}]")
        validated.append((activity[0], activity[1]))

    if not validated:
        return []

    sorted_activities = sorted(
        validated,
        key=lambda activity: (activity[1], activity[0]),
    )

    selected: list[tuple[Real, Real]] = []
    last_finish: Real | None = None

    for start, finish in sorted_activities:
        if last_finish is None or start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected


# ============================================================
# FRACTIONAL KNAPSACK
# ============================================================


def fractional_knapsack(
    weights: Sequence[Real],
    values: Sequence[Real],
    capacity: Real,
) -> float:
    """Return the maximum value for the Fractional Knapsack problem.

    Unlike 0/1 Knapsack, fractions of items can be selected.

    Items are selected according to decreasing value-to-weight ratio.

    Parameters
    ----------
    weights:
        Positive item weights.
    values:
        Non-negative item values.
    capacity:
        Non-negative maximum capacity.

    Returns
    -------
    float
        Maximum achievable value.

    Examples
    --------
    >>> fractional_knapsack([10, 20, 30], [60, 100, 120], 50)
    240.0
    """
    _validate_sequence(weights, "weights")
    _validate_sequence(values, "values")

    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length.")

    _validate_non_negative_number(capacity, "capacity")

    items: list[tuple[float, float, float]] = []

    for index, (weight, value) in enumerate(zip(weights, values)):
        _validate_positive_number(weight, f"weights[{index}]")
        _validate_non_negative_number(value, f"values[{index}]")

        ratio = float(value) / float(weight)
        items.append((ratio, float(weight), float(value)))

    if capacity == 0 or not items:
        return 0.0

    items.sort(key=lambda item: item[0], reverse=True)

    remaining = float(capacity)
    total_value = 0.0

    for ratio, weight, value in items:
        if remaining <= 0:
            break

        if weight <= remaining:
            total_value += value
            remaining -= weight
        else:
            fraction = remaining / weight
            total_value += value * fraction
            remaining = 0.0

    return total_value


# ============================================================
# GREEDY COIN CHANGE
# ============================================================


def greedy_coin_change(
    coins: Sequence[int],
    amount: int,
) -> list[int]:
    """Return coins selected by the greedy coin-change strategy.

    The largest available coin is always selected first.

    Important:
        This algorithm does NOT guarantee an optimal solution for every
        coin system.

    Examples
    --------
    >>> greedy_coin_change([1, 5, 10, 25], 41)
    [25, 10, 5, 1]
    """
    _validate_sequence(coins, "coins")

    if isinstance(amount, bool) or not isinstance(amount, int):
        raise TypeError("amount must be an integer.")

    if amount < 0:
        raise ValueError("amount must be non-negative.")

    validated_coins: list[int] = []

    for index, coin in enumerate(coins):
        if isinstance(coin, bool) or not isinstance(coin, int):
            raise TypeError(f"coins[{index}] must be an integer.")

        if coin <= 0:
            raise ValueError(f"coins[{index}] must be positive.")

        validated_coins.append(coin)

    if amount == 0:
        return []

    if not validated_coins:
        return []

    sorted_coins = sorted(set(validated_coins), reverse=True)

    result: list[int] = []
    remaining = amount

    for coin in sorted_coins:
        while coin <= remaining:
            result.append(coin)
            remaining -= coin

    if remaining != 0:
        return []

    return result


# ============================================================
# HUFFMAN CODING
# ============================================================


class _HuffmanNode:
    """Internal node used by Huffman coding."""

    __slots__ = ("frequency", "symbol", "left", "right")

    def __init__(
        self,
        frequency: int,
        symbol: T | None = None,
        left: _HuffmanNode | None = None,
        right: _HuffmanNode | None = None,
    ) -> None:
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right


def _build_huffman_codes_from_frequencies(
    frequencies: dict[T, int],
) -> dict[T, str]:
    """Build Huffman codes from a frequency mapping."""
    if not frequencies:
        return {}

    if len(frequencies) == 1:
        symbol = next(iter(frequencies))
        return {symbol: "0"}

    heap: list[tuple[int, int, _HuffmanNode]] = []
    counter = 0

    for symbol, frequency in frequencies.items():
        node = _HuffmanNode(frequency=frequency, symbol=symbol)
        heapq.heappush(heap, (frequency, counter, node))
        counter += 1

    while len(heap) > 1:
        frequency_a, _, node_a = heapq.heappop(heap)
        frequency_b, _, node_b = heapq.heappop(heap)

        merged = _HuffmanNode(
            frequency=frequency_a + frequency_b,
            left=node_a,
            right=node_b,
        )

        heapq.heappush(heap, (merged.frequency, counter, merged))
        counter += 1

    root = heap[0][2]

    codes: dict[T, str] = {}

    def visit(node: _HuffmanNode, prefix: str) -> None:
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
            return

        if node.left is not None:
            visit(node.left, prefix + "0")

        if node.right is not None:
            visit(node.right, prefix + "1")

    visit(root, "")

    return codes


def huffman_coding(data: Sequence[T] | str) -> dict[T, str]:
    """Return Huffman codes for the symbols in data.

    The returned dictionary maps each symbol to its binary Huffman code.

    Examples
    --------
    >>> codes = huffman_coding("aaabbc")
    >>> set(codes) == {"a", "b", "c"}
    True
    """
    if isinstance(data, bytes):
        raise TypeError("data must be a sequence or string, not bytes.")

    if not isinstance(data, Sequence):
        raise TypeError("data must be a sequence.")

    if len(data) == 0:
        return {}

    frequencies = dict(Counter(data))

    return _build_huffman_codes_from_frequencies(frequencies)


# ============================================================
# JOB SEQUENCING WITH DEADLINES
# ============================================================


def job_sequencing(
    jobs: Sequence[tuple[T, int, Real]],
) -> tuple[list[T], Real]:
    """Maximize profit by scheduling jobs before their deadlines.

    Each job is represented as:

        (job_id, deadline, profit)

    Every job requires one time slot.

    Returns
    -------
    tuple[list[T], Real]
        A list of selected job IDs in scheduled order and the
        corresponding total profit.

    Examples
    --------
    >>> job_sequencing(
    ...     [("A", 2, 100), ("B", 1, 19), ("C", 2, 27), ("D", 1, 25),
    ...      ("E", 3, 15)]
    ... )
    (['A', 'C', 'E'], 142)
    """
    _validate_sequence(jobs, "jobs")

    validated: list[tuple[T, int, Real]] = []

    for index, job in enumerate(jobs):
        if not isinstance(job, (list, tuple)):
            raise TypeError(f"jobs[{index}] must be a tuple or list.")

        if len(job) != 3:
            raise ValueError(
                f"jobs[{index}] must contain exactly three values."
            )

        job_id, deadline, profit = job

        if isinstance(deadline, bool) or not isinstance(deadline, int):
            raise TypeError(f"jobs[{index}] deadline must be an integer.")

        if deadline <= 0:
            raise ValueError(f"jobs[{index}] deadline must be positive.")

        _validate_numeric(profit, f"jobs[{index}] profit")

        validated.append((job_id, deadline, profit))

    if not validated:
        return [], 0

    max_deadline = max(job[1] for job in validated)

    slots: list[tuple[T, Real] | None] = [None] * (max_deadline + 1)

    sorted_jobs = sorted(
        validated,
        key=lambda job: job[2],
        reverse=True,
    )

    total_profit: Real = 0
    scheduled: list[T] = []

    for job_id, deadline, profit in sorted_jobs:
        latest_slot = min(deadline, max_deadline)

        for slot in range(latest_slot, 0, -1):
            if slots[slot] is None:
                slots[slot] = (job_id, profit)
                total_profit += profit
                break

    for slot in range(1, len(slots)):
        if slots[slot] is not None:
            scheduled.append(slots[slot][0])

    return scheduled, total_profit


# ============================================================
# INTERVAL SCHEDULING
# ============================================================


def interval_scheduling(
    intervals: Sequence[tuple[Real, Real]],
) -> list[tuple[Real, Real]]:
    """Select a maximum-size compatible set of intervals.

    This uses the same earliest-finish-time greedy strategy as
    activity selection.

    Examples
    --------
    >>> interval_scheduling(
    ...     [(1, 3), (2, 5), (4, 7), (6, 9), (8, 10)]
    ... )
    [(1, 3), (4, 7), (8, 10)]
    """
    return activity_selection(intervals)

