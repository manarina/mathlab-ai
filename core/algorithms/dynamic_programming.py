
"""Dynamic Programming algorithms for MathLab AI.

This module contains classical dynamic programming algorithms:
- Fibonacci
- Climbing Stairs
- 0/1 Knapsack
- Coin Change
- Longest Common Subsequence (LCS)
- Longest Increasing Subsequence (LIS)
- Matrix Chain Multiplication

The functions are designed to be deterministic, testable and independent
from the Streamlit interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import TypeVar


T = TypeVar("T")


# ============================================================
# VALIDATION HELPERS
# ============================================================


def _validate_non_negative_integer(value: int, name: str) -> None:
    """Validate that value is a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate that value is a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_sequence(data: Sequence[T], name: str) -> None:
    """Validate a sequence input."""
    if isinstance(data, (str, bytes)):
        return

    if not isinstance(data, Sequence):
        raise TypeError(f"{name} must be a sequence.")


def _validate_non_negative_integers(
    data: Sequence[int],
    name: str,
) -> None:
    """Validate that all sequence elements are non-negative integers."""
    _validate_sequence(data, name)

    for value in data:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"All elements of {name} must be integers.")

        if value < 0:
            raise ValueError(f"All elements of {name} must be non-negative.")


# ============================================================
# FIBONACCI
# ============================================================


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number.

    Uses the convention:
        F(0) = 0
        F(1) = 1

    Examples
    --------
    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(10)
    55
    """
    _validate_non_negative_integer(n, "n")

    if n <= 1:
        return n

    previous = 0
    current = 1

    for _ in range(2, n + 1):
        previous, current = current, previous + current

    return current


# ============================================================
# CLIMBING STAIRS
# ============================================================


def climbing_stairs(n: int) -> int:
    """Return the number of ways to climb n stairs.

    Each move can climb either 1 or 2 stairs.

    Convention:
        climbing_stairs(0) = 1

    Examples
    --------
    >>> climbing_stairs(2)
    2
    >>> climbing_stairs(5)
    8
    """
    _validate_non_negative_integer(n, "n")

    if n <= 1:
        return 1

    previous = 1
    current = 1

    for _ in range(2, n + 1):
        previous, current = current, previous + current

    return current


# ============================================================
# 0/1 KNAPSACK
# ============================================================


def knapsack_01(
    weights: Sequence[int],
    values: Sequence[Real],
    capacity: int,
) -> Real:
    """Solve the 0/1 Knapsack problem.

    Each item can be selected at most once.

    Parameters
    ----------
    weights:
        Non-negative integer weights.
    values:
        Numeric values corresponding to each item.
    capacity:
        Maximum allowed total weight.

    Returns
    -------
    Real
        Maximum achievable value.

    Examples
    --------
    >>> knapsack_01([1, 2, 3], [6, 10, 12], 5)
    22
    """
    _validate_non_negative_integers(weights, "weights")

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("values must be a sequence.")

    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length.")

    _validate_non_negative_integer(capacity, "capacity")

    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("All elements of values must be numeric.")

    dp: list[Real] = [0] * (capacity + 1)

    for weight, value in zip(weights, values):
        for current_capacity in range(capacity, weight - 1, -1):
            dp[current_capacity] = max(
                dp[current_capacity],
                dp[current_capacity - weight] + value,
            )

    return dp[capacity]


# ============================================================
# COIN CHANGE
# ============================================================


def coin_change(coins: Sequence[int], amount: int) -> int:
    """Return the minimum number of coins needed to make an amount.

    Returns -1 when the amount cannot be formed.

    Coins can be used an unlimited number of times.

    Examples
    --------
    >>> coin_change([1, 2, 5], 11)
    3
    >>> coin_change([2], 3)
    -1
    """
    _validate_non_negative_integer(amount, "amount")
    _validate_non_negative_integers(coins, "coins")

    if any(coin == 0 for coin in coins):
        raise ValueError("coins must contain only positive integers.")

    if amount == 0:
        return 0

    if not coins:
        return -1

    infinity = amount + 1
    dp = [infinity] * (amount + 1)
    dp[0] = 0

    for current_amount in range(1, amount + 1):
        for coin in coins:
            if coin <= current_amount:
                dp[current_amount] = min(
                    dp[current_amount],
                    dp[current_amount - coin] + 1,
                )

    return -1 if dp[amount] == infinity else dp[amount]


# ============================================================
# LONGEST COMMON SUBSEQUENCE
# ============================================================


def longest_common_subsequence(
    first: Sequence[T],
    second: Sequence[T],
) -> list[T]:
    """Return one longest common subsequence of two sequences.

    The original input sequences are not modified.

    Examples
    --------
    >>> longest_common_subsequence("ABCBDAB", "BDCABA")
    ['B', 'D', 'A', 'B']
    """
    _validate_sequence(first, "first")
    _validate_sequence(second, "second")

    rows = len(first)
    columns = len(second)

    dp = [[0] * (columns + 1) for _ in range(rows + 1)]

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            if first[i - 1] == second[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1],
                )

    result: list[T] = []
    i = rows
    j = columns

    while i > 0 and j > 0:
        if first[i - 1] == second[j - 1]:
            result.append(first[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    result.reverse()
    return result


# ============================================================
# LONGEST INCREASING SUBSEQUENCE
# ============================================================


def longest_increasing_subsequence(
    data: Sequence[Real],
) -> list[Real]:
    """Return one longest strictly increasing subsequence.

    Examples
    --------
    >>> longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])
    [2, 3, 7, 101]
    """
    _validate_sequence(data, "data")

    for value in data:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("All elements of data must be numeric.")

    if not data:
        return []

    values = list(data)
    n = len(values)

    lengths = [1] * n
    previous = [-1] * n

    for i in range(n):
        for j in range(i):
            if values[j] < values[i] and lengths[j] + 1 > lengths[i]:
                lengths[i] = lengths[j] + 1
                previous[i] = j

    max_length = max(lengths)
    index = lengths.index(max_length)

    result: list[Real] = []

    while index != -1:
        result.append(values[index])
        index = previous[index]

    result.reverse()
    return result


# ============================================================
# MATRIX CHAIN MULTIPLICATION
# ============================================================


def matrix_chain_multiplication(
    dimensions: Sequence[int],
) -> int:
    """Return the minimum number of scalar multiplications.

    If matrices have dimensions:

        A1: p0 x p1
        A2: p1 x p2
        ...
        An: p(n-1) x pn

    then ``dimensions`` is:

        [p0, p1, ..., pn]

    Examples
    --------
    >>> matrix_chain_multiplication([10, 20, 30])
    6000
    >>> matrix_chain_multiplication([40, 20, 30, 10, 30])
    26000
    """
    _validate_non_negative_integers(dimensions, "dimensions")

    if len(dimensions) < 2:
        raise ValueError(
            "dimensions must contain at least two dimensions."
        )

    if any(dimension == 0 for dimension in dimensions):
        raise ValueError("matrix dimensions must be positive.")

    matrix_count = len(dimensions) - 1

    if matrix_count == 1:
        return 0

    dp = [
        [0] * matrix_count
        for _ in range(matrix_count)
    ]

    for chain_length in range(2, matrix_count + 1):
        for i in range(matrix_count - chain_length + 1):
            j = i + chain_length - 1
            dp[i][j] = float("inf")

            for k in range(i, j):
                cost = (
                    dp[i][k]
                    + dp[k + 1][j]
                    + dimensions[i]
                    * dimensions[k + 1]
                    * dimensions[j + 1]
                )

                if cost < dp[i][j]:
                    dp[i][j] = cost

    return int(dp[0][matrix_count - 1])

