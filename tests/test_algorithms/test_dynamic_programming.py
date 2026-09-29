
"""Tests for Dynamic Programming algorithms.

Test coverage:
- Fibonacci
- Climbing Stairs
- 0/1 Knapsack
- Coin Change
- Longest Common Subsequence (LCS)
- Longest Increasing Subsequence (LIS)
- Matrix Chain Multiplication
"""

import pytest

from core.algorithms.dynamic_programming import (
    climbing_stairs,
    coin_change,
    fibonacci,
    knapsack_01,
    longest_common_subsequence,
    longest_increasing_subsequence,
    matrix_chain_multiplication,
)


# ============================================================
# FIBONACCI
# ============================================================


class TestFibonacci:
    """Tests for fibonacci()."""

    def test_fibonacci_zero(self):
        assert fibonacci(0) == 0

    def test_fibonacci_one(self):
        assert fibonacci(1) == 1

    def test_fibonacci_two(self):
        assert fibonacci(2) == 1

    def test_fibonacci_small_values(self):
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

        for n, expected_value in enumerate(expected):
            assert fibonacci(n) == expected_value

    def test_fibonacci_ten(self):
        assert fibonacci(10) == 55

    def test_fibonacci_twenty(self):
        assert fibonacci(20) == 6765

    def test_fibonacci_larger_value(self):
        assert fibonacci(30) == 832040

    def test_fibonacci_returns_integer(self):
        assert isinstance(fibonacci(15), int)

    def test_fibonacci_rejects_negative_integer(self):
        with pytest.raises(ValueError):
            fibonacci(-1)

    def test_fibonacci_rejects_large_negative_integer(self):
        with pytest.raises(ValueError):
            fibonacci(-100)

    @pytest.mark.parametrize("value", [1.5, 2.0, "10", None, []])
    def test_fibonacci_rejects_non_integer(self, value):
        with pytest.raises(TypeError):
            fibonacci(value)

    def test_fibonacci_rejects_boolean(self):
        with pytest.raises(TypeError):
            fibonacci(True)


# ============================================================
# CLIMBING STAIRS
# ============================================================


class TestClimbingStairs:
    """Tests for climbing_stairs()."""

    def test_climbing_stairs_zero(self):
        assert climbing_stairs(0) == 1

    def test_climbing_stairs_one(self):
        assert climbing_stairs(1) == 1

    def test_climbing_stairs_two(self):
        assert climbing_stairs(2) == 2

    def test_climbing_stairs_small_values(self):
        expected = [1, 1, 2, 3, 5, 8, 13, 21]

        for n, expected_value in enumerate(expected):
            assert climbing_stairs(n) == expected_value

    def test_climbing_stairs_five(self):
        assert climbing_stairs(5) == 8

    def test_climbing_stairs_ten(self):
        assert climbing_stairs(10) == 89

    def test_climbing_stairs_twenty(self):
        assert climbing_stairs(20) == 10946

    def test_climbing_stairs_returns_integer(self):
        assert isinstance(climbing_stairs(10), int)

    def test_climbing_stairs_rejects_negative(self):
        with pytest.raises(ValueError):
            climbing_stairs(-1)

    @pytest.mark.parametrize("value", [1.5, 2.0, "5", None, []])
    def test_climbing_stairs_rejects_non_integer(self, value):
        with pytest.raises(TypeError):
            climbing_stairs(value)

    def test_climbing_stairs_rejects_boolean(self):
        with pytest.raises(TypeError):
            climbing_stairs(True)


# ============================================================
# 0/1 KNAPSACK
# ============================================================


class TestKnapsack01:
    """Tests for knapsack_01()."""

    def test_knapsack_classic_example(self):
        weights = [1, 2, 3]
        values = [6, 10, 12]

        assert knapsack_01(weights, values, 5) == 22

    def test_knapsack_selects_best_combination(self):
        weights = [2, 3, 4, 5]
        values = [3, 4, 5, 6]

        assert knapsack_01(weights, values, 5) == 7

    def test_knapsack_zero_capacity(self):
        weights = [1, 2, 3]
        values = [10, 20, 30]

        assert knapsack_01(weights, values, 0) == 0

    def test_knapsack_empty_items(self):
        assert knapsack_01([], [], 10) == 0

    def test_knapsack_single_item_fits(self):
        assert knapsack_01([5], [10], 5) == 10

    def test_knapsack_single_item_does_not_fit(self):
        assert knapsack_01([6], [10], 5) == 0

    def test_knapsack_item_used_at_most_once(self):
        weights = [2]
        values = [10]

        assert knapsack_01(weights, values, 6) == 10

    def test_knapsack_handles_zero_weight_item(self):
        weights = [0, 2]
        values = [10, 5]

        assert knapsack_01(weights, values, 2) == 15

    def test_knapsack_handles_negative_values(self):
        weights = [1, 2]
        values = [-10, 5]

        assert knapsack_01(weights, values, 2) == 5

    def test_knapsack_returns_numeric_value(self):
        result = knapsack_01([1, 2], [2.5, 4.5], 2)

        assert result == 4.5

    def test_knapsack_does_not_modify_weights(self):
        weights = [1, 2, 3]
        original = weights.copy()

        knapsack_01(weights, [6, 10, 12], 5)

        assert weights == original

    def test_knapsack_does_not_modify_values(self):
        values = [6, 10, 12]
        original = values.copy()

        knapsack_01([1, 2, 3], values, 5)

        assert values == original

    def test_knapsack_rejects_different_lengths(self):
        with pytest.raises(ValueError):
            knapsack_01([1, 2], [10], 5)

    def test_knapsack_rejects_negative_weight(self):
        with pytest.raises(ValueError):
            knapsack_01([-1, 2], [10, 20], 5)

    def test_knapsack_rejects_non_integer_weight(self):
        with pytest.raises(TypeError):
            knapsack_01([1.5, 2], [10, 20], 5)

    def test_knapsack_rejects_non_numeric_value(self):
        with pytest.raises(TypeError):
            knapsack_01([1, 2], [10, "20"], 5)

    def test_knapsack_rejects_negative_capacity(self):
        with pytest.raises(ValueError):
            knapsack_01([1, 2], [10, 20], -1)

    def test_knapsack_rejects_non_integer_capacity(self):
        with pytest.raises(TypeError):
            knapsack_01([1, 2], [10, 20], 2.5)

    def test_knapsack_rejects_boolean_capacity(self):
        with pytest.raises(TypeError):
            knapsack_01([1], [10], True)

    def test_knapsack_rejects_non_sequence_weights(self):
        with pytest.raises(TypeError):
            knapsack_01(123, [10], 5)

    def test_knapsack_rejects_non_sequence_values(self):
        with pytest.raises(TypeError):
            knapsack_01([1], 10, 5)


# ============================================================
# COIN CHANGE
# ============================================================


class TestCoinChange:
    """Tests for coin_change()."""

    def test_coin_change_classic_example(self):
        assert coin_change([1, 2, 5], 11) == 3

    def test_coin_change_impossible_amount(self):
        assert coin_change([2], 3) == -1

    def test_coin_change_zero_amount(self):
        assert coin_change([1, 2, 5], 0) == 0

    def test_coin_change_single_coin(self):
        assert coin_change([1], 5) == 5

    def test_coin_change_exact_coin(self):
        assert coin_change([5], 5) == 1

    def test_coin_change_prefers_minimum_number(self):
        assert coin_change([1, 3, 4], 6) == 2

    def test_coin_change_multiple_combinations(self):
        assert coin_change([1, 2, 5], 10) == 2

    def test_coin_change_empty_coins(self):
        assert coin_change([], 10) == -1

    def test_coin_change_empty_coins_zero_amount(self):
        assert coin_change([], 0) == 0

    def test_coin_change_duplicate_coins(self):
        assert coin_change([1, 1, 2, 2], 4) == 2

    def test_coin_change_does_not_modify_coins(self):
        coins = [1, 2, 5]
        original = coins.copy()

        coin_change(coins, 11)

        assert coins == original

    def test_coin_change_rejects_negative_amount(self):
        with pytest.raises(ValueError):
            coin_change([1, 2, 5], -1)

    def test_coin_change_rejects_non_integer_amount(self):
        with pytest.raises(TypeError):
            coin_change([1, 2, 5], 2.5)

    def test_coin_change_rejects_zero_coin(self):
        with pytest.raises(ValueError):
            coin_change([0, 1, 2], 5)

    def test_coin_change_rejects_negative_coin(self):
        with pytest.raises(ValueError):
            coin_change([-1, 2, 5], 5)

    def test_coin_change_rejects_non_integer_coin(self):
        with pytest.raises(TypeError):
            coin_change([1, 2.5, 5], 5)

    def test_coin_change_rejects_boolean_coin(self):
        with pytest.raises(TypeError):
            coin_change([1, True, 5], 5)

    def test_coin_change_rejects_non_sequence_coins(self):
        with pytest.raises(TypeError):
            coin_change(5, 10)


# ============================================================
# LONGEST COMMON SUBSEQUENCE
# ============================================================


class TestLongestCommonSubsequence:
    """Tests for longest_common_subsequence()."""

    def test_lcs_classic_example(self):
        result = longest_common_subsequence(
            "ABCBDAB",
            "BDCABA",
        )

        assert len(result) == 4
        assert "".join(result) in {"BCBA", "BDAB"}

    def test_lcs_simple_case(self):
        result = longest_common_subsequence(
            "ABC",
            "AC",
        )

        assert result == ["A", "C"]

    def test_lcs_identical_sequences(self):
        result = longest_common_subsequence(
            "HELLO",
            "HELLO",
        )

        assert result == list("HELLO")

    def test_lcs_no_common_elements(self):
        result = longest_common_subsequence(
            "ABC",
            "XYZ",
        )

        assert result == []

    def test_lcs_empty_first(self):
        result = longest_common_subsequence(
            "",
            "ABC",
        )

        assert result == []

    def test_lcs_empty_second(self):
        result = longest_common_subsequence(
            "ABC",
            "",
        )

        assert result == []

    def test_lcs_both_empty(self):
        assert longest_common_subsequence("", "") == []

    def test_lcs_single_matching_element(self):
        result = longest_common_subsequence(
            "A",
            "A",
        )

        assert result == ["A"]

    def test_lcs_single_non_matching_element(self):
        result = longest_common_subsequence(
            "A",
            "B",
        )

        assert result == []

    def test_lcs_preserves_subsequence_property(self):
        first = "AGGTAB"
        second = "GXTXAYB"

        result = longest_common_subsequence(first, second)

        assert "".join(result) == "GTAB"

    def test_lcs_result_has_optimal_length(self):
        first = "ABCBDAB"
        second = "BDCABA"

        result = longest_common_subsequence(first, second)

        assert len(result) == 4

    def test_lcs_supports_lists(self):
        result = longest_common_subsequence(
            [1, 2, 3, 4],
            [2, 4],
        )

        assert result == [2, 4]

    def test_lcs_does_not_modify_first(self):
        first = [1, 2, 3]
        original = first.copy()

        longest_common_subsequence(first, [2, 3])

        assert first == original

    def test_lcs_does_not_modify_second(self):
        second = [2, 3]
        original = second.copy()

        longest_common_subsequence([1, 2, 3], second)

        assert second == original

    def test_lcs_rejects_non_sequence_first(self):
        with pytest.raises(TypeError):
            longest_common_subsequence(123, "ABC")

    def test_lcs_rejects_non_sequence_second(self):
        with pytest.raises(TypeError):
            longest_common_subsequence("ABC", 123)


# ============================================================
# LONGEST INCREASING SUBSEQUENCE
# ============================================================


class TestLongestIncreasingSubsequence:
    """Tests for longest_increasing_subsequence()."""

    def test_lis_classic_example(self):
        data = [10, 9, 2, 5, 3, 7, 101, 18]

        result = longest_increasing_subsequence(data)

        assert len(result) == 4
        assert all(
            result[index] < result[index + 1]
            for index in range(len(result) - 1)
        )

    def test_lis_simple_case(self):
        result = longest_increasing_subsequence([1, 2, 3])

        assert result == [1, 2, 3]

    def test_lis_decreasing_sequence(self):
        result = longest_increasing_subsequence([5, 4, 3, 2, 1])

        assert len(result) == 1
        assert result[0] in {1, 2, 3, 4, 5}

    def test_lis_empty_sequence(self):
        assert longest_increasing_subsequence([]) == []

    def test_lis_single_element(self):
        assert longest_increasing_subsequence([42]) == [42]

    def test_lis_all_equal(self):
        result = longest_increasing_subsequence([5, 5, 5, 5])

        assert len(result) == 1
        assert result == [5]

    def test_lis_handles_duplicates(self):
        data = [1, 3, 2, 3, 4]

        result = longest_increasing_subsequence(data)

        assert len(result) == 4
        assert all(
            result[index] < result[index + 1]
            for index in range(len(result) - 1)
        )

    def test_lis_handles_negative_values(self):
        data = [-5, -2, -4, -1, 0]

        result = longest_increasing_subsequence(data)

        assert len(result) == 4
        assert result == [-5, -4, -1, 0] or result == [-5, -2, -1, 0]

    def test_lis_handles_floats(self):
        result = longest_increasing_subsequence(
            [1.5, 2.5, 1.8, 3.0]
        )

        assert len(result) == 3
        assert all(
            result[index] < result[index + 1]
            for index in range(len(result) - 1)
        )

    def test_lis_returns_subsequence_of_original(self):
        data = [10, 22, 9, 33, 21, 50, 41, 60]

        result = longest_increasing_subsequence(data)

        iterator = iter(data)

        for value in result:
            assert value in iterator

    def test_lis_does_not_modify_input(self):
        data = [10, 9, 2, 5, 3, 7]
        original = data.copy()

        longest_increasing_subsequence(data)

        assert data == original

    def test_lis_rejects_non_numeric_value(self):
        with pytest.raises(TypeError):
            longest_increasing_subsequence([1, 2, "3"])

    def test_lis_rejects_boolean_value(self):
        with pytest.raises(TypeError):
            longest_increasing_subsequence([1, True, 3])

    def test_lis_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            longest_increasing_subsequence(123)


# ============================================================
# MATRIX CHAIN MULTIPLICATION
# ============================================================


class TestMatrixChainMultiplication:
    """Tests for matrix_chain_multiplication()."""

    def test_matrix_chain_two_matrices(self):
        assert matrix_chain_multiplication([10, 20, 30]) == 6000

    def test_matrix_chain_classic_example(self):
        dimensions = [40, 20, 30, 10, 30]

        assert matrix_chain_multiplication(dimensions) == 26000

    def test_matrix_chain_another_example(self):
        dimensions = [10, 20, 30, 40, 30]

        assert matrix_chain_multiplication(dimensions) == 30000

    def test_matrix_chain_three_matrices(self):
        dimensions = [10, 30, 5, 60]

        assert matrix_chain_multiplication(dimensions) == 4500

    def test_matrix_chain_single_matrix(self):
        assert matrix_chain_multiplication([10, 20]) == 0

    def test_matrix_chain_large_example(self):
        dimensions = [5, 10, 3, 12, 5, 50, 6]

        assert matrix_chain_multiplication(dimensions) == 2010

    def test_matrix_chain_does_not_modify_dimensions(self):
        dimensions = [10, 20, 30, 40]
        original = dimensions.copy()

        matrix_chain_multiplication(dimensions)

        assert dimensions == original

    def test_matrix_chain_rejects_empty_dimensions(self):
        with pytest.raises(ValueError):
            matrix_chain_multiplication([])

    def test_matrix_chain_rejects_single_dimension(self):
        with pytest.raises(ValueError):
            matrix_chain_multiplication([10])

    def test_matrix_chain_rejects_zero_dimension(self):
        with pytest.raises(ValueError):
            matrix_chain_multiplication([10, 0, 20])

    def test_matrix_chain_rejects_negative_dimension(self):
        with pytest.raises(ValueError):
            matrix_chain_multiplication([10, -20, 30])

    def test_matrix_chain_rejects_non_integer_dimension(self):
        with pytest.raises(TypeError):
            matrix_chain_multiplication([10, 20.5, 30])

    def test_matrix_chain_rejects_boolean_dimension(self):
        with pytest.raises(TypeError):
            matrix_chain_multiplication([10, True, 30])

    def test_matrix_chain_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            matrix_chain_multiplication(123)


# ============================================================
# CROSS-ALGORITHM CONSISTENCY
# ============================================================


class TestDynamicProgrammingConsistency:
    """Cross-algorithm sanity checks."""

    def test_fibonacci_and_climbing_stairs_relationship(self):
        # Number of ways to climb n stairs equals F(n + 1).
        for n in range(10):
            assert climbing_stairs(n) == fibonacci(n + 1)

    def test_coin_change_result_is_valid(self):
        coins = [1, 3, 4]
        amount = 6

        result = coin_change(coins, amount)

        assert result == 2

    def test_lcs_result_is_common_subsequence(self):
        first = "ABCBDAB"
        second = "BDCABA"

        result = longest_common_subsequence(first, second)

        def is_subsequence(candidate, sequence):
            iterator = iter(sequence)

            return all(
                any(value == current for current in iterator)
                for value in candidate
            )

        assert is_subsequence(result, first)
        assert is_subsequence(result, second)

    def test_lis_result_is_strictly_increasing(self):
        data = [10, 9, 2, 5, 3, 7, 101, 18]

        result = longest_increasing_subsequence(data)

        assert all(
            result[index] < result[index + 1]
            for index in range(len(result) - 1)
        )

    def test_knapsack_capacity_is_respected(self):
        weights = [1, 2, 3]
        values = [6, 10, 12]
        capacity = 5

        # The optimal solution is items with weights 2 + 3.
        result = knapsack_01(weights, values, capacity)

        assert result == 22

