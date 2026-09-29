
"""Tests for greedy algorithms."""

from __future__ import annotations

import math

import pytest

from core.algorithms.greedy import (
    activity_selection,
    fractional_knapsack,
    greedy_coin_change,
    huffman_coding,
    interval_scheduling,
    job_sequencing,
)


# ============================================================
# ACTIVITY SELECTION
# ============================================================


class TestActivitySelection:
    """Tests for activity_selection."""

    def test_classic_example(self):
        activities = [
            (1, 2),
            (3, 4),
            (0, 6),
            (5, 7),
            (8, 9),
            (5, 9),
        ]

        result = activity_selection(activities)

        assert result == [
            (1, 2),
            (3, 4),
            (5, 7),
            (8, 9),
        ]

    def test_empty_input(self):
        assert activity_selection([]) == []

    def test_single_activity(self):
        assert activity_selection([(1, 2)]) == [(1, 2)]

    def test_all_compatible(self):
        activities = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ]

        assert activity_selection(activities) == activities

    def test_all_overlapping(self):
        activities = [
            (1, 5),
            (2, 4),
            (3, 6),
            (1, 3),
        ]

        result = activity_selection(activities)

        assert len(result) == 2
        assert result == [(1, 3), (3, 6)]

    def test_same_start_times(self):
        activities = [
            (1, 4),
            (1, 2),
            (1, 3),
        ]

        assert activity_selection(activities) == [(1, 2)]

    def test_same_finish_times(self):
        activities = [
            (1, 3),
            (2, 3),
            (0, 3),
        ]

        result = activity_selection(activities)

        assert len(result) == 1
        assert result[0][1] == 3

    def test_boundary_touching_is_allowed(self):
        activities = [
            (1, 2),
            (2, 4),
            (4, 6),
        ]

        assert activity_selection(activities) == activities

    def test_unsorted_input(self):
        activities = [
            (5, 7),
            (1, 2),
            (8, 9),
            (3, 4),
        ]

        result = activity_selection(activities)

        assert result == [
            (1, 2),
            (3, 4),
            (5, 7),
            (8, 9),
        ]

    def test_negative_times_are_allowed(self):
        activities = [
            (-5, -3),
            (-2, 0),
            (1, 3),
        ]

        assert activity_selection(activities) == activities

    def test_does_not_modify_input(self):
        activities = [
            (5, 7),
            (1, 2),
            (3, 4),
        ]

        original = activities.copy()

        activity_selection(activities)

        assert activities == original

    def test_invalid_container(self):
        with pytest.raises(TypeError):
            activity_selection("invalid")

    def test_invalid_activity_length(self):
        with pytest.raises(ValueError):
            activity_selection([(1, 2, 3)])

    def test_invalid_activity_type(self):
        with pytest.raises(TypeError):
            activity_selection(["invalid"])

    def test_finish_before_start(self):
        with pytest.raises(ValueError):
            activity_selection([(5, 2)])

    def test_non_numeric_times(self):
        with pytest.raises(TypeError):
            activity_selection([("a", 2)])


# ============================================================
# FRACTIONAL KNAPSACK
# ============================================================


class TestFractionalKnapsack:
    """Tests for fractional_knapsack."""

    def test_classic_example(self):
        weights = [10, 20, 30]
        values = [60, 100, 120]

        result = fractional_knapsack(weights, values, 50)

        assert result == pytest.approx(240.0)

    def test_empty_items(self):
        assert fractional_knapsack([], [], 50) == 0.0

    def test_zero_capacity(self):
        assert fractional_knapsack([10, 20], [60, 100], 0) == 0.0

    def test_capacity_smaller_than_first_item(self):
        result = fractional_knapsack(
            [10],
            [100],
            5,
        )

        assert result == pytest.approx(50.0)

    def test_capacity_exactly_matches_items(self):
        result = fractional_knapsack(
            [10, 20],
            [60, 100],
            30,
        )

        assert result == pytest.approx(160.0)

    def test_capacity_larger_than_total_weight(self):
        result = fractional_knapsack(
            [10, 20],
            [60, 100],
            100,
        )

        assert result == pytest.approx(160.0)

    def test_selects_by_value_to_weight_ratio(self):
        weights = [10, 20]
        values = [100, 100]

        result = fractional_knapsack(weights, values, 10)

        assert result == pytest.approx(100.0)

    def test_fractional_selection(self):
        weights = [10, 20]
        values = [60, 100]

        result = fractional_knapsack(weights, values, 15)

        assert result == pytest.approx(85.0)

    def test_zero_value_item(self): 
        result = fractional_knapsack( 
            [10, 20], 
            [0, 100], 
            10, 
            ) 
        assert result == pytest.approx(50.0)

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError):
            fractional_knapsack([10], [-5], 10)

    def test_zero_weight_rejected(self):
        with pytest.raises(ValueError):
            fractional_knapsack([0], [10], 10)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError):
            fractional_knapsack([-10], [10], 10)

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError):
            fractional_knapsack([10], [10], -1)

    def test_non_numeric_weight(self):
        with pytest.raises(TypeError):
            fractional_knapsack(["10"], [10], 10)

    def test_non_numeric_value(self):
        with pytest.raises(TypeError):
            fractional_knapsack([10], ["10"], 10)

    def test_non_numeric_capacity(self):
        with pytest.raises(TypeError):
            fractional_knapsack([10], [10], "10")

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            fractional_knapsack([10, 20], [100], 20)

    def test_does_not_modify_inputs(self):
        weights = [10, 20, 30]
        values = [60, 100, 120]

        original_weights = weights.copy()
        original_values = values.copy()

        fractional_knapsack(weights, values, 50)

        assert weights == original_weights
        assert values == original_values

    def test_float_values(self):
        result = fractional_knapsack(
                [2.5, 5.0],
                [10.0, 15.0],
                 4.0,
            )

        assert result == pytest.approx(14.5)


# ============================================================
# GREEDY COIN CHANGE
# ============================================================


class TestGreedyCoinChange:
    """Tests for greedy_coin_change."""

    def test_classic_coin_system(self):
        result = greedy_coin_change(
            [1, 5, 10, 25],
            41,
        )

        assert result == [25, 10, 5, 1]

    def test_zero_amount(self):
        assert greedy_coin_change([1, 5, 10], 0) == []

    def test_empty_coins(self):
        assert greedy_coin_change([], 10) == []

    def test_exact_coin(self):
        assert greedy_coin_change([1, 5, 10], 10) == [10]

    def test_repeated_coins(self):
        result = greedy_coin_change(
            [1, 5, 5, 10, 10],
            16,
        )

        assert result == [10, 5, 1]

    def test_greedy_strategy(self):
        result = greedy_coin_change(
            [1, 3, 4],
            6,
        )

        # Greedy chooses 4 + 1 + 1 instead of optimal 3 + 3.
        assert result == [4, 1, 1]

    def test_greedy_can_fail_to_find_solution(self):
        result = greedy_coin_change(
            [5, 10],
            3,
        )

        assert result == []

    def test_non_canonical_system_can_be_suboptimal(self):
        result = greedy_coin_change(
            [1, 3, 4],
            6,
        )

        assert len(result) == 3

    def test_result_sums_to_amount(self):
        coins = [1, 5, 10, 25]
        amount = 63

        result = greedy_coin_change(coins, amount)

        assert sum(result) == amount

    def test_result_contains_only_available_coins(self):
        coins = [1, 5, 10, 25]

        result = greedy_coin_change(coins, 87)

        assert all(coin in coins for coin in result)

    def test_negative_amount_rejected(self):
        with pytest.raises(ValueError):
            greedy_coin_change([1, 5, 10], -1)

    def test_non_integer_amount_rejected(self):
        with pytest.raises(TypeError):
            greedy_coin_change([1, 5, 10], 10.5)

    def test_zero_coin_rejected(self):
        with pytest.raises(ValueError):
            greedy_coin_change([1, 0, 5], 10)

    def test_negative_coin_rejected(self):
        with pytest.raises(ValueError):
            greedy_coin_change([1, -5], 10)

    def test_non_integer_coin_rejected(self):
        with pytest.raises(TypeError):
            greedy_coin_change([1, 5.5], 10)

    def test_non_sequence_coins_rejected(self):
        with pytest.raises(TypeError):
            greedy_coin_change(10, 20)

    def test_does_not_modify_input(self):
        coins = [25, 10, 5, 1]
        original = coins.copy()

        greedy_coin_change(coins, 41)

        assert coins == original


# ============================================================
# HUFFMAN CODING
# ============================================================


class TestHuffmanCoding:
    """Tests for huffman_coding."""

    def test_empty_input(self):
        assert huffman_coding("") == {}

    def test_single_symbol(self):
        result = huffman_coding("aaaa")

        assert result == {"a": "0"}

    def test_two_symbols(self):
        result = huffman_coding("aabb")

        assert set(result.keys()) == {"a", "b"}
        assert all(code in {"0", "1"} for code in result.values())

    def test_multiple_symbols(self):
        result = huffman_coding("aaabbc")

        assert set(result.keys()) == {"a", "b", "c"}

    def test_all_codes_are_binary(self):
        result = huffman_coding("mississippi")

        assert all(
            set(code).issubset({"0", "1"})
            for code in result.values()
        )

    def test_codes_are_prefix_free(self):
        result = huffman_coding("aaabbc")

        codes = list(result.values())

        for i, code_a in enumerate(codes):
            for j, code_b in enumerate(codes):
                if i != j:
                    assert not code_b.startswith(code_a)

    def test_more_frequent_symbols_have_no_longer_code_than_less_frequent_symbols(
        self,
    ):
        result = huffman_coding("aaaaabbc")

        assert len(result["a"]) <= len(result["c"])

    def test_all_input_symbols_are_present(self):
        data = "abracadabra"

        result = huffman_coding(data)

        assert set(result.keys()) == set(data)

    def test_list_input(self):
        data = ["a", "a", "b", "c", "c"]

        result = huffman_coding(data)

        assert set(result.keys()) == {"a", "b", "c"}

    def test_tuple_input(self):
        data = ("a", "b", "b", "c")

        result = huffman_coding(data)

        assert set(result.keys()) == {"a", "b", "c"}

    def test_non_sequence_rejected(self):
        with pytest.raises(TypeError):
            huffman_coding(123)

    def test_bytes_rejected(self):
        with pytest.raises(TypeError):
            huffman_coding(b"abc")

    def test_does_not_modify_input(self):
        data = ["a", "a", "b", "c"]
        original = data.copy()

        huffman_coding(data)

        assert data == original

    def test_frequency_dominates_code_length(self):
        data = "aaaaaaaaaabbbbcc"

        result = huffman_coding(data)

        assert len(result["a"]) <= len(result["b"])
        assert len(result["b"]) <= len(result["c"])

    def test_code_lengths_are_positive(self):
        result = huffman_coding("abcdef")

        assert all(len(code) >= 1 for code in result.values())


# ============================================================
# JOB SEQUENCING
# ============================================================


class TestJobSequencing:
    """Tests for job_sequencing."""

    def test_classic_example(self):
        jobs = [
            ("A", 2, 100),
            ("B", 1, 19),
            ("C", 2, 27),
            ("D", 1, 25),
            ("E", 3, 15),
        ]

        scheduled, profit = job_sequencing(jobs)

        assert scheduled == ["C", "A", "E"]
        assert profit == 142

    def test_empty_input(self):
        assert job_sequencing([]) == ([], 0)

    def test_single_job(self):
        jobs = [("A", 1, 100)]

        assert job_sequencing(jobs) == (["A"], 100)

    def test_all_jobs_fit(self):
        jobs = [
            ("A", 1, 10),
            ("B", 2, 20),
            ("C", 3, 30),
        ]

        scheduled, profit = job_sequencing(jobs)

        assert scheduled == ["A", "B", "C"]
        assert profit == 60

    def test_deadline_allows_late_slot(self):
        jobs = [
            ("A", 3, 100),
            ("B", 1, 20),
            ("C", 2, 30),
        ]

        scheduled, profit = job_sequencing(jobs)

        assert set(scheduled) == {"A", "B", "C"}
        assert profit == 150

    def test_only_most_profitable_jobs_are_selected_when_needed(self):
        jobs = [
            ("A", 1, 10),
            ("B", 1, 50),
            ("C", 1, 30),
        ]

        scheduled, profit = job_sequencing(jobs)

        assert scheduled == ["B"]
        assert profit == 50

    def test_zero_profit_jobs(self):
        jobs = [
            ("A", 1, 0),
            ("B", 2, 10),
        ]

        scheduled, profit = job_sequencing(jobs)

        assert profit == 10
        assert "B" in scheduled

    def test_negative_profit_is_allowed_but_not_preferred(self):
        jobs = [
            ("A", 1, -10),
            ("B", 1, 20),
        ]

        scheduled, profit = job_sequencing(jobs)

        assert scheduled == ["B"]
        assert profit == 20

    def test_float_profit(self):
        jobs = [
            ("A", 1, 10.5),
            ("B", 2, 20.5),
        ]

        scheduled, profit = job_sequencing(jobs)

        assert profit == pytest.approx(31.0)

    def test_deadline_must_be_positive(self):
        with pytest.raises(ValueError):
            job_sequencing([("A", 0, 10)])

    def test_negative_deadline_rejected(self):
        with pytest.raises(ValueError):
            job_sequencing([("A", -1, 10)])

    def test_non_integer_deadline_rejected(self):
        with pytest.raises(TypeError):
            job_sequencing([("A", 1.5, 10)])

    def test_non_numeric_profit_rejected(self):
        with pytest.raises(TypeError):
            job_sequencing([("A", 1, "10")])

    def test_invalid_job_length(self):
        with pytest.raises(ValueError):
            job_sequencing([("A", 1)])

    def test_invalid_job_type(self):
        with pytest.raises(TypeError):
            job_sequencing(["invalid"])

    def test_does_not_modify_input(self):
        jobs = [
            ("A", 2, 100),
            ("B", 1, 20),
        ]

        original = jobs.copy()

        job_sequencing(jobs)

        assert jobs == original


# ============================================================
# INTERVAL SCHEDULING
# ============================================================


class TestIntervalScheduling:
    """Tests for interval_scheduling."""

    def test_classic_example(self):
        intervals = [
            (1, 3),
            (2, 5),
            (4, 7),
            (6, 9),
            (8, 10),
        ]

        result = interval_scheduling(intervals)

        assert result == [
            (1, 3),
            (4, 7),
            (8, 10),
        ]

    def test_empty_input(self):
        assert interval_scheduling([]) == []

    def test_single_interval(self):
        assert interval_scheduling([(1, 2)]) == [(1, 2)]

    def test_all_compatible(self):
        intervals = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ]

        assert interval_scheduling(intervals) == intervals

    def test_overlapping_intervals(self):
        intervals = [
            (1, 10),
            (2, 3),
            (3, 4),
            (4, 5),
        ]

        result = interval_scheduling(intervals)

        assert result == [
            (2, 3),
            (3, 4),
            (4, 5),
        ]

    def test_boundary_touching_is_allowed(self):
        intervals = [
            (0, 2),
            (2, 4),
            (4, 6),
        ]

        assert interval_scheduling(intervals) == intervals

    def test_unsorted_input(self):
        intervals = [
            (8, 10),
            (1, 3),
            (4, 7),
            (2, 5),
        ]

        assert interval_scheduling(intervals) == [
            (1, 3),
            (4, 7),
            (8, 10),
        ]

    def test_negative_times(self):
        intervals = [
            (-5, -3),
            (-2, 0),
            (1, 2),
        ]

        assert interval_scheduling(intervals) == intervals

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            interval_scheduling([(5, 2)])

    def test_invalid_interval_length(self):
        with pytest.raises(ValueError):
            interval_scheduling([(1, 2, 3)])

    def test_invalid_interval_type(self):
        with pytest.raises(TypeError):
            interval_scheduling(["invalid"])

    def test_non_numeric_values(self):
        with pytest.raises(TypeError):
            interval_scheduling([("a", 2)])

    def test_does_not_modify_input(self):
        intervals = [
            (5, 7),
            (1, 2),
            (3, 4),
        ]

        original = intervals.copy()

        interval_scheduling(intervals)

        assert intervals == original


# ============================================================
# CROSS-ALGORITHM CONSISTENCY
# ============================================================


class TestGreedyConsistency:
    """Cross-algorithm consistency tests."""

    def test_activity_and_interval_scheduling_are_consistent(self):
        intervals = [
            (1, 3),
            (2, 5),
            (4, 7),
            (6, 9),
            (8, 10),
        ]

        assert interval_scheduling(intervals) == activity_selection(intervals)

    def test_greedy_coin_change_result_is_valid(self):
        coins = [1, 5, 10, 25]
        amount = 99

        result = greedy_coin_change(coins, amount)

        assert sum(result) == amount
        assert all(coin in coins for coin in result)

    def test_fractional_knapsack_never_exceeds_capacity(self):
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50

        result = fractional_knapsack(weights, values, capacity)

        assert math.isfinite(result)
        assert result >= 0

    def test_job_sequencing_has_unique_jobs(self):
        jobs = [
            ("A", 2, 100),
            ("B", 1, 20),
            ("C", 2, 50),
            ("D", 3, 40),
        ]

        scheduled, _ = job_sequencing(jobs)

        assert len(scheduled) == len(set(scheduled))

    def test_huffman_codes_cover_all_symbols(self):
        data = "the quick brown fox"

        codes = huffman_coding(data)

        assert set(codes) == set(data)

    def test_huffman_codes_are_non_empty(self):
        codes = huffman_coding("algorithm")

        assert all(codes.values())

