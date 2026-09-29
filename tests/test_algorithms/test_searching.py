
"""
Tests des algorithmes de recherche de MathLab AI.

Algorithmes testés :
- Linear Search
- Binary Search
- Jump Search
- Interpolation Search
- Exponential Search
"""

import math

import pytest

from core.algorithms.searching import (
    binary_search,
    exponential_search,
    interpolation_search,
    jump_search,
    linear_search,
)


# ============================================================
# LINEAR SEARCH
# ============================================================


class TestLinearSearch:
    """Tests de la recherche linéaire."""

    def test_finds_existing_element(self):
        data = [10, 20, 30, 40, 50]

        assert linear_search(data, 30) == 2

    def test_finds_first_element(self):
        data = [10, 20, 30, 40, 50]

        assert linear_search(data, 10) == 0

    def test_finds_last_element(self):
        data = [10, 20, 30, 40, 50]

        assert linear_search(data, 50) == 4

    def test_returns_minus_one_when_element_is_absent(self):
        data = [10, 20, 30, 40, 50]

        assert linear_search(data, 99) == -1

    def test_empty_list_returns_minus_one(self):
        assert linear_search([], 10) == -1

    def test_handles_single_element_found(self):
        assert linear_search([42], 42) == 0

    def test_handles_single_element_absent(self):
        assert linear_search([42], 10) == -1

    def test_returns_first_occurrence_with_duplicates(self):
        data = [10, 20, 20, 20, 30]

        assert linear_search(data, 20) == 1

    def test_works_with_negative_values(self):
        data = [-10, -5, 0, 5, 10]

        assert linear_search(data, -5) == 1

    def test_works_with_strings(self):
        data = ["math", "python", "algorithm"]

        assert linear_search(data, "python") == 1

    def test_rejects_string_as_data_sequence(self):
        with pytest.raises(TypeError, match="séquence"):
            linear_search("12345", 3)


# ============================================================
# BINARY SEARCH
# ============================================================


class TestBinarySearch:
    """Tests de la recherche binaire."""

    def test_finds_existing_element(self):
        data = [10, 20, 30, 40, 50]

        assert binary_search(data, 30) == 2

    def test_finds_first_element(self):
        data = [10, 20, 30, 40, 50]

        assert binary_search(data, 10) == 0

    def test_finds_last_element(self):
        data = [10, 20, 30, 40, 50]

        assert binary_search(data, 50) == 4

    def test_returns_minus_one_when_element_is_absent(self):
        data = [10, 20, 30, 40, 50]

        assert binary_search(data, 35) == -1

    def test_empty_list_returns_minus_one(self):
        assert binary_search([], 10) == -1

    def test_handles_single_element_found(self):
        assert binary_search([42], 42) == 0

    def test_handles_single_element_absent(self):
        assert binary_search([42], 10) == -1

    def test_handles_negative_values(self):
        data = [-20, -10, -5, 0, 10]

        assert binary_search(data, -5) == 2

    def test_handles_duplicates(self):
        data = [10, 20, 20, 20, 30]

        result = binary_search(data, 20)

        assert result in (1, 2, 3)

    def test_rejects_unsorted_data(self):
        data = [10, 30, 20, 40]

        with pytest.raises(ValueError, match="triées"):
            binary_search(data, 20)

    def test_accepts_already_sorted_data(self):
        data = [1, 2, 3, 4, 5]

        assert binary_search(data, 4) == 3


# ============================================================
# JUMP SEARCH
# ============================================================


class TestJumpSearch:
    """Tests de la recherche par saut."""

    def test_finds_existing_element(self):
        data = [10, 20, 30, 40, 50, 60, 70, 80]

        assert jump_search(data, 50) == 4

    def test_finds_first_element(self):
        data = [10, 20, 30, 40, 50]

        assert jump_search(data, 10) == 0

    def test_finds_last_element(self):
        data = [10, 20, 30, 40, 50]

        assert jump_search(data, 50) == 4

    def test_returns_minus_one_when_element_is_absent(self):
        data = [10, 20, 30, 40, 50]

        assert jump_search(data, 35) == -1

    def test_empty_list_returns_minus_one(self):
        assert jump_search([], 10) == -1

    def test_handles_single_element_found(self):
        assert jump_search([42], 42) == 0

    def test_handles_single_element_absent(self):
        assert jump_search([42], 10) == -1

    def test_handles_negative_values(self):
        data = [-20, -10, -5, 0, 10, 20]

        assert jump_search(data, -5) == 2

    def test_handles_duplicates(self):
        data = [10, 20, 20, 20, 30, 40]

        result = jump_search(data, 20)

        assert result in (1, 2, 3)

    def test_rejects_unsorted_data(self):
        data = [10, 30, 20, 40]

        with pytest.raises(ValueError, match="triées"):
            jump_search(data, 20)

    def test_searches_large_sequence(self):
        data = list(range(1000))

        assert jump_search(data, 789) == 789


# ============================================================
# INTERPOLATION SEARCH
# ============================================================


class TestInterpolationSearch:
    """Tests de la recherche par interpolation."""

    def test_finds_existing_element(self):
        data = [10, 20, 30, 40, 50]

        assert interpolation_search(data, 30) == 2

    def test_finds_first_element(self):
        data = [10, 20, 30, 40, 50]

        assert interpolation_search(data, 10) == 0

    def test_finds_last_element(self):
        data = [10, 20, 30, 40, 50]

        assert interpolation_search(data, 50) == 4

    def test_returns_minus_one_when_element_is_absent(self):
        data = [10, 20, 30, 40, 50]

        assert interpolation_search(data, 35) == -1

    def test_empty_list_returns_minus_one(self):
        assert interpolation_search([], 10) == -1

    def test_handles_single_element_found(self):
        assert interpolation_search([42], 42) == 0

    def test_handles_single_element_absent(self):
        assert interpolation_search([42], 10) == -1

    def test_handles_negative_values(self):
        data = [-20, -10, 0, 10, 20]

        assert interpolation_search(data, 0) == 2

    def test_handles_duplicates(self):
        data = [10, 20, 20, 20, 30]

        result = interpolation_search(data, 20)

        assert result in (1, 2, 3)

    def test_handles_uniformly_distributed_values(self):
        data = list(range(100))

        assert interpolation_search(data, 75) == 75

    def test_rejects_unsorted_data(self):
        data = [10, 30, 20, 40]

        with pytest.raises(ValueError, match="triées"):
            interpolation_search(data, 20)

    def test_rejects_non_numeric_data(self):
        data = ["a", "b", "c"]

        with pytest.raises(TypeError, match="numériques"):
            interpolation_search(data, "b")

    def test_rejects_non_numeric_target(self):
        data = [10, 20, 30]

        with pytest.raises(TypeError, match="numérique"):
            interpolation_search(data, "20")

    def test_rejects_nan_target(self):
        data = [10, 20, 30]

        with pytest.raises(ValueError, match="finie"):
            interpolation_search(data, math.nan)

    def test_rejects_infinite_target(self):
        data = [10, 20, 30]

        with pytest.raises(ValueError, match="finie"):
            interpolation_search(data, math.inf)

    def test_rejects_non_finite_data(self):
        data = [10, 20, math.inf]

        with pytest.raises(ValueError, match="finies"):
            interpolation_search(data, 20)


# ============================================================
# EXPONENTIAL SEARCH
# ============================================================


class TestExponentialSearch:
    """Tests de la recherche exponentielle."""

    def test_finds_existing_element(self):
        data = [10, 20, 30, 40, 50, 60, 70, 80]

        assert exponential_search(data, 50) == 4

    def test_finds_first_element(self):
        data = [10, 20, 30, 40, 50]

        assert exponential_search(data, 10) == 0

    def test_finds_last_element(self):
        data = [10, 20, 30, 40, 50]

        assert exponential_search(data, 50) == 4

    def test_returns_minus_one_when_element_is_absent(self):
        data = [10, 20, 30, 40, 50]

        assert exponential_search(data, 35) == -1

    def test_empty_list_returns_minus_one(self):
        assert exponential_search([], 10) == -1

    def test_handles_single_element_found(self):
        assert exponential_search([42], 42) == 0

    def test_handles_single_element_absent(self):
        assert exponential_search([42], 10) == -1

    def test_handles_negative_values(self):
        data = [-20, -10, -5, 0, 10, 20]

        assert exponential_search(data, -5) == 2

    def test_handles_duplicates(self):
        data = [10, 20, 20, 20, 30]

        result = exponential_search(data, 20)

        assert result in (1, 2, 3)

    def test_searches_large_sequence(self):
        data = list(range(1000))

        assert exponential_search(data, 789) == 789

    def test_rejects_unsorted_data(self):
        data = [10, 30, 20, 40]

        with pytest.raises(ValueError, match="triées"):
            exponential_search(data, 20)


# ============================================================
# TESTS COMPARATIFS
# ============================================================


class TestSearchAlgorithmsConsistency:
    """
    Vérifie que les différents algorithmes donnent des résultats
    cohérents sur les mêmes données.
    """

    @pytest.mark.parametrize(
        "target",
        [0, 5, 10, 25, 49],
    )
    def test_all_sorted_searches_find_same_target(self, target):
        data = list(range(50))

        binary_result = binary_search(data, target)
        jump_result = jump_search(data, target)
        interpolation_result = interpolation_search(data, target)
        exponential_result = exponential_search(data, target)

        assert binary_result == target
        assert jump_result == target
        assert interpolation_result == target
        assert exponential_result == target

    @pytest.mark.parametrize(
        "target",
        [-1, 50, 100],
    )
    def test_all_sorted_searches_return_minus_one_when_absent(self, target):
        data = list(range(50))

        assert binary_search(data, target) == -1
        assert jump_search(data, target) == -1
        assert interpolation_search(data, target) == -1
        assert exponential_search(data, target) == -1

    def test_linear_search_works_on_unsorted_data(self):
        data = [40, 10, 50, 20, 30]

        assert linear_search(data, 50) == 2

    def test_sorted_searches_work_on_same_dataset(self):
        data = [10, 20, 30, 40, 50, 60, 70]

        assert binary_search(data, 40) == 3
        assert jump_search(data, 40) == 3
        assert interpolation_search(data, 40) == 3
        assert exponential_search(data, 40) == 3

