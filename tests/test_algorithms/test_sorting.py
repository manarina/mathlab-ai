
"""
Tests des algorithmes de tri de MathLab AI.

Algorithmes testés :
- Bubble Sort
- Selection Sort
- Insertion Sort
- Merge Sort
- Quick Sort
- Heap Sort
- Counting Sort
- Radix Sort
- Bucket Sort
"""

import math

import pytest

from core.algorithms.sorting import (
    bubble_sort,
    bucket_sort,
    counting_sort,
    heap_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
    radix_sort,
    selection_sort,
)


# ============================================================
# DONNÉES COMMUNES
# ============================================================


SORTING_FUNCTIONS = [
    bubble_sort,
    selection_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
    heap_sort,
]


# ============================================================
# BUBBLE SORT
# ============================================================


class TestBubbleSort:
    """Tests de l'algorithme Bubble Sort."""

    def test_sorts_unsorted_list(self):
        data = [5, 2, 8, 1, 3]

        assert bubble_sort(data) == [1, 2, 3, 5, 8]

    def test_empty_list(self):
        assert bubble_sort([]) == []

    def test_single_element(self):
        assert bubble_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert bubble_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [5, 4, 3, 2, 1]

        assert bubble_sort(data) == [1, 2, 3, 4, 5]

    def test_negative_values(self):
        data = [3, -1, -5, 2, 0]

        assert bubble_sort(data) == [-5, -1, 0, 2, 3]

    def test_duplicates(self):
        data = [4, 2, 4, 1, 2]

        assert bubble_sort(data) == [1, 2, 2, 4, 4]

    def test_does_not_modify_original_data(self):
        data = [5, 2, 3, 1]
        original = data.copy()

        bubble_sort(data)

        assert data == original

    def test_works_with_strings(self):
        data = ["python", "algorithm", "math"]

        assert bubble_sort(data) == ["algorithm", "math", "python"]


# ============================================================
# SELECTION SORT
# ============================================================


class TestSelectionSort:
    """Tests de l'algorithme Selection Sort."""

    def test_sorts_unsorted_list(self):
        data = [5, 2, 8, 1, 3]

        assert selection_sort(data) == [1, 2, 3, 5, 8]

    def test_empty_list(self):
        assert selection_sort([]) == []

    def test_single_element(self):
        assert selection_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert selection_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [5, 4, 3, 2, 1]

        assert selection_sort(data) == [1, 2, 3, 4, 5]

    def test_negative_values(self):
        data = [3, -1, -5, 2, 0]

        assert selection_sort(data) == [-5, -1, 0, 2, 3]

    def test_duplicates(self):
        data = [4, 2, 4, 1, 2]

        assert selection_sort(data) == [1, 2, 2, 4, 4]

    def test_does_not_modify_original_data(self):
        data = [5, 2, 3, 1]
        original = data.copy()

        selection_sort(data)

        assert data == original

    def test_works_with_strings(self):
        data = ["python", "algorithm", "math"]

        assert selection_sort(data) == ["algorithm", "math", "python"]


# ============================================================
# INSERTION SORT
# ============================================================


class TestInsertionSort:
    """Tests de l'algorithme Insertion Sort."""

    def test_sorts_unsorted_list(self):
        data = [5, 2, 8, 1, 3]

        assert insertion_sort(data) == [1, 2, 3, 5, 8]

    def test_empty_list(self):
        assert insertion_sort([]) == []

    def test_single_element(self):
        assert insertion_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert insertion_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [5, 4, 3, 2, 1]

        assert insertion_sort(data) == [1, 2, 3, 4, 5]

    def test_negative_values(self):
        data = [3, -1, -5, 2, 0]

        assert insertion_sort(data) == [-5, -1, 0, 2, 3]

    def test_duplicates(self):
        data = [4, 2, 4, 1, 2]

        assert insertion_sort(data) == [1, 2, 2, 4, 4]

    def test_does_not_modify_original_data(self):
        data = [5, 2, 3, 1]
        original = data.copy()

        insertion_sort(data)

        assert data == original

    def test_works_with_strings(self):
        data = ["python", "algorithm", "math"]

        assert insertion_sort(data) == ["algorithm", "math", "python"]


# ============================================================
# MERGE SORT
# ============================================================


class TestMergeSort:
    """Tests de l'algorithme Merge Sort."""

    def test_sorts_unsorted_list(self):
        data = [5, 2, 8, 1, 3]

        assert merge_sort(data) == [1, 2, 3, 5, 8]

    def test_empty_list(self):
        assert merge_sort([]) == []

    def test_single_element(self):
        assert merge_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert merge_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [5, 4, 3, 2, 1]

        assert merge_sort(data) == [1, 2, 3, 4, 5]

    def test_negative_values(self):
        data = [3, -1, -5, 2, 0]

        assert merge_sort(data) == [-5, -1, 0, 2, 3]

    def test_duplicates(self):
        data = [4, 2, 4, 1, 2]

        assert merge_sort(data) == [1, 2, 2, 4, 4]

    def test_does_not_modify_original_data(self):
        data = [5, 2, 3, 1]
        original = data.copy()

        merge_sort(data)

        assert data == original

    def test_works_with_strings(self):
        data = ["python", "algorithm", "math"]

        assert merge_sort(data) == ["algorithm", "math", "python"]


# ============================================================
# QUICK SORT
# ============================================================


class TestQuickSort:
    """Tests de l'algorithme Quick Sort."""

    def test_sorts_unsorted_list(self):
        data = [5, 2, 8, 1, 3]

        assert quick_sort(data) == [1, 2, 3, 5, 8]

    def test_empty_list(self):
        assert quick_sort([]) == []

    def test_single_element(self):
        assert quick_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert quick_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [5, 4, 3, 2, 1]

        assert quick_sort(data) == [1, 2, 3, 4, 5]

    def test_negative_values(self):
        data = [3, -1, -5, 2, 0]

        assert quick_sort(data) == [-5, -1, 0, 2, 3]

    def test_duplicates(self):
        data = [4, 2, 4, 1, 2]

        assert quick_sort(data) == [1, 2, 2, 4, 4]

    def test_does_not_modify_original_data(self):
        data = [5, 2, 3, 1]
        original = data.copy()

        quick_sort(data)

        assert data == original

    def test_works_with_strings(self):
        data = ["python", "algorithm", "math"]

        assert quick_sort(data) == ["algorithm", "math", "python"]


# ============================================================
# HEAP SORT
# ============================================================


class TestHeapSort:
    """Tests de l'algorithme Heap Sort."""

    def test_sorts_unsorted_list(self):
        data = [5, 2, 8, 1, 3]

        assert heap_sort(data) == [1, 2, 3, 5, 8]

    def test_empty_list(self):
        assert heap_sort([]) == []

    def test_single_element(self):
        assert heap_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert heap_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [5, 4, 3, 2, 1]

        assert heap_sort(data) == [1, 2, 3, 4, 5]

    def test_negative_values(self):
        data = [3, -1, -5, 2, 0]

        assert heap_sort(data) == [-5, -1, 0, 2, 3]

    def test_duplicates(self):
        data = [4, 2, 4, 1, 2]

        assert heap_sort(data) == [1, 2, 2, 4, 4]

    def test_does_not_modify_original_data(self):
        data = [5, 2, 3, 1]
        original = data.copy()

        heap_sort(data)

        assert data == original

    def test_works_with_strings(self):
        data = ["python", "algorithm", "math"]

        assert heap_sort(data) == ["algorithm", "math", "python"]


# ============================================================
# COUNTING SORT
# ============================================================


class TestCountingSort:
    """Tests de l'algorithme Counting Sort."""

    def test_sorts_unsorted_integers(self):
        data = [5, 2, 8, 1, 3]

        assert counting_sort(data) == [1, 2, 3, 5, 8]

    def test_empty_list(self):
        assert counting_sort([]) == []

    def test_single_element(self):
        assert counting_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert counting_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [5, 4, 3, 2, 1]

        assert counting_sort(data) == [1, 2, 3, 4, 5]

    def test_negative_values(self):
        data = [3, -1, -5, 2, 0]

        assert counting_sort(data) == [-5, -1, 0, 2, 3]

    def test_duplicates(self):
        data = [4, 2, 4, 1, 2]

        assert counting_sort(data) == [1, 2, 2, 4, 4]

    def test_does_not_modify_original_data(self):
        data = [5, 2, 3, 1]
        original = data.copy()

        counting_sort(data)

        assert data == original

    def test_supports_large_integer_values(self):
        data = [1000, 5, 500, 1]

        assert counting_sort(data) == [1, 5, 500, 1000]

    def test_rejects_floats(self):
        data = [3, 1.5, 2]

        with pytest.raises(TypeError, match="entiers"):
            counting_sort(data)

    def test_rejects_strings(self):
        data = ["3", "1", "2"]

        with pytest.raises(TypeError, match="entiers"):
            counting_sort(data)


# ============================================================
# RADIX SORT
# ============================================================


class TestRadixSort:
    """Tests de l'algorithme Radix Sort."""

    def test_sorts_unsorted_integers(self):
        data = [170, 45, 75, 90, 802, 24, 2, 66]

        assert radix_sort(data) == [2, 24, 45, 66, 75, 90, 170, 802]

    def test_empty_list(self):
        assert radix_sort([]) == []

    def test_single_element(self):
        assert radix_sort([42]) == [42]

    def test_already_sorted(self):
        data = [1, 2, 3, 4, 5]

        assert radix_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        data = [500, 400, 300, 200, 100]

        assert radix_sort(data) == [100, 200, 300, 400, 500]

    def test_negative_values(self):
        data = [-170, 45, -75, 90, -802, 24, -2, 66]

        assert radix_sort(data) == [
            -802,
            -170,
            -75,
            -2,
            24,
            45,
            66,
            90,
        ]

    def test_duplicates(self):
        data = [45, 12, 45, 3, 12]

        assert radix_sort(data) == [3, 12, 12, 45, 45]

    def test_does_not_modify_original_data(self):
        data = [170, 45, 75, 90]
        original = data.copy()

        radix_sort(data)

        assert data == original

    def test_supports_zero(self):
        data = [0, 10, 5, 0, 2]

        assert radix_sort(data) == [0, 0, 2, 5, 10]

    def test_rejects_floats(self):
        data = [3, 1.5, 2]

        with pytest.raises(TypeError, match="entiers"):
            radix_sort(data)

    def test_rejects_strings(self):
        data = ["3", "1", "2"]

        with pytest.raises(TypeError, match="entiers"):
            radix_sort(data)


# ============================================================
# BUCKET SORT
# ============================================================


class TestBucketSort:
    """Tests de l'algorithme Bucket Sort."""

    def test_sorts_unsorted_numbers(self):
        data = [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]

        assert bucket_sort(data) == sorted(data)

    def test_empty_list(self):
        assert bucket_sort([]) == []

    def test_single_element(self):
        assert bucket_sort([0.42]) == [0.42]

    def test_already_sorted(self):
        data = [0.1, 0.2, 0.3, 0.4, 0.5]

        assert bucket_sort(data) == data

    def test_reverse_sorted(self):
        data = [0.5, 0.4, 0.3, 0.2, 0.1]

        assert bucket_sort(data) == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_negative_values(self):
        data = [-0.5, 0.2, -0.1, 0.7, -0.8]

        assert bucket_sort(data) == sorted(data)

    def test_duplicates(self):
        data = [0.4, 0.2, 0.4, 0.1, 0.2]

        assert bucket_sort(data) == [0.1, 0.2, 0.2, 0.4, 0.4]

    def test_all_values_equal(self):
        data = [2.5, 2.5, 2.5]

        assert bucket_sort(data) == [2.5, 2.5, 2.5]

    def test_does_not_modify_original_data(self):
        data = [0.5, 0.1, 0.3]
        original = data.copy()

        bucket_sort(data)

        assert data == original

    def test_rejects_strings(self):
        data = ["3", "1", "2"]

        with pytest.raises(TypeError, match="numériques"):
            bucket_sort(data)

    def test_rejects_nan(self):
        data = [0.1, math.nan, 0.3]

        with pytest.raises(ValueError, match="finies"):
            bucket_sort(data)

    def test_rejects_infinity(self):
        data = [0.1, math.inf, 0.3]

        with pytest.raises(ValueError, match="finies"):
            bucket_sort(data)


# ============================================================
# TESTS COMPARATIFS
# ============================================================


class TestComparisonOfSortingAlgorithms:
    """
    Vérifie que les six algorithmes généraux produisent
    exactement le même résultat.
    """

    @pytest.mark.parametrize(
        "data",
        [
            [5, 2, 8, 1, 3],
            [10, 9, 8, 7, 6],
            [-5, 2, -1, 0, 4],
            [3, 3, 1, 2, 1],
            [],
            [42],
        ],
    )
    def test_general_sorting_algorithms_are_consistent(self, data):
        expected = sorted(data)

        for sorting_function in SORTING_FUNCTIONS:
            assert sorting_function(data) == expected

    def test_all_integer_sorting_algorithms_are_consistent(self):
        data = [170, 45, 75, 90, 802, 24, 2, 66, -5, 0]

        expected = sorted(data)

        assert counting_sort(data) == expected
        assert radix_sort(data) == expected

    def test_sorting_algorithms_do_not_modify_original_data(self):
        data = [9, 3, 7, 1, 5]
        original = data.copy()

        for sorting_function in SORTING_FUNCTIONS:
            sorting_function(data)

        assert data == original

    def test_large_dataset(self):
        data = list(range(1000, 0, -1))
        expected = list(range(1, 1001))

        for sorting_function in SORTING_FUNCTIONS:
            assert sorting_function(data) == expected

    def test_random_like_dataset(self):
        data = [
            42,
            7,
            19,
            3,
            88,
            12,
            56,
            1,
            34,
            21,
            5,
            73,
        ]

        expected = sorted(data)

        for sorting_function in SORTING_FUNCTIONS:
            assert sorting_function(data) == expected

