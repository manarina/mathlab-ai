
"""
Tests for String Algorithms
===========================

Tests unitaires pour :
- Naive String Search
- KMP (Knuth-Morris-Pratt)
- Rabin-Karp
- Z Algorithm
- Levenshtein Distance
"""

import pytest

from core.algorithms.string_algorithms import (
    kmp_search,
    levenshtein_distance,
    naive_string_search,
    rabin_karp_search,
    z_algorithm_search,
)


# ============================================================================
# HELPERS
# ============================================================================


SEARCH_FUNCTIONS = [
    naive_string_search,
    kmp_search,
    rabin_karp_search,
    z_algorithm_search,
]


# ============================================================================
# NAIVE STRING SEARCH
# ============================================================================


class TestNaiveStringSearch:
    """Tests for naive string search."""

    def test_single_occurrence(self):
        assert naive_string_search("hello world", "world") == [6]

    def test_occurrence_at_beginning(self):
        assert naive_string_search("hello world", "hello") == [0]

    def test_occurrence_at_end(self):
        assert naive_string_search("hello world", "world") == [6]

    def test_multiple_occurrences(self):
        assert naive_string_search("abcabcabc", "abc") == [0, 3, 6]

    def test_overlapping_occurrences(self):
        assert naive_string_search("aaaa", "aa") == [0, 1, 2]

    def test_single_character(self):
        assert naive_string_search("banana", "a") == [1, 3, 5]

    def test_pattern_not_found(self):
        assert naive_string_search("hello", "xyz") == []

    def test_pattern_longer_than_text(self):
        assert naive_string_search("abc", "abcdef") == []

    def test_empty_text(self):
        assert naive_string_search("", "abc") == []

    def test_empty_pattern(self):
        assert naive_string_search("abc", "") == []

    def test_identical_strings(self):
        assert naive_string_search("algorithm", "algorithm") == [0]

    def test_case_sensitive(self):
        assert naive_string_search("Hello", "hello") == []

    def test_case_matching(self):
        assert naive_string_search("Hello", "Hello") == [0]

    def test_spaces(self):
        assert naive_string_search("hello world hello", "hello") == [0, 12]

    def test_unicode(self):
        assert naive_string_search("été été", "été") == [0, 4]

    def test_special_characters(self):
        assert naive_string_search("a+b+c+a+b", "a+b") == [0, 6]

    def test_invalid_text(self):
        with pytest.raises(ValueError):
            naive_string_search(123, "abc")

    def test_invalid_pattern(self):
        with pytest.raises(ValueError):
            naive_string_search("abc", 123)


# ============================================================================
# KMP
# ============================================================================


class TestKMP:
    """Tests for Knuth-Morris-Pratt."""

    def test_single_occurrence(self):
        assert kmp_search("hello world", "world") == [6]

    def test_multiple_occurrences(self):
        assert kmp_search("abcabcabc", "abc") == [0, 3, 6]

    def test_overlapping_occurrences(self):
        assert kmp_search("aaaa", "aa") == [0, 1, 2]

    def test_complex_overlap(self):
        assert kmp_search("abababab", "abab") == [0, 2, 4]

    def test_pattern_not_found(self):
        assert kmp_search("hello", "xyz") == []

    def test_pattern_longer_than_text(self):
        assert kmp_search("abc", "abcdef") == []

    def test_empty_text(self):
        assert kmp_search("", "abc") == []

    def test_empty_pattern(self):
        assert kmp_search("abc", "") == []

    def test_identical_strings(self):
        assert kmp_search("algorithm", "algorithm") == [0]

    def test_single_character(self):
        assert kmp_search("banana", "a") == [1, 3, 5]

    def test_repeated_pattern(self):
        assert kmp_search("aaaaaa", "aaa") == [0, 1, 2, 3]

    def test_unicode(self):
        assert kmp_search("été été", "été") == [0, 4]

    def test_spaces(self):
        assert kmp_search("hello world hello", "world") == [6]

    def test_case_sensitive(self):
        assert kmp_search("Hello", "hello") == []

    def test_invalid_text(self):
        with pytest.raises(ValueError):
            kmp_search(123, "abc")

    def test_invalid_pattern(self):
        with pytest.raises(ValueError):
            kmp_search("abc", 123)


# ============================================================================
# RABIN-KARP
# ============================================================================


class TestRabinKarp:
    """Tests for Rabin-Karp string search."""

    def test_single_occurrence(self):
        assert rabin_karp_search("hello world", "world") == [6]

    def test_multiple_occurrences(self):
        assert rabin_karp_search("abcabcabc", "abc") == [0, 3, 6]

    def test_overlapping_occurrences(self):
        assert rabin_karp_search("aaaa", "aa") == [0, 1, 2]

    def test_complex_overlap(self):
        assert rabin_karp_search("abababab", "abab") == [0, 2, 4]

    def test_pattern_not_found(self):
        assert rabin_karp_search("hello", "xyz") == []

    def test_pattern_longer_than_text(self):
        assert rabin_karp_search("abc", "abcdef") == []

    def test_empty_text(self):
        assert rabin_karp_search("", "abc") == []

    def test_empty_pattern(self):
        assert rabin_karp_search("abc", "") == []

    def test_identical_strings(self):
        assert rabin_karp_search("algorithm", "algorithm") == [0]

    def test_single_character(self):
        assert rabin_karp_search("banana", "a") == [1, 3, 5]

    def test_repeated_pattern(self):
        assert rabin_karp_search("aaaaaa", "aaa") == [0, 1, 2, 3]

    def test_unicode(self):
        assert rabin_karp_search("été été", "été") == [0, 4]

    def test_spaces(self):
        assert rabin_karp_search("hello world hello", "world") == [6]

    def test_case_sensitive(self):
        assert rabin_karp_search("Hello", "hello") == []

    def test_invalid_text(self):
        with pytest.raises(ValueError):
            rabin_karp_search(123, "abc")

    def test_invalid_pattern(self):
        with pytest.raises(ValueError):
            rabin_karp_search("abc", 123)


# ============================================================================
# Z ALGORITHM
# ============================================================================


class TestZAlgorithm:
    """Tests for Z algorithm string search."""

    def test_single_occurrence(self):
        assert z_algorithm_search("hello world", "world") == [6]

    def test_multiple_occurrences(self):
        assert z_algorithm_search("abcabcabc", "abc") == [0, 3, 6]

    def test_overlapping_occurrences(self):
        assert z_algorithm_search("aaaa", "aa") == [0, 1, 2]

    def test_complex_overlap(self):
        assert z_algorithm_search("abababab", "abab") == [0, 2, 4]

    def test_pattern_not_found(self):
        assert z_algorithm_search("hello", "xyz") == []

    def test_pattern_longer_than_text(self):
        assert z_algorithm_search("abc", "abcdef") == []

    def test_empty_text(self):
        assert z_algorithm_search("", "abc") == []

    def test_empty_pattern(self):
        assert z_algorithm_search("abc", "") == []

    def test_identical_strings(self):
        assert z_algorithm_search("algorithm", "algorithm") == [0]

    def test_single_character(self):
        assert z_algorithm_search("banana", "a") == [1, 3, 5]

    def test_repeated_pattern(self):
        assert z_algorithm_search("aaaaaa", "aaa") == [0, 1, 2, 3]

    def test_unicode(self):
        assert z_algorithm_search("été été", "été") == [0, 4]

    def test_spaces(self):
        assert z_algorithm_search("hello world hello", "world") == [6]

    def test_case_sensitive(self):
        assert z_algorithm_search("Hello", "hello") == []

    def test_invalid_text(self):
        with pytest.raises(ValueError):
            z_algorithm_search(123, "abc")

    def test_invalid_pattern(self):
        with pytest.raises(ValueError):
            z_algorithm_search("abc", 123)


# ============================================================================
# LEVENSHTEIN DISTANCE
# ============================================================================


class TestLevenshteinDistance:
    """Tests for Levenshtein distance."""

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_empty_first(self):
        assert levenshtein_distance("", "abc") == 3

    def test_empty_second(self):
        assert levenshtein_distance("abc", "") == 3

    def test_kitten_sitting(self):
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_flaw_lawn(self):
        assert levenshtein_distance("flaw", "lawn") == 2

    def test_single_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("cats", "cat") == 1

    def test_single_substitution(self):
        assert levenshtein_distance("cat", "bat") == 1

    def test_multiple_operations(self):
        assert levenshtein_distance("Saturday", "Sunday") == 3

    def test_single_character_same(self):
        assert levenshtein_distance("a", "a") == 0

    def test_single_character_different(self):
        assert levenshtein_distance("a", "b") == 1

    def test_unicode(self):
        assert levenshtein_distance("été", "ete") == 2

    def test_spaces(self):
        assert levenshtein_distance("hello world", "hello") == 6

    def test_case_sensitive(self):
        assert levenshtein_distance("Hello", "hello") == 1

    def test_symmetry(self):
        first = "algorithm"
        second = "algorithms"

        assert (
            levenshtein_distance(first, second)
            == levenshtein_distance(second, first)
        )

    def test_distance_upper_bound(self):
        first = "abc"
        second = "xyz"

        distance = levenshtein_distance(first, second)

        assert distance <= max(len(first), len(second))

    def test_invalid_first(self):
        with pytest.raises(ValueError):
            levenshtein_distance(123, "abc")

    def test_invalid_second(self):
        with pytest.raises(ValueError):
            levenshtein_distance("abc", 123)


# ============================================================================
# CROSS-ALGORITHM CONSISTENCY
# ============================================================================


class TestSearchAlgorithmConsistency:
    """Ensure all search algorithms return the same results."""

    @pytest.mark.parametrize(
        "text,pattern,expected",
        [
            ("abcabcabc", "abc", [0, 3, 6]),
            ("aaaa", "aa", [0, 1, 2]),
            ("abababab", "abab", [0, 2, 4]),
            ("banana", "ana", [1, 3]),
            ("mississippi", "issi", [1, 4]),
            ("hello world", "world", [6]),
            ("python programming", "program", [7]),
            ("abcdef", "xyz", []),
            ("abcdef", "abcdef", [0]),
            ("abcdef", "abcdefg", []),
            ("", "abc", []),
            ("abc", "", []),
            ("été été", "été", [0, 4]),
        ],
    )
    def test_all_search_algorithms(
        self,
        text,
        pattern,
        expected,
    ):
        for search_function in SEARCH_FUNCTIONS:
            assert search_function(text, pattern) == expected

    def test_long_repeated_text(self):
        text = "abc" * 1000
        pattern = "abc"

        expected = list(range(0, len(text), 3))

        for search_function in SEARCH_FUNCTIONS:
            assert search_function(text, pattern) == expected

    def test_long_overlapping_text(self):
        text = "a" * 1000
        pattern = "aaa"

        expected = list(range(998))

        for search_function in SEARCH_FUNCTIONS:
            assert search_function(text, pattern) == expected


# ============================================================================
# LEVENSHTEIN PROPERTIES
# ============================================================================


class TestLevenshteinProperties:
    """Property-based style tests for Levenshtein distance."""

    @pytest.mark.parametrize(
        "first,second",
        [
            ("", ""),
            ("a", "a"),
            ("abc", "abc"),
            ("hello", "hello"),
            ("algorithm", "algorithm"),
            ("été", "été"),
        ],
    )
    def test_identity(self, first, second):
        assert levenshtein_distance(first, second) == 0

    @pytest.mark.parametrize(
        "first,second",
        [
            ("abc", "xyz"),
            ("hello", "world"),
            ("cat", "dog"),
            ("python", "java"),
        ],
    )
    def test_positive_for_different_strings(self, first, second):
        assert levenshtein_distance(first, second) > 0

    @pytest.mark.parametrize(
        "first,second",
        [
            ("abc", "abcd"),
            ("hello", "hell"),
            ("cat", "bat"),
            ("python", "pythons"),
        ],
    )
    def test_distance_one(self, first, second):
        assert levenshtein_distance(first, second) == 1

    def test_symmetry_multiple_cases(self):
        cases = [
            ("abc", "xyz"),
            ("kitten", "sitting"),
            ("hello", "world"),
            ("algorithm", "algorithms"),
            ("été", "ete"),
        ]

        for first, second in cases:
            assert levenshtein_distance(
                first,
                second,
            ) == levenshtein_distance(
                second,
                first,
            )

