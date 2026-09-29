
"""
Tests for MathLab AI — Algorithm Explanations
=============================================

Validation du module :
    core.algorithms.algorithm_explanations
"""

import pytest

from core.algorithms.algorithm_explanations import (
    format_complexity,
    get_algorithm_complexity,
    get_algorithm_description,
    get_algorithm_example,
    get_algorithm_explanation,
    get_algorithm_idea,
    get_algorithm_steps,
    get_algorithms_by_category,
    get_category_summary,
    get_full_explanation,
    list_algorithm_categories,
    list_explained_algorithms,
    search_algorithm_explanations,
)


# ============================================================================
# CONSTANTES DE TEST
# ============================================================================

EXPECTED_CATEGORIES = {
    "Searching",
    "Sorting",
    "Graphs",
    "Dynamic Programming",
    "Greedy",
    "Backtracking",
    "Trees",
    "Mathematical",
    "Strings",
}

EXPECTED_ALGORITHMS = {
    # Searching
    "linear_search",
    "binary_search",
    "jump_search",
    "interpolation_search",
    "exponential_search",
    # Sorting
    "bubble_sort",
    "selection_sort",
    "insertion_sort",
    "merge_sort",
    "quick_sort",
    "heap_sort",
    "counting_sort",
    "radix_sort",
    "bucket_sort",
    # Graphs
    "bfs",
    "dfs",
    "dijkstra",
    "bellman_ford",
    "floyd_warshall",
    "kruskal",
    "prim",
    # Dynamic Programming
    "fibonacci",
    "climbing_stairs",
    "knapsack_01",
    "coin_change",
    "longest_common_subsequence",
    "longest_increasing_subsequence",
    "matrix_chain_multiplication",
    # Greedy
    "activity_selection",
    "fractional_knapsack",
    "greedy_coin_change",
    "huffman_coding",
    "job_sequencing",
    "interval_scheduling",
    # Backtracking
    "n_queens",
    "solve_sudoku",
    "solve_maze",
    "subsets",
    "permutations",
    "combination_sum",
    # Trees
    "preorder_traversal",
    "inorder_traversal",
    "postorder_traversal",
    "level_order_traversal",
    "bst_search",
    "bst_insert",
    "bst_delete",
    "min_heap",
    "priority_queue",
    # Mathematical
    "gcd",
    "extended_gcd",
    "generate_primes",
    "sieve_of_eratosthenes",
    "fast_power",
    "modular_power",
    "factorial",
    "pascal_triangle",
    # Strings
    "naive_search",
    "kmp_search",
    "rabin_karp_search",
    "z_search",
    "levenshtein_distance",
}


# ============================================================================
# TESTS — LISTES ET CATÉGORIES
# ============================================================================


class TestListsAndCategories:
    """Tests des fonctions de découverte."""

    def test_list_explained_algorithms_returns_list(self):
        result = list_explained_algorithms()

        assert isinstance(result, list)
        assert result

    def test_list_explained_algorithms_contains_all_expected_algorithms(self):
        result = set(list_explained_algorithms())

        assert result == EXPECTED_ALGORITHMS

    def test_list_explained_algorithms_contains_no_duplicates(self):
        result = list_explained_algorithms()

        assert len(result) == len(set(result))

    def test_list_algorithm_categories_returns_list(self):
        result = list_algorithm_categories()

        assert isinstance(result, list)
        assert result

    def test_list_algorithm_categories_contains_expected_categories(self):
        result = set(list_algorithm_categories())

        assert result == EXPECTED_CATEGORIES

    def test_list_algorithm_categories_contains_no_duplicates(self):
        result = list_algorithm_categories()

        assert len(result) == len(set(result))


# ============================================================================
# TESTS — EXPLICATION COMPLÈTE
# ============================================================================


class TestGetAlgorithmExplanation:
    """Tests de get_algorithm_explanation."""

    def test_returns_dictionary(self):
        result = get_algorithm_explanation("binary_search")

        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        result = get_algorithm_explanation("binary_search")

        expected_keys = {
            "name",
            "category",
            "description",
            "idea",
            "steps",
            "complexity",
            "advantages",
            "limitations",
            "example",
        }

        assert set(result.keys()) == expected_keys

    @pytest.mark.parametrize(
        "algorithm",
        sorted(EXPECTED_ALGORITHMS),
    )
    def test_all_algorithms_have_complete_explanation(self, algorithm):
        result = get_algorithm_explanation(algorithm)

        assert result["name"]
        assert result["category"]
        assert result["description"]
        assert result["idea"]
        assert result["steps"]
        assert result["complexity"]
        assert result["advantages"]
        assert result["limitations"]
        assert result["example"]

    def test_steps_are_list(self):
        result = get_algorithm_explanation("merge_sort")

        assert isinstance(result["steps"], list)
        assert all(isinstance(step, str) for step in result["steps"])

    def test_advantages_are_list(self):
        result = get_algorithm_explanation("dijkstra")

        assert isinstance(result["advantages"], list)
        assert all(isinstance(item, str) for item in result["advantages"])

    def test_limitations_are_list(self):
        result = get_algorithm_explanation("quick_sort")

        assert isinstance(result["limitations"], list)
        assert all(isinstance(item, str) for item in result["limitations"])

    def test_example_is_dictionary(self):
        result = get_algorithm_explanation("linear_search")

        assert isinstance(result["example"], dict)
        assert "input" in result["example"]
        assert "result" in result["example"]

    def test_unknown_algorithm_raises_value_error(self):
        with pytest.raises(ValueError):
            get_algorithm_explanation("unknown_algorithm")

    def test_empty_algorithm_raises_value_error(self):
        with pytest.raises(ValueError):
            get_algorithm_explanation("")

    def test_whitespace_algorithm_raises_value_error(self):
        with pytest.raises(ValueError):
            get_algorithm_explanation("   ")

    @pytest.mark.parametrize(
        "invalid_value",
        [
            None,
            123,
            3.14,
            [],
            {},
        ],
    )
    def test_invalid_algorithm_type_raises_value_error(self, invalid_value):
        with pytest.raises(ValueError):
            get_algorithm_explanation(invalid_value)


# ============================================================================
# TESTS — DESCRIPTION
# ============================================================================


class TestAlgorithmDescription:
    """Tests de get_algorithm_description."""

    @pytest.mark.parametrize(
        "algorithm",
        [
            "linear_search",
            "binary_search",
            "merge_sort",
            "dijkstra",
            "fibonacci",
            "n_queens",
            "gcd",
            "kmp_search",
        ],
    )
    def test_returns_non_empty_string(self, algorithm):
        result = get_algorithm_description(algorithm)

        assert isinstance(result, str)
        assert result.strip()

    def test_binary_search_description_mentions_sorted_data(self):
        result = get_algorithm_description("binary_search").lower()

        assert "tri" in result

    def test_dijkstra_description_mentions_shortest_paths(self):
        result = get_algorithm_description("dijkstra").lower()

        assert "chemin" in result

    def test_levenshtein_description_mentions_transformations(self):
        result = get_algorithm_description("levenshtein_distance").lower()

        assert "insertion" in result
        assert "suppression" in result


# ============================================================================
# TESTS — IDÉE PRINCIPALE
# ============================================================================


class TestAlgorithmIdea:
    """Tests de get_algorithm_idea."""

    @pytest.mark.parametrize(
        "algorithm",
        sorted(EXPECTED_ALGORITHMS),
    )
    def test_returns_non_empty_string(self, algorithm):
        result = get_algorithm_idea(algorithm)

        assert isinstance(result, str)
        assert result.strip()

    def test_binary_search_idea_mentions_division(self):
        result = get_algorithm_idea("binary_search").lower()

        assert "moitié" in result

    def test_gcd_idea_mentions_modulo(self):
        result = get_algorithm_idea("gcd").lower()

        assert "mod" in result


# ============================================================================
# TESTS — ÉTAPES
# ============================================================================


class TestAlgorithmSteps:
    """Tests de get_algorithm_steps."""

    @pytest.mark.parametrize(
        "algorithm",
        sorted(EXPECTED_ALGORITHMS),
    )
    def test_returns_non_empty_list(self, algorithm):
        result = get_algorithm_steps(algorithm)

        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.parametrize(
        "algorithm",
        [
            "linear_search",
            "binary_search",
            "bubble_sort",
            "merge_sort",
            "dijkstra",
            "fibonacci",
            "n_queens",
            "gcd",
            "kmp_search",
        ],
    )
    def test_each_step_is_string(self, algorithm):
        result = get_algorithm_steps(algorithm)

        assert all(isinstance(step, str) for step in result)
        assert all(step.strip() for step in result)


# ============================================================================
# TESTS — COMPLEXITÉ
# ============================================================================


class TestAlgorithmComplexity:
    """Tests de get_algorithm_complexity."""

    def test_returns_dictionary(self):
        result = get_algorithm_complexity("binary_search")

        assert isinstance(result, dict)

    def test_contains_required_complexity_keys(self):
        result = get_algorithm_complexity("binary_search")

        assert set(result.keys()) == {
            "best",
            "average",
            "worst",
            "space",
        }

    @pytest.mark.parametrize(
        "algorithm",
        sorted(EXPECTED_ALGORITHMS),
    )
    def test_all_algorithms_have_complete_complexity(self, algorithm):
        result = get_algorithm_complexity(algorithm)

        assert set(result.keys()) == {
            "best",
            "average",
            "worst",
            "space",
        }

        for value in result.values():
            assert isinstance(value, str)
            assert value.strip()

    def test_linear_search_complexity(self):
        result = get_algorithm_complexity("linear_search")

        assert result == {
            "best": "O(1)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(1)",
        }

    def test_binary_search_complexity(self):
        result = get_algorithm_complexity("binary_search")

        assert result == {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        }

    def test_merge_sort_complexity(self):
        result = get_algorithm_complexity("merge_sort")

        assert result == {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)",
        }

    def test_quick_sort_complexity(self):
        result = get_algorithm_complexity("quick_sort")

        assert result == {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n²)",
            "space": "O(log n)",
        }

    def test_kmp_complexity(self):
        result = get_algorithm_complexity("kmp_search")

        assert result == {
            "best": "O(n + m)",
            "average": "O(n + m)",
            "worst": "O(n + m)",
            "space": "O(m)",
        }

    def test_levenshtein_complexity(self):
        result = get_algorithm_complexity("levenshtein_distance")

        assert result == {
            "best": "O(nm)",
            "average": "O(nm)",
            "worst": "O(nm)",
            "space": "O(min(n, m))",
        }


# ============================================================================
# TESTS — EXEMPLES
# ============================================================================


class TestAlgorithmExample:
    """Tests de get_algorithm_example."""

    @pytest.mark.parametrize(
        "algorithm",
        sorted(EXPECTED_ALGORITHMS),
    )
    def test_returns_dictionary(self, algorithm):
        result = get_algorithm_example(algorithm)

        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "algorithm",
        sorted(EXPECTED_ALGORITHMS),
    )
    def test_example_contains_input_or_result(self, algorithm):
        result = get_algorithm_example(algorithm)

        assert "input" in result
        assert "result" in result

    def test_binary_search_example(self):
        result = get_algorithm_example("binary_search")

        assert result["input"]
        assert result["target"]
        assert result["result"]

    def test_fibonacci_example(self):
        result = get_algorithm_example("fibonacci")

        assert "n = 7" in result["input"]
        assert result["result"] == "13"

    def test_factorial_example(self):
        result = get_algorithm_example("factorial")

        assert result["input"] == "5!"
        assert result["result"] == "120"


# ============================================================================
# TESTS — ALGORITHMES PAR CATÉGORIE
# ============================================================================


class TestAlgorithmsByCategory:
    """Tests de get_algorithms_by_category."""

    def test_searching_category(self):
        result = get_algorithms_by_category("Searching")

        assert set(result) == {
            "linear_search",
            "binary_search",
            "jump_search",
            "interpolation_search",
            "exponential_search",
        }

    def test_sorting_category(self):
        result = get_algorithms_by_category("Sorting")

        assert set(result) == {
            "bubble_sort",
            "selection_sort",
            "insertion_sort",
            "merge_sort",
            "quick_sort",
            "heap_sort",
            "counting_sort",
            "radix_sort",
            "bucket_sort",
        }

    def test_graph_category(self):
        result = get_algorithms_by_category("Graphs")

        assert set(result) == {
            "bfs",
            "dfs",
            "dijkstra",
            "bellman_ford",
            "floyd_warshall",
            "kruskal",
            "prim",
        }

    def test_dynamic_programming_category(self):
        result = get_algorithms_by_category("Dynamic Programming")

        assert set(result) == {
            "fibonacci",
            "climbing_stairs",
            "knapsack_01",
            "coin_change",
            "longest_common_subsequence",
            "longest_increasing_subsequence",
            "matrix_chain_multiplication",
        }

    def test_greedy_category(self):
        result = get_algorithms_by_category("Greedy")

        assert set(result) == {
            "activity_selection",
            "fractional_knapsack",
            "greedy_coin_change",
            "huffman_coding",
            "job_sequencing",
            "interval_scheduling",
        }

    def test_backtracking_category(self):
        result = get_algorithms_by_category("Backtracking")

        assert set(result) == {
            "n_queens",
            "solve_sudoku",
            "solve_maze",
            "subsets",
            "permutations",
            "combination_sum",
        }

    def test_trees_category(self):
        result = get_algorithms_by_category("Trees")

        assert set(result) == {
            "preorder_traversal",
            "inorder_traversal",
            "postorder_traversal",
            "level_order_traversal",
            "bst_search",
            "bst_insert",
            "bst_delete",
            "min_heap",
            "priority_queue",
        }

    def test_mathematical_category(self):
        result = get_algorithms_by_category("Mathematical")

        assert set(result) == {
            "gcd",
            "extended_gcd",
            "generate_primes",
            "sieve_of_eratosthenes",
            "fast_power",
            "modular_power",
            "factorial",
            "pascal_triangle",
        }

    def test_strings_category(self):
        result = get_algorithms_by_category("Strings")

        assert set(result) == {
            "naive_search",
            "kmp_search",
            "rabin_karp_search",
            "z_search",
            "levenshtein_distance",
        }

    @pytest.mark.parametrize(
        "category",
        [
            "searching",
            "SEARCHING",
            "Searching",
            " searching ",
        ],
    )
    def test_category_search_is_case_insensitive(self, category):
        result = get_algorithms_by_category(category)

        assert set(result) == {
            "linear_search",
            "binary_search",
            "jump_search",
            "interpolation_search",
            "exponential_search",
        }

    def test_unknown_category_returns_empty_list(self):
        result = get_algorithms_by_category("Unknown Category")

        assert result == []

    @pytest.mark.parametrize(
        "invalid_value",
        [
            None,
            123,
            3.14,
            [],
            {},
        ],
    )
    def test_invalid_category_type_raises_value_error(self, invalid_value):
        with pytest.raises(ValueError):
            get_algorithms_by_category(invalid_value)

    def test_empty_category_raises_value_error(self):
        with pytest.raises(ValueError):
            get_algorithms_by_category("")

    def test_whitespace_category_raises_value_error(self):
        with pytest.raises(ValueError):
            get_algorithms_by_category("   ")


# ============================================================================
# TESTS — RÉSUMÉ DE CATÉGORIE
# ============================================================================


class TestCategorySummary:
    """Tests de get_category_summary."""

    def test_searching_summary(self):
        result = get_category_summary("Searching")

        assert isinstance(result, dict)
        assert result["category"] == "Searching"
        assert result["count"] == 5
        assert set(result["algorithms"]) == {
            "linear_search",
            "binary_search",
            "jump_search",
            "interpolation_search",
            "exponential_search",
        }

    @pytest.mark.parametrize(
        "category",
        sorted(EXPECTED_CATEGORIES),
    )
    def test_every_category_has_valid_summary(self, category):
        result = get_category_summary(category)

        assert result["category"] == category
        assert result["count"] > 0
        assert len(result["algorithms"]) == result["count"]

    def test_unknown_category_raises_value_error(self):
        with pytest.raises(ValueError):
            get_category_summary("Unknown Category")

    def test_case_insensitive_category(self):
        result = get_category_summary("searching")

        assert result["category"] == "Searching"

    def test_invalid_category_type_raises_value_error(self):
        with pytest.raises(ValueError):
            get_category_summary(None)


# ============================================================================
# TESTS — RECHERCHE
# ============================================================================


class TestSearchAlgorithmExplanations:
    """Tests de search_algorithm_explanations."""

    def test_search_by_algorithm_name(self):
        result = search_algorithm_explanations("binary_search")

        assert "binary_search" in result

    def test_search_by_display_name(self):
        result = search_algorithm_explanations("Binary Search")

        assert "binary_search" in result

    def test_search_is_case_insensitive(self):
        result = search_algorithm_explanations("BINARY SEARCH")

        assert "binary_search" in result

    def test_search_by_category(self):
        result = search_algorithm_explanations("Searching")

        assert set(result) == {
            "linear_search",
            "binary_search",
            "jump_search",
            "interpolation_search",
            "exponential_search",
        }

    def test_search_by_description_keyword(self):
        result = search_algorithm_explanations("triée")

        assert result

        assert any(
            algorithm in result
            for algorithm in [
                "binary_search",
                "jump_search",
                "interpolation_search",
                "exponential_search",
            ]
        )

    def test_search_by_idea_keyword(self):
        result = search_algorithm_explanations("pivot")

        assert "quick_sort" in result

    def test_search_unknown_keyword_returns_empty_list(self):
        result = search_algorithm_explanations(
            "mot_cle_totalement_inexistant"
        )

        assert result == []

    def test_search_does_not_duplicate_results(self):
        result = search_algorithm_explanations("search")

        assert len(result) == len(set(result))

    def test_search_result_contains_only_known_algorithms(self):
        result = search_algorithm_explanations("algorithm")

        assert set(result).issubset(EXPECTED_ALGORITHMS)

    @pytest.mark.parametrize(
        "invalid_value",
        [
            None,
            123,
            3.14,
            [],
            {},
        ],
    )
    def test_invalid_keyword_type_raises_value_error(self, invalid_value):
        with pytest.raises(ValueError):
            search_algorithm_explanations(invalid_value)

    def test_empty_keyword_raises_value_error(self):
        with pytest.raises(ValueError):
            search_algorithm_explanations("")

    def test_whitespace_keyword_raises_value_error(self):
        with pytest.raises(ValueError):
            search_algorithm_explanations("   ")


# ============================================================================
# TESTS — FORMATAGE DE LA COMPLEXITÉ
# ============================================================================


class TestFormatComplexity:
    """Tests de format_complexity."""

    def test_returns_string(self):
        result = format_complexity("binary_search")

        assert isinstance(result, str)

    def test_contains_all_complexity_labels(self):
        result = format_complexity("binary_search")

        assert "Meilleur cas" in result
        assert "Cas moyen" in result
        assert "Pire cas" in result
        assert "Espace" in result

    def test_contains_binary_search_complexities(self):
        result = format_complexity("binary_search")

        assert "O(1)" in result
        assert "O(log n)" in result

    def test_contains_merge_sort_complexity(self):
        result = format_complexity("merge_sort")

        assert "O(n log n)" in result
        assert "O(n)" in result

    def test_unknown_algorithm_raises_value_error(self):
        with pytest.raises(ValueError):
            format_complexity("unknown_algorithm")


# ============================================================================
# TESTS — EXPLICATION TEXTUELLE COMPLÈTE
# ============================================================================


class TestFullExplanation:
    """Tests de get_full_explanation."""

    def test_returns_string(self):
        result = get_full_explanation("binary_search")

        assert isinstance(result, str)

    def test_contains_algorithm_name(self):
        result = get_full_explanation("binary_search")

        assert "Binary Search" in result

    def test_contains_category(self):
        result = get_full_explanation("binary_search")

        assert "Searching" in result

    def test_contains_main_sections(self):
        result = get_full_explanation("binary_search")

        assert "## Description" in result
        assert "## Idée principale" in result
        assert "## Étapes" in result
        assert "## Complexité" in result
        assert "## Avantages" in result
        assert "## Limitations" in result
        assert "## Exemple" in result

    def test_contains_steps(self):
        result = get_full_explanation("linear_search")

        assert "Commencer au premier élément." in result
        assert "retourner -1" in result

    def test_contains_complexity(self):
        result = get_full_explanation("quick_sort")

        assert "O(n log n)" in result
        assert "O(n²)" in result
        assert "O(log n)" in result

    def test_contains_example(self):
        result = get_full_explanation("factorial")

        assert "5!" in result
        assert "120" in result

    def test_unknown_algorithm_raises_value_error(self):
        with pytest.raises(ValueError):
            get_full_explanation("unknown_algorithm")


# ============================================================================
# TESTS — PROTECTION DES DONNÉES INTERNES
# ============================================================================


class TestDataProtection:
    """
    Vérifie que les résultats retournés ne permettent pas de modifier
    directement les structures internes du module.
    """

    def test_modifying_steps_does_not_modify_source(self):
        result = get_algorithm_explanation("binary_search")

        original_steps = list(result["steps"])

        result["steps"].append("Modification externe")

        fresh_result = get_algorithm_explanation("binary_search")

        assert fresh_result["steps"] == original_steps

    def test_modifying_advantages_does_not_modify_source(self):
        result = get_algorithm_explanation("binary_search")

        original_advantages = list(result["advantages"])

        result["advantages"].append("Modification externe")

        fresh_result = get_algorithm_explanation("binary_search")

        assert fresh_result["advantages"] == original_advantages

    def test_modifying_limitations_does_not_modify_source(self):
        result = get_algorithm_explanation("binary_search")

        original_limitations = list(result["limitations"])

        result["limitations"].append("Modification externe")

        fresh_result = get_algorithm_explanation("binary_search")

        assert fresh_result["limitations"] == original_limitations

    def test_modifying_complexity_does_not_modify_source(self):
        result = get_algorithm_explanation("binary_search")

        result["complexity"]["best"] = "INVALID"

        fresh_result = get_algorithm_explanation("binary_search")

        assert fresh_result["complexity"]["best"] == "O(1)"

    def test_modifying_example_does_not_modify_source(self):
        result = get_algorithm_explanation("binary_search")

        result["example"]["input"] = "INVALID"

        fresh_result = get_algorithm_explanation("binary_search")

        assert fresh_result["example"]["input"] != "INVALID"


# ============================================================================
# TESTS — COHÉRENCE GLOBALE
# ============================================================================


class TestGlobalConsistency:
    """Tests de cohérence de l'ensemble de la base d'explications."""

    def test_all_explained_algorithms_have_known_categories(self):
        categories = set(list_algorithm_categories())

        for algorithm in list_explained_algorithms():
            explanation = get_algorithm_explanation(algorithm)

            assert explanation["category"] in categories

    def test_category_counts_sum_to_algorithm_count(self):
        total = 0

        for category in list_algorithm_categories():
            total += get_category_summary(category)["count"]

        assert total == len(list_explained_algorithms())

    def test_every_algorithm_can_be_searched_by_its_name(self):
        for algorithm in list_explained_algorithms():
            result = search_algorithm_explanations(algorithm)

            assert algorithm in result

    def test_every_algorithm_has_complexity(self):
        for algorithm in list_explained_algorithms():
            complexity = get_algorithm_complexity(algorithm)

            assert all(
                isinstance(value, str)
                for value in complexity.values()
            )

    def test_every_algorithm_has_at_least_one_step(self):
        for algorithm in list_explained_algorithms():
            steps = get_algorithm_steps(algorithm)

            assert len(steps) >= 1

    def test_every_algorithm_has_description_and_idea(self):
        for algorithm in list_explained_algorithms():
            description = get_algorithm_description(algorithm)
            idea = get_algorithm_idea(algorithm)

            assert description.strip()
            assert idea.strip()

