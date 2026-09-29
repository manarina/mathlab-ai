
"""
Tests for Backtracking Algorithms
=================================

Covered algorithms:
- N-Queens
- Sudoku Solver
- Maze Solver
- Subsets
- Permutations
- Combination Sum
"""

from copy import deepcopy

import pytest

from core.algorithms.backtracking import (
    combination_sum,
    n_queens,
    permutations,
    solve_maze,
    solve_sudoku,
    subsets,
)


# ============================================================================
# N-Queens
# ============================================================================


class TestNQueens:
    """Tests for the N-Queens solver."""

    def test_n_queens_n1(self):
        assert n_queens(1) == [[0]]

    def test_n_queens_n2_has_no_solution(self):
        assert n_queens(2) == []

    def test_n_queens_n3_has_no_solution(self):
        assert n_queens(3) == []

    def test_n_queens_n4_has_two_solutions(self):
        result = n_queens(4)

        expected = [
            [1, 3, 0, 2],
            [2, 0, 3, 1],
        ]

        assert result == expected

    def test_n_queens_n5_has_ten_solutions(self):
        result = n_queens(5)

        assert len(result) == 10

    def test_n_queens_solution_has_correct_length(self):
        result = n_queens(6)

        assert all(len(solution) == 6 for solution in result)

    def test_n_queens_solutions_use_each_column_once(self):
        result = n_queens(6)

        for solution in result:
            assert sorted(solution) == list(range(6))

    def test_n_queens_solutions_are_valid(self):
        result = n_queens(6)

        for solution in result:
            for row_1 in range(6):
                for row_2 in range(row_1 + 1, 6):
                    column_1 = solution[row_1]
                    column_2 = solution[row_2]

                    assert column_1 != column_2
                    assert abs(row_1 - row_2) != abs(column_1 - column_2)

    def test_n_queens_returns_new_lists(self):
        first = n_queens(4)
        second = n_queens(4)

        assert first == second
        assert first is not second

    @pytest.mark.parametrize("invalid_n", [0, -1, -5, 2.5, "4", None, True])
    def test_n_queens_rejects_invalid_n(self, invalid_n):
        with pytest.raises(ValueError):
            n_queens(invalid_n)


# ============================================================================
# Sudoku
# ============================================================================


class TestSudoku:
    """Tests for the Sudoku solver."""

    @staticmethod
    def valid_board():
        return [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

    @staticmethod
    def solved_board():
        return [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]

    def test_sudoku_solves_valid_board(self):
        board = self.valid_board()

        result = solve_sudoku(board)

        assert result == self.solved_board()

    def test_sudoku_solution_contains_numbers_1_to_9(self):
        result = solve_sudoku(self.valid_board())

        expected = set(range(1, 10))

        for row in result:
            assert set(row) == expected

    def test_sudoku_solution_has_valid_columns(self):
        result = solve_sudoku(self.valid_board())

        for column in range(9):
            values = {result[row][column] for row in range(9)}
            assert values == set(range(1, 10))

    def test_sudoku_solution_has_valid_boxes(self):
        result = solve_sudoku(self.valid_board())

        for box_row in range(0, 9, 3):
            for box_column in range(0, 9, 3):
                values = set()

                for row in range(box_row, box_row + 3):
                    for column in range(box_column, box_column + 3):
                        values.add(result[row][column])

                assert values == set(range(1, 10))

    def test_sudoku_does_not_modify_input(self):
        board = self.valid_board()
        original = deepcopy(board)

        solve_sudoku(board)

        assert board == original

    def test_sudoku_accepts_already_solved_board(self):
        board = self.solved_board()
        original = deepcopy(board)

        result = solve_sudoku(board)

        assert result == board
        assert board == original

    def test_sudoku_rejects_wrong_number_of_rows(self):
        board = [[0] * 9 for _ in range(8)]

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_wrong_number_of_columns(self):
        board = [[0] * 8 for _ in range(9)]

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_invalid_value_above_9(self):
        board = self.valid_board()
        board[0][0] = 10

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_negative_value(self):
        board = self.valid_board()
        board[0][0] = -1

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_non_integer_value(self):
        board = self.valid_board()
        board[0][0] = 2.5

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_duplicate_in_row(self):
        board = self.valid_board()
        board[0] = [5, 3, 5, 0, 7, 0, 0, 0, 0]

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_duplicate_in_column(self):
        board = self.valid_board()
        board[0][0] = 6

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_duplicate_in_box(self):
        board = self.valid_board()
        board[0][2] = 9

        with pytest.raises(ValueError):
            solve_sudoku(board)

    def test_sudoku_rejects_unsolvable_board(self):
        board = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 1],
        ]

        with pytest.raises(ValueError):
            solve_sudoku(board)

    @pytest.mark.parametrize(
        "invalid_board",
        [
            [],
            None,
            "sudoku",
            [[0] * 9],
            [[0] * 9 for _ in range(10)],
        ],
    )
    def test_sudoku_rejects_invalid_board_structure(self, invalid_board):
        with pytest.raises(ValueError):
            solve_sudoku(invalid_board)


# ============================================================================
# Maze Solver
# ============================================================================


class TestMaze:
    """Tests for the maze solver."""

    def test_maze_finds_path(self):
        maze = [
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
        ]

        result = solve_maze(maze)

        assert result[0] == (0, 0)
        assert result[-1] == (3, 3)

    def test_maze_path_contains_only_open_cells(self):
        maze = [
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
        ]

        result = solve_maze(maze)

        for row, column in result:
            assert maze[row][column] == 0

    def test_maze_path_moves_to_adjacent_cells(self):
        maze = [
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
        ]

        result = solve_maze(maze)

        for first, second in zip(result, result[1:]):
            row_difference = abs(first[0] - second[0])
            column_difference = abs(first[1] - second[1])

            assert row_difference + column_difference == 1

    def test_maze_custom_start_and_end(self):
        maze = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]

        result = solve_maze(
            maze,
            start=(0, 0),
            end=(2, 2),
        )

        assert result[0] == (0, 0)
        assert result[-1] == (2, 2)

    def test_maze_start_equals_end(self):
        maze = [[0]]

        assert solve_maze(maze, start=(0, 0), end=(0, 0)) == [(0, 0)]

    def test_maze_single_row(self):
        maze = [[0, 0, 0, 0]]

        assert solve_maze(maze) == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
        ]

    def test_maze_single_column(self):
        maze = [
            [0],
            [0],
            [0],
        ]

        assert solve_maze(maze) == [
            (0, 0),
            (1, 0),
            (2, 0),
        ]

    def test_maze_does_not_modify_input(self):
        maze = [
            [0, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
        ]

        original = deepcopy(maze)

        solve_maze(maze)

        assert maze == original

    def test_maze_raises_when_no_path_exists(self):
        maze = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze)

    def test_maze_rejects_empty_maze(self):
        with pytest.raises(ValueError):
            solve_maze([])

    def test_maze_rejects_empty_row(self):
        with pytest.raises(ValueError):
            solve_maze([[]])

    def test_maze_rejects_non_rectangular_maze(self):
        maze = [
            [0, 0, 0],
            [0, 0],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze)

    def test_maze_rejects_invalid_cell_value(self):
        maze = [
            [0, 0],
            [0, 2],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze)

    def test_maze_rejects_blocked_start(self):
        maze = [
            [1, 0],
            [0, 0],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze)

    def test_maze_rejects_blocked_end(self):
        maze = [
            [0, 0],
            [0, 1],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze)

    def test_maze_rejects_start_outside(self):
        maze = [
            [0, 0],
            [0, 0],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze, start=(2, 0))

    def test_maze_rejects_end_outside(self):
        maze = [
            [0, 0],
            [0, 0],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze, end=(2, 2))

    @pytest.mark.parametrize(
        "invalid_start",
        [
            (0,),
            (0, 0, 0),
            [0, 0],
            ("0", 0),
            (0, "1"),
        ],
    )
    def test_maze_rejects_invalid_start(self, invalid_start):
        maze = [
            [0, 0],
            [0, 0],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze, start=invalid_start)

    @pytest.mark.parametrize(
        "invalid_end",
        [
            (0,),
            (0, 0, 0),
            [0, 0],
            ("0", 0),
            (0, "1"),
        ],
    )
    def test_maze_rejects_invalid_end(self, invalid_end):
        maze = [
            [0, 0],
            [0, 0],
        ]

        with pytest.raises(ValueError):
            solve_maze(maze, end=invalid_end)


# ============================================================================
# Subsets
# ============================================================================


class TestSubsets:
    """Tests for subset generation."""

    def test_subsets_empty_input(self):
        assert subsets([]) == [[]]

    def test_subsets_single_element(self):
        assert subsets([1]) == [[], [1]]

    def test_subsets_two_elements(self):
        assert subsets([1, 2]) == [
            [],
            [2],
            [1],
            [1, 2],
        ]

    def test_subsets_three_elements(self):
        result = subsets([1, 2, 3])

        assert len(result) == 8

        expected = [
            [],
            [3],
            [2],
            [2, 3],
            [1],
            [1, 3],
            [1, 2],
            [1, 2, 3],
        ]

        assert result == expected

    def test_subsets_number_of_results_is_power_of_two(self):
        data = [1, 2, 3, 4]

        result = subsets(data)

        assert len(result) == 2 ** len(data)

    def test_subsets_does_not_modify_input(self):
        data = [1, 2, 3]
        original = data.copy()

        subsets(data)

        assert data == original

    def test_subsets_supports_strings(self):
        result = subsets(["a", "b"])

        assert result == [
            [],
            ["b"],
            ["a"],
            ["a", "b"],
        ]

    @pytest.mark.parametrize(
        "invalid_data",
        [
            "abc",
            b"abc",
            None,
            123,
        ],
    )
    def test_subsets_rejects_invalid_input(self, invalid_data):
        with pytest.raises(ValueError):
            subsets(invalid_data)


# ============================================================================
# Permutations
# ============================================================================


class TestPermutations:
    """Tests for permutation generation."""

    def test_permutations_empty_input(self):
        assert permutations([]) == [[]]

    def test_permutations_single_element(self):
        assert permutations([1]) == [[1]]

    def test_permutations_two_elements(self):
        assert permutations([1, 2]) == [
            [1, 2],
            [2, 1],
        ]

    def test_permutations_three_elements(self):
        result = permutations([1, 2, 3])

        assert len(result) == 6

        expected = {
            (1, 2, 3),
            (1, 3, 2),
            (2, 1, 3),
            (2, 3, 1),
            (3, 1, 2),
            (3, 2, 1),
        }

        assert {tuple(item) for item in result} == expected

    def test_permutations_with_duplicates_are_unique(self):
        result = permutations([1, 1, 2])

        expected = {
            (1, 1, 2),
            (1, 2, 1),
            (2, 1, 1),
        }

        assert {tuple(item) for item in result} == expected
        assert len(result) == 3

    def test_permutations_number_for_distinct_elements(self):
        data = [1, 2, 3, 4]

        result = permutations(data)

        assert len(result) == 24

    def test_permutations_does_not_modify_input(self):
        data = [1, 2, 3]
        original = data.copy()

        permutations(data)

        assert data == original

    def test_permutations_supports_strings(self):
        result = permutations(["a", "b"])

        assert result == [
            ["a", "b"],
            ["b", "a"],
        ]

    @pytest.mark.parametrize(
        "invalid_data",
        [
            "abc",
            b"abc",
            None,
            123,
        ],
    )
    def test_permutations_rejects_invalid_input(self, invalid_data):
        with pytest.raises(ValueError):
            permutations(invalid_data)


# ============================================================================
# Combination Sum
# ============================================================================


class TestCombinationSum:
    """Tests for Combination Sum."""

    def test_combination_sum_standard_example(self):
        result = combination_sum([2, 3, 6, 7], 7)

        assert result == [
            [2, 2, 3],
            [7],
        ]

    def test_combination_sum_second_standard_example(self):
        result = combination_sum([2, 3, 5], 8)

        assert result == [
            [2, 2, 2, 2],
            [2, 3, 3],
            [3, 5],
        ]

    def test_combination_sum_no_solution(self):
        assert combination_sum([2], 1) == []

    def test_combination_sum_empty_candidates(self):
        assert combination_sum([], 7) == []

    def test_combination_sum_target_zero(self):
        assert combination_sum([1, 2, 3], 0) == [[]]

    def test_combination_sum_single_candidate_exact_target(self):
        assert combination_sum([5], 10) == [[5, 5]]

    def test_combination_sum_single_candidate_non_multiple(self):
        assert combination_sum([5], 11) == []

    def test_combination_sum_removes_duplicate_candidates(self):
        result = combination_sum([2, 2, 3], 7)

        assert result == [
            [2, 2, 3],
        ]

    def test_combination_sum_results_are_sorted(self):
        result = combination_sum([7, 2, 3, 6], 7)

        for combination in result:
            assert combination == sorted(combination)

    def test_combination_sum_results_are_unique(self):
        result = combination_sum([2, 3, 6, 7], 7)

        assert len(result) == len(
            {tuple(combination) for combination in result}
        )

    def test_combination_sum_does_not_modify_input(self):
        candidates = [2, 3, 6, 7]
        original = candidates.copy()

        combination_sum(candidates, 7)

        assert candidates == original

    def test_combination_sum_large_target(self):
        result = combination_sum([2, 5], 10)

        assert result == [
            [2, 2, 2, 2, 2],
            [5, 5],
        ]

    def test_combination_sum_candidate_larger_than_target(self):
        assert combination_sum([10, 20, 30], 7) == []

    @pytest.mark.parametrize(
        "invalid_candidates",
        [
            None,
            "123",
            b"123",
            123,
        ],
    )
    def test_combination_sum_rejects_invalid_candidates(self, invalid_candidates):
        with pytest.raises(ValueError):
            combination_sum(invalid_candidates, 7)

    @pytest.mark.parametrize(
        "invalid_candidates",
        [
            [0, 1, 2],
            [-1, 2, 3],
            [2, 0],
            [2, -5],
        ],
    )
    def test_combination_sum_rejects_non_positive_candidates(
        self,
        invalid_candidates,
    ):
        with pytest.raises(ValueError):
            combination_sum(invalid_candidates, 7)

    def test_combination_sum_rejects_non_integer_candidate(self):
        with pytest.raises(ValueError):
            combination_sum([2, 3.5, 7], 7)

    @pytest.mark.parametrize(
        "invalid_target",
        [
            -1,
            -10,
            2.5,
            "7",
            None,
            True,
        ],
    )
    def test_combination_sum_rejects_invalid_target(self, invalid_target):
        with pytest.raises(ValueError):
            combination_sum([2, 3, 7], invalid_target)


# ============================================================================
# Cross-algorithm checks
# ============================================================================


class TestBacktrackingConsistency:
    """General consistency checks across the backtracking algorithms."""

    def test_n_queens_all_solutions_are_distinct(self):
        result = n_queens(5)

        assert len(result) == len({tuple(solution) for solution in result})

    def test_subsets_all_results_are_lists(self):
        result = subsets([1, 2, 3])

        assert all(isinstance(item, list) for item in result)

    def test_permutations_all_results_are_lists(self):
        result = permutations([1, 2, 3])

        assert all(isinstance(item, list) for item in result)

    def test_combination_sum_all_results_are_lists(self):
        result = combination_sum([2, 3, 6, 7], 7)

        assert all(isinstance(item, list) for item in result)

    def test_combination_sum_every_result_reaches_target(self):
        candidates = [2, 3, 6, 7]
        target = 7

        result = combination_sum(candidates, target)

        for combination in result:
            assert sum(combination) == target

    def test_maze_result_is_a_valid_path(self):
        maze = [
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
        ]

        path = solve_maze(maze)

        assert path[0] == (0, 0)
        assert path[-1] == (3, 3)

        for row, column in path:
            assert maze[row][column] == 0

