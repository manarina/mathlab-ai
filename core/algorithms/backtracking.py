
"""
Backtracking Algorithms
=======================

Algorithms implemented:
- N-Queens
- Sudoku Solver
- Maze Solver
- Subsets
- Permutations
- Combination Sum

The functions in this module:
- validate their inputs,
- avoid mutating user-provided data,
- produce deterministic results,
- raise ValueError for invalid inputs or unsolvable problems.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any, TypeVar


T = TypeVar("T")


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate that value is a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_sequence(data: Sequence[Any], name: str) -> None:
    """Validate a sequence input."""
    if isinstance(data, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of elements, not a string.")

    if not isinstance(data, Sequence):
        raise ValueError(f"{name} must be a sequence.")


# ============================================================================
# N-Queens
# ============================================================================


def n_queens(n: int) -> list[list[int]]:
    """
    Solve the N-Queens problem.

    Each solution is represented by a list where:
    - index = row
    - value = column
    - rows and columns are 0-based

    Example for n=4:

        [
            [1, 3, 0, 2],
            [2, 0, 3, 1],
        ]

    Parameters
    ----------
    n:
        Number of queens and size of the chessboard.

    Returns
    -------
    list[list[int]]
        All valid solutions.

    Raises
    ------
    ValueError
        If n is not a positive integer.
    """
    _validate_positive_integer(n, "n")

    solutions: list[list[int]] = []
    board = [-1] * n

    used_columns: set[int] = set()
    used_diagonals: set[int] = set()
    used_antidiagonals: set[int] = set()

    def backtrack(row: int) -> None:
        if row == n:
            solutions.append(board.copy())
            return

        for column in range(n):
            diagonal = row - column
            antidiagonal = row + column

            if column in used_columns:
                continue

            if diagonal in used_diagonals:
                continue

            if antidiagonal in used_antidiagonals:
                continue

            board[row] = column

            used_columns.add(column)
            used_diagonals.add(diagonal)
            used_antidiagonals.add(antidiagonal)

            backtrack(row + 1)

            used_columns.remove(column)
            used_diagonals.remove(diagonal)
            used_antidiagonals.remove(antidiagonal)

            board[row] = -1

    backtrack(0)

    return solutions


# ============================================================================
# Sudoku Solver
# ============================================================================


def _validate_sudoku_board(board: Sequence[Sequence[int]]) -> None:
    """Validate the structure and initial state of a Sudoku board."""
    if not isinstance(board, Sequence) or isinstance(board, (str, bytes)):
        raise ValueError("board must be a sequence.")

    if len(board) != 9:
        raise ValueError("Sudoku board must contain exactly 9 rows.")

    for row in board:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("Each Sudoku row must be a sequence.")

        if len(row) != 9:
            raise ValueError("Each Sudoku row must contain exactly 9 values.")

        for value in row:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("Sudoku values must be integers.")

            if value < 0 or value > 9:
                raise ValueError("Sudoku values must be between 0 and 9.")

    # Check duplicate values in rows, columns and 3x3 boxes.
    for row in board:
        values = [value for value in row if value != 0]
        if len(values) != len(set(values)):
            raise ValueError("Sudoku board contains duplicate values in a row.")

    for column in range(9):
        values = [
            board[row][column]
            for row in range(9)
            if board[row][column] != 0
        ]

        if len(values) != len(set(values)):
            raise ValueError("Sudoku board contains duplicate values in a column.")

    for box_row in range(0, 9, 3):
        for box_column in range(0, 9, 3):
            values = []

            for row in range(box_row, box_row + 3):
                for column in range(box_column, box_column + 3):
                    value = board[row][column]

                    if value != 0:
                        values.append(value)

            if len(values) != len(set(values)):
                raise ValueError(
                    "Sudoku board contains duplicate values in a 3x3 box."
                )


def solve_sudoku(board: Sequence[Sequence[int]]) -> list[list[int]]:
    """
    Solve a 9x9 Sudoku board.

    Empty cells must contain 0.

    The input board is never modified.

    Parameters
    ----------
    board:
        9x9 Sudoku board.

    Returns
    -------
    list[list[int]]
        Solved Sudoku board.

    Raises
    ------
    ValueError
        If the board is invalid or has no solution.
    """
    _validate_sudoku_board(board)

    grid = [list(row) for row in board]

    def find_empty_cell() -> tuple[int, int] | None:
        for row in range(9):
            for column in range(9):
                if grid[row][column] == 0:
                    return row, column

        return None

    def is_valid(row: int, column: int, value: int) -> bool:
        # Row
        if value in grid[row]:
            return False

        # Column
        for current_row in range(9):
            if grid[current_row][column] == value:
                return False

        # 3x3 box
        box_row = (row // 3) * 3
        box_column = (column // 3) * 3

        for current_row in range(box_row, box_row + 3):
            for current_column in range(box_column, box_column + 3):
                if grid[current_row][current_column] == value:
                    return False

        return True

    def backtrack() -> bool:
        empty_cell = find_empty_cell()

        if empty_cell is None:
            return True

        row, column = empty_cell

        for value in range(1, 10):
            if not is_valid(row, column, value):
                continue

            grid[row][column] = value

            if backtrack():
                return True

            grid[row][column] = 0

        return False

    if not backtrack():
        raise ValueError("Sudoku board has no solution.")

    return grid


# ============================================================================
# Maze Solver
# ============================================================================


def _validate_maze(
    maze: Sequence[Sequence[int]],
) -> tuple[int, int]:
    """Validate a rectangular binary maze and return (rows, columns)."""
    if not isinstance(maze, Sequence) or isinstance(maze, (str, bytes)):
        raise ValueError("maze must be a sequence of rows.")

    if len(maze) == 0:
        raise ValueError("maze must not be empty.")

    if not isinstance(maze[0], Sequence) or isinstance(maze[0], (str, bytes)):
        raise ValueError("Each maze row must be a sequence.")

    columns = len(maze[0])

    if columns == 0:
        raise ValueError("maze rows must not be empty.")

    for row in maze:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("Each maze row must be a sequence.")

        if len(row) != columns:
            raise ValueError("maze must be rectangular.")

        for value in row:
            if value not in (0, 1):
                raise ValueError("maze cells must be either 0 or 1.")

    return len(maze), columns


def solve_maze(
    maze: Sequence[Sequence[int]],
    start: tuple[int, int] = (0, 0),
    end: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """
    Find a path through a binary maze using backtracking.

    Convention:
    - 0 = open cell
    - 1 = blocked cell

    Movement is allowed in four directions:
    - up
    - down
    - left
    - right

    Parameters
    ----------
    maze:
        Rectangular binary maze.
    start:
        Starting coordinate as (row, column).
    end:
        Destination coordinate. Defaults to bottom-right cell.

    Returns
    -------
    list[tuple[int, int]]
        Coordinates of a valid path from start to end, inclusive.

    Raises
    ------
    ValueError
        If the maze is invalid, start/end are invalid or blocked,
        or no path exists.
    """
    rows, columns = _validate_maze(maze)

    if (
        not isinstance(start, tuple)
        or len(start) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in start)
    ):
        raise ValueError("start must be a tuple of two integers.")

    if end is None:
        end = (rows - 1, columns - 1)

    if (
        not isinstance(end, tuple)
        or len(end) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in end)
    ):
        raise ValueError("end must be a tuple of two integers.")

    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))

    def is_inside(position: tuple[int, int]) -> bool:
        row, column = position
        return 0 <= row < rows and 0 <= column < columns

    if not is_inside(start):
        raise ValueError("start is outside the maze.")

    if not is_inside(end):
        raise ValueError("end is outside the maze.")

    if maze[start[0]][start[1]] == 1:
        raise ValueError("start cell is blocked.")

    if maze[end[0]][end[1]] == 1:
        raise ValueError("end cell is blocked.")

    path: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()

    directions = (
        (-1, 0),  # up
        (0, 1),   # right
        (1, 0),   # down
        (0, -1),  # left
    )

    def backtrack(position: tuple[int, int]) -> bool:
        if position in visited:
            return False

        visited.add(position)
        path.append(position)

        if position == end:
            return True

        row, column = position

        for delta_row, delta_column in directions:
            next_position = (
                row + delta_row,
                column + delta_column,
            )

            if not is_inside(next_position):
                continue

            if maze[next_position[0]][next_position[1]] == 1:
                continue

            if backtrack(next_position):
                return True

        path.pop()
        return False

    if not backtrack(start):
        raise ValueError("No path exists between start and end.")

    return path


# ============================================================================
# Subsets
# ============================================================================


def subsets(data: Sequence[T]) -> list[list[T]]:
    """
    Generate all subsets of a sequence.

    The input is not modified.

    For n elements, the function returns 2^n subsets.

    Parameters
    ----------
    data:
        Input sequence.

    Returns
    -------
    list[list[T]]
        All subsets.
    """
    _validate_sequence(data, "data")

    items = list(data)
    result: list[list[T]] = []

    current: list[T] = []

    def backtrack(index: int) -> None:
        if index == len(items):
            result.append(current.copy())
            return

        # Exclude current element.
        backtrack(index + 1)

        # Include current element.
        current.append(items[index])
        backtrack(index + 1)
        current.pop()

    backtrack(0)

    return result


# ============================================================================
# Permutations
# ============================================================================


def permutations(data: Sequence[T]) -> list[list[T]]:
    """
    Generate all permutations of a sequence.

    For n distinct elements, the function returns n! permutations.

    Duplicate input values are handled without producing duplicate
    permutations.

    Parameters
    ----------
    data:
        Input sequence.

    Returns
    -------
    list[list[T]]
        All unique permutations.
    """
    _validate_sequence(data, "data")

    items = list(data)
    result: list[list[T]] = []
    current: list[T] = []
    used = [False] * len(items)

    def backtrack() -> None:
        if len(current) == len(items):
            result.append(current.copy())
            return

        used_at_level: set[Any] = set()

        for index, value in enumerate(items):
            if used[index]:
                continue

            if value in used_at_level:
                continue

            used_at_level.add(value)

            used[index] = True
            current.append(value)

            backtrack()

            current.pop()
            used[index] = False

    backtrack()

    return result


# ============================================================================
# Combination Sum
# ============================================================================


def combination_sum(
    candidates: Sequence[int],
    target: int,
) -> list[list[int]]:
    """
    Find all unique combinations that sum to target.

    Each candidate can be used unlimited times.

    Candidates must be positive integers.

    The returned combinations are sorted in nondecreasing order,
    and the final result is deterministic.

    Example
    -------
    combination_sum([2, 3, 6, 7], 7)

    returns:

        [
            [2, 2, 3],
            [7],
        ]

    Parameters
    ----------
    candidates:
        Positive integer candidate values.
    target:
        Non-negative target value.

    Returns
    -------
    list[list[int]]
        All unique combinations.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    _validate_sequence(candidates, "candidates")

    if isinstance(target, bool) or not isinstance(target, Integral):
        raise ValueError("target must be an integer.")

    if target < 0:
        raise ValueError("target must be non-negative.")

    normalized_candidates: list[int] = []

    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, Integral):
            raise ValueError("candidates must contain only integers.")

        if candidate <= 0:
            raise ValueError("candidates must contain only positive integers.")

        normalized_candidates.append(int(candidate))

    # Remove duplicates and sort for deterministic backtracking.
    normalized_candidates = sorted(set(normalized_candidates))

    result: list[list[int]] = []
    current: list[int] = []

    def backtrack(start_index: int, remaining: int) -> None:
        if remaining == 0:
            result.append(current.copy())
            return

        for index in range(start_index, len(normalized_candidates)):
            candidate = normalized_candidates[index]

            if candidate > remaining:
                break

            current.append(candidate)

            # The same candidate can be reused.
            backtrack(index, remaining - candidate)

            current.pop()

    if target == 0:
        return [[]]

    backtrack(0, int(target))

    return result

