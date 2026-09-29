
"""
Tests for Tree Algorithms
=========================

Covered:
- Preorder traversal
- Inorder traversal
- Postorder traversal
- Level order traversal
- BST search
- BST insert
- BST delete
- Min Heap
- Priority Queue
"""

import pytest

from core.algorithms.tree_algorithms import (
    MinHeap,
    PriorityQueue,
    TreeNode,
    bst_delete,
    bst_insert,
    bst_search,
    inorder_traversal,
    level_order_traversal,
    postorder_traversal,
    preorder_traversal,
)


# ============================================================================
# Fixtures / Helpers
# ============================================================================


def build_sample_tree():
    """
    Build the following binary tree:
        1
       / \\
      2   3
     / \\   \\
    4   5   6
       /
      7

    """
    return TreeNode(
        1,
        left=TreeNode(
            2,
            left=TreeNode(4),
            right=TreeNode(
                5,
                left=TreeNode(7),
            ),
        ),
        right=TreeNode(
            3,
            right=TreeNode(6),
        ),
    )


def build_sample_bst():
    """
    Build the following BST:

            8
           / \
          3   10
         / \\    \
        1   6    14
           / \\   /
          4   7 13
    """
    root = TreeNode(8)

    bst_insert(root, 3)
    bst_insert(root, 10)
    bst_insert(root, 1)
    bst_insert(root, 6)
    bst_insert(root, 14)
    bst_insert(root, 4)
    bst_insert(root, 7)
    bst_insert(root, 13)

    return root


# ============================================================================
# TreeNode
# ============================================================================


class TestTreeNode:
    """Tests for TreeNode."""

    def test_tree_node_stores_value(self):
        node = TreeNode(10)

        assert node.value == 10

    def test_tree_node_children_default_to_none(self):
        node = TreeNode(10)

        assert node.left is None
        assert node.right is None

    def test_tree_node_accepts_children(self):
        left = TreeNode(5)
        right = TreeNode(15)
        root = TreeNode(10, left, right)

        assert root.left is left
        assert root.right is right

    def test_tree_node_repr(self):
        node = TreeNode(42)

        assert repr(node) == "TreeNode(42)"


# ============================================================================
# Preorder
# ============================================================================


class TestPreorderTraversal:
    """Tests for preorder traversal."""

    def test_preorder_empty_tree(self):
        assert preorder_traversal(None) == []

    def test_preorder_single_node(self):
        root = TreeNode(1)

        assert preorder_traversal(root) == [1]

    def test_preorder_sample_tree(self):
        root = build_sample_tree()

        assert preorder_traversal(root) == [
            1,
            2,
            4,
            5,
            7,
            3,
            6,
        ]

    def test_preorder_left_skewed_tree(self):
        root = TreeNode(
            3,
            left=TreeNode(
                2,
                left=TreeNode(1),
            ),
        )

        assert preorder_traversal(root) == [3, 2, 1]

    def test_preorder_right_skewed_tree(self):
        root = TreeNode(
            1,
            right=TreeNode(
                2,
                right=TreeNode(3),
            ),
        )

        assert preorder_traversal(root) == [1, 2, 3]

    def test_preorder_does_not_modify_tree(self):
        root = build_sample_tree()

        before = inorder_traversal(root)

        preorder_traversal(root)

        after = inorder_traversal(root)

        assert before == after

    def test_preorder_rejects_invalid_root(self):
        with pytest.raises(ValueError):
            preorder_traversal(10)


# ============================================================================
# Inorder
# ============================================================================


class TestInorderTraversal:
    """Tests for inorder traversal."""

    def test_inorder_empty_tree(self):
        assert inorder_traversal(None) == []

    def test_inorder_single_node(self):
        root = TreeNode(1)

        assert inorder_traversal(root) == [1]

    def test_inorder_sample_tree(self):
        root = build_sample_tree()

        assert inorder_traversal(root) == [
            4,
            2,
            7,
            5,
            1,
            3,
            6,
        ]

    def test_inorder_left_skewed_tree(self):
        root = TreeNode(
            3,
            left=TreeNode(
                2,
                left=TreeNode(1),
            ),
        )

        assert inorder_traversal(root) == [1, 2, 3]

    def test_inorder_right_skewed_tree(self):
        root = TreeNode(
            1,
            right=TreeNode(
                2,
                right=TreeNode(3),
            ),
        )

        assert inorder_traversal(root) == [1, 2, 3]

    def test_inorder_does_not_modify_tree(self):
        root = build_sample_tree()

        before = preorder_traversal(root)

        inorder_traversal(root)

        after = preorder_traversal(root)

        assert before == after

    def test_inorder_rejects_invalid_root(self):
        with pytest.raises(ValueError):
            inorder_traversal("tree")


# ============================================================================
# Postorder
# ============================================================================


class TestPostorderTraversal:
    """Tests for postorder traversal."""

    def test_postorder_empty_tree(self):
        assert postorder_traversal(None) == []

    def test_postorder_single_node(self):
        root = TreeNode(1)

        assert postorder_traversal(root) == [1]

    def test_postorder_sample_tree(self):
        root = build_sample_tree()

        assert postorder_traversal(root) == [
            4,
            7,
            5,
            2,
            6,
            3,
            1,
        ]

    def test_postorder_left_skewed_tree(self):
        root = TreeNode(
            3,
            left=TreeNode(
                2,
                left=TreeNode(1),
            ),
        )

        assert postorder_traversal(root) == [1, 2, 3]

    def test_postorder_right_skewed_tree(self):
        root = TreeNode(
            1,
            right=TreeNode(
                2,
                right=TreeNode(3),
            ),
        )

        assert postorder_traversal(root) == [3, 2, 1]

    def test_postorder_does_not_modify_tree(self):
        root = build_sample_tree()

        before = level_order_traversal(root)

        postorder_traversal(root)

        after = level_order_traversal(root)

        assert before == after

    def test_postorder_rejects_invalid_root(self):
        with pytest.raises(ValueError):
            postorder_traversal([])


# ============================================================================
# Level Order
# ============================================================================


class TestLevelOrderTraversal:
    """Tests for level-order traversal."""

    def test_level_order_empty_tree(self):
        assert level_order_traversal(None) == []

    def test_level_order_single_node(self):
        root = TreeNode(1)

        assert level_order_traversal(root) == [1]

    def test_level_order_sample_tree(self):
        root = build_sample_tree()

        assert level_order_traversal(root) == [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]

    def test_level_order_left_skewed_tree(self):
        root = TreeNode(
            3,
            left=TreeNode(
                2,
                left=TreeNode(1),
            ),
        )

        assert level_order_traversal(root) == [3, 2, 1]

    def test_level_order_right_skewed_tree(self):
        root = TreeNode(
            1,
            right=TreeNode(
                2,
                right=TreeNode(3),
            ),
        )

        assert level_order_traversal(root) == [1, 2, 3]

    def test_level_order_does_not_modify_tree(self):
        root = build_sample_tree()

        before = preorder_traversal(root)

        level_order_traversal(root)

        after = preorder_traversal(root)

        assert before == after

    def test_level_order_rejects_invalid_root(self):
        with pytest.raises(ValueError):
            level_order_traversal(123)


# ============================================================================
# BST Search
# ============================================================================


class TestBSTSearch:
    """Tests for BST search."""

    def test_bst_search_finds_root(self):
        root = build_sample_bst()

        result = bst_search(root, 8)

        assert result is root
        assert result.value == 8

    def test_bst_search_finds_left_node(self):
        root = build_sample_bst()

        result = bst_search(root, 4)

        assert result is not None
        assert result.value == 4

    def test_bst_search_finds_right_node(self):
        root = build_sample_bst()

        result = bst_search(root, 13)

        assert result is not None
        assert result.value == 13

    def test_bst_search_missing_value(self):
        root = build_sample_bst()

        assert bst_search(root, 100) is None

    def test_bst_search_empty_tree(self):
        assert bst_search(None, 10) is None

    def test_bst_search_does_not_modify_tree(self):
        root = build_sample_bst()

        before = inorder_traversal(root)

        bst_search(root, 7)

        after = inorder_traversal(root)

        assert before == after

    def test_bst_search_rejects_invalid_root(self):
        with pytest.raises(ValueError):
            bst_search("tree", 5)

    def test_bst_search_rejects_boolean_value(self):
        root = build_sample_bst()

        with pytest.raises(ValueError):
            bst_search(root, True)


# ============================================================================
# BST Insert
# ============================================================================


class TestBSTInsert:
    """Tests for BST insertion."""

    def test_bst_insert_into_empty_tree(self):
        root = bst_insert(None, 10)

        assert isinstance(root, TreeNode)
        assert root.value == 10

    def test_bst_insert_left_child(self):
        root = TreeNode(10)

        result = bst_insert(root, 5)

        assert result is root
        assert root.left is not None
        assert root.left.value == 5

    def test_bst_insert_right_child(self):
        root = TreeNode(10)

        result = bst_insert(root, 15)

        assert result is root
        assert root.right is not None
        assert root.right.value == 15

    def test_bst_insert_multiple_values(self):
        root = None

        for value in [8, 3, 10, 1, 6, 14]:
            root = bst_insert(root, value)

        assert inorder_traversal(root) == [
            1,
            3,
            6,
            8,
            10,
            14,
        ]

    def test_bst_insert_duplicate_is_ignored(self):
        root = TreeNode(10)

        bst_insert(root, 10)

        assert inorder_traversal(root) == [10]

    def test_bst_insert_duplicate_does_not_create_node(self):
        root = build_sample_bst()

        before = inorder_traversal(root)

        bst_insert(root, 6)

        after = inorder_traversal(root)

        assert after == before

    def test_bst_insert_returns_same_root(self):
        root = TreeNode(10)

        result = bst_insert(root, 5)

        assert result is root

    def test_bst_insert_rejects_invalid_root(self):
        with pytest.raises(ValueError):
            bst_insert([], 5)

    def test_bst_insert_rejects_boolean_value(self):
        root = TreeNode(10)

        with pytest.raises(ValueError):
            bst_insert(root, False)


# ============================================================================
# BST Delete
# ============================================================================


class TestBSTDelete:
    """Tests for BST deletion."""

    def test_bst_delete_missing_value(self):
        root = build_sample_bst()

        result = bst_delete(root, 100)

        assert result is root
        assert inorder_traversal(result) == [
            1,
            3,
            4,
            6,
            7,
            8,
            10,
            13,
            14,
        ]

    def test_bst_delete_leaf(self):
        root = build_sample_bst()

        result = bst_delete(root, 1)

        assert inorder_traversal(result) == [
            3,
            4,
            6,
            7,
            8,
            10,
            13,
            14,
        ]

    def test_bst_delete_node_with_one_child(self):
        root = build_sample_bst()

        result = bst_delete(root, 14)

        assert inorder_traversal(result) == [
            1,
            3,
            4,
            6,
            7,
            8,
            10,
            13,
        ]

    def test_bst_delete_node_with_two_children(self):
        root = build_sample_bst()

        result = bst_delete(root, 3)

        assert inorder_traversal(result) == [
            1,
            4,
            6,
            7,
            8,
            10,
            13,
            14,
        ]

    def test_bst_delete_root_with_two_children(self):
        root = build_sample_bst()

        result = bst_delete(root, 8)

        assert result is not None
        assert result.value == 10
        assert inorder_traversal(result) == [
            1,
            3,
            4,
            6,
            7,
            10,
            13,
            14,
        ]

    def test_bst_delete_root_with_only_left_child(self):
        root = TreeNode(
            5,
            left=TreeNode(3),
        )

        result = bst_delete(root, 5)

        assert result is not None
        assert result.value == 3

    def test_bst_delete_root_with_only_right_child(self):
        root = TreeNode(
            5,
            right=TreeNode(8),
        )

        result = bst_delete(root, 5)

        assert result is not None
        assert result.value == 8

    def test_bst_delete_single_node(self):
        root = TreeNode(5)

        result = bst_delete(root, 5)

        assert result is None

    def test_bst_delete_from_empty_tree(self):
        assert bst_delete(None, 5) is None

    def test_bst_delete_preserves_bst_property(self):
        root = build_sample_bst()

        result = bst_delete(root, 6)

        values = inorder_traversal(result)

        assert values == sorted(values)

    def test_bst_delete_rejects_invalid_root(self):
        with pytest.raises(ValueError):
            bst_delete("tree", 5)

    def test_bst_delete_rejects_boolean_value(self):
        root = build_sample_bst()

        with pytest.raises(ValueError):
            bst_delete(root, True)


# ============================================================================
# Min Heap
# ============================================================================


class TestMinHeap:
    """Tests for MinHeap."""

    def test_heap_starts_empty(self):
        heap = MinHeap()

        assert heap.is_empty()
        assert heap.size() == 0

    def test_heap_push_and_peek(self):
        heap = MinHeap()

        heap.push(5)
        heap.push(2)
        heap.push(8)

        assert heap.peek() == 2
        assert heap.size() == 3

    def test_heap_pop_returns_smallest(self):
        heap = MinHeap()

        for value in [5, 2, 8, 1, 4]:
            heap.push(value)

        assert heap.pop() == 1
        assert heap.pop() == 2
        assert heap.pop() == 4
        assert heap.pop() == 5
        assert heap.pop() == 8

    def test_heap_pop_produces_sorted_order(self):
        values = [9, 3, 7, 1, 8, 2, 5]

        heap = MinHeap()

        for value in values:
            heap.push(value)

        result = [heap.pop() for _ in values]

        assert result == sorted(values)

    def test_heap_builds_from_list(self):
        values = [5, 1, 8, 2, 4]

        heap = MinHeap(values)

        assert heap.peek() == 1
        assert heap.size() == 5

    def test_heap_constructor_does_not_modify_input(self):
        values = [5, 1, 8, 2, 4]
        original = values.copy()

        MinHeap(values)

        assert values == original

    def test_heap_to_list_returns_copy(self):
        heap = MinHeap([3, 1, 2])

        result = heap.to_list()
        result.append(100)

        assert heap.size() == 3
        assert 100 not in heap.to_list()

    def test_heap_empty_after_all_pops(self):
        heap = MinHeap([3, 1, 2])

        heap.pop()
        heap.pop()
        heap.pop()

        assert heap.is_empty()
        assert heap.size() == 0

    def test_heap_push_after_becoming_empty(self):
        heap = MinHeap([2])

        assert heap.pop() == 2
        assert heap.is_empty()

        heap.push(1)

        assert heap.peek() == 1

    def test_heap_single_element(self):
        heap = MinHeap([42])

        assert heap.peek() == 42
        assert heap.pop() == 42
        assert heap.is_empty()

    def test_heap_duplicate_values(self):
        heap = MinHeap([3, 1, 3, 1, 2])

        result = [heap.pop() for _ in range(5)]

        assert result == [1, 1, 2, 3, 3]

    def test_heap_negative_values(self):
        heap = MinHeap([3, -1, 2, -5, 0])

        result = [heap.pop() for _ in range(5)]

        assert result == [-5, -1, 0, 2, 3]

    def test_heap_peek_does_not_remove_element(self):
        heap = MinHeap([3, 1, 2])

        assert heap.peek() == 1
        assert heap.size() == 3
        assert heap.peek() == 1

    def test_heap_pop_empty_raises(self):
        heap = MinHeap()

        with pytest.raises(IndexError):
            heap.pop()

    def test_heap_peek_empty_raises(self):
        heap = MinHeap()

        with pytest.raises(IndexError):
            heap.peek()

    def test_heap_rejects_non_list_constructor_input(self):
        with pytest.raises(ValueError):
            MinHeap((3, 1, 2))


# ============================================================================
# Priority Queue
# ============================================================================


class TestPriorityQueue:
    """Tests for PriorityQueue."""

    def test_priority_queue_starts_empty(self):
        queue = PriorityQueue()

        assert queue.is_empty()
        assert queue.size() == 0

    def test_priority_queue_push_and_peek(self):
        queue = PriorityQueue()

        queue.push("normal", priority=5)
        queue.push("urgent", priority=1)

        assert queue.peek() == "urgent"
        assert queue.size() == 2

    def test_priority_queue_pops_lowest_priority_first(self):
        queue = PriorityQueue()

        queue.push("low", priority=10)
        queue.push("high", priority=1)
        queue.push("medium", priority=5)

        assert queue.pop() == "high"
        assert queue.pop() == "medium"
        assert queue.pop() == "low"

    def test_priority_queue_same_priority_is_fifo(self):
        queue = PriorityQueue()

        queue.push("first", priority=1)
        queue.push("second", priority=1)
        queue.push("third", priority=1)

        assert queue.pop() == "first"
        assert queue.pop() == "second"
        assert queue.pop() == "third"

    def test_priority_queue_mixed_priorities(self):
        queue = PriorityQueue()

        queue.push("A", priority=3)
        queue.push("B", priority=1)
        queue.push("C", priority=2)
        queue.push("D", priority=1)

        assert queue.pop() == "B"
        assert queue.pop() == "D"
        assert queue.pop() == "C"
        assert queue.pop() == "A"

    def test_priority_queue_negative_priority(self):
        queue = PriorityQueue()

        queue.push("normal", priority=0)
        queue.push("urgent", priority=-10)

        assert queue.pop() == "urgent"
        assert queue.pop() == "normal"

    def test_priority_queue_float_priority(self):
        queue = PriorityQueue()

        queue.push("A", priority=2.5)
        queue.push("B", priority=1.5)
        queue.push("C", priority=3.5)

        assert queue.pop() == "B"
        assert queue.pop() == "A"
        assert queue.pop() == "C"

    def test_priority_queue_size_updates(self):
        queue = PriorityQueue()

        queue.push("A", priority=1)
        queue.push("B", priority=2)

        assert queue.size() == 2

        queue.pop()

        assert queue.size() == 1

        queue.pop()

        assert queue.size() == 0
        assert queue.is_empty()

    def test_priority_queue_peek_does_not_remove(self):
        queue = PriorityQueue()

        queue.push("A", priority=1)
        queue.push("B", priority=2)

        assert queue.peek() == "A"
        assert queue.size() == 2
        assert queue.peek() == "A"

    def test_priority_queue_pop_empty_raises(self):
        queue = PriorityQueue()

        with pytest.raises(IndexError):
            queue.pop()

    def test_priority_queue_peek_empty_raises(self):
        queue = PriorityQueue()

        with pytest.raises(IndexError):
            queue.peek()

    @pytest.mark.parametrize(
        "invalid_priority",
        [
            True,
            False,
            "1",
            None,
            [1],
        ],
    )
    def test_priority_queue_rejects_invalid_priority(self, invalid_priority):
        queue = PriorityQueue()

        with pytest.raises(ValueError):
            queue.push("item", priority=invalid_priority)


# ============================================================================
# Cross-algorithm consistency
# ============================================================================


class TestTreeAlgorithmsConsistency:
    """Consistency checks between tree algorithms."""

    def test_bst_inorder_is_sorted(self):
        root = build_sample_bst()

        result = inorder_traversal(root)

        assert result == sorted(result)

    def test_bst_search_matches_inorder_values(self):
        root = build_sample_bst()

        values = inorder_traversal(root)

        for value in values:
            assert bst_search(root, value) is not None

    def test_bst_insert_then_search(self):
        root = None
        values = [50, 30, 70, 20, 40, 60, 80]

        for value in values:
            root = bst_insert(root, value)

        for value in values:
            node = bst_search(root, value)

            assert node is not None
            assert node.value == value

    def test_bst_delete_removes_only_requested_value(self):
        root = build_sample_bst()

        original = inorder_traversal(root)

        result = bst_delete(root, 6)

        after = inorder_traversal(result)

        assert 6 in original
        assert 6 not in after
        assert len(after) == len(original) - 1

        remaining_original = original.copy()
        remaining_original.remove(6)

        assert after == remaining_original

    def test_heap_pop_order_matches_sorted_input(self):
        values = [10, 4, 7, 1, 9, 2, 6]

        heap = MinHeap(values)

        result = []

        while not heap.is_empty():
            result.append(heap.pop())

        assert result == sorted(values)

    def test_priority_queue_pop_order_matches_priority_order(self):
        queue = PriorityQueue()

        items = [
            ("A", 4),
            ("B", 2),
            ("C", 5),
            ("D", 1),
            ("E", 3),
        ]

        for value, priority in items:
            queue.push(value, priority)

        result = [queue.pop() for _ in items]

        assert result == ["D", "B", "E", "A", "C"]

