
"""
Tree Algorithms
===============

Algorithms and data structures implemented:
- Binary Tree Traversals:
    - Preorder
    - Inorder
    - Postorder
    - Level Order
- Binary Search Tree:
    - Search
    - Insert
    - Delete
- Min Heap
- Priority Queue

Design principles:
- deterministic behavior,
- explicit validation,
- no unnecessary mutation of user data,
- simple and reusable data structures.
"""

from __future__ import annotations

from collections import deque
from numbers import Real
from typing import Any


# ============================================================================
# Binary Tree Node
# ============================================================================


class TreeNode:
    """
    Node used by binary trees and binary search trees.

    Parameters
    ----------
    value:
        Value stored in the node.
    left:
        Left child.
    right:
        Right child.
    """

    def __init__(
        self,
        value: Any,
        left: TreeNode | None = None,
        right: TreeNode | None = None,
    ) -> None:
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"TreeNode({self.value!r})"


# ============================================================================
# Validation
# ============================================================================


def _validate_tree_root(root: TreeNode | None) -> None:
    """Validate a binary tree root."""
    if root is not None and not isinstance(root, TreeNode):
        raise ValueError("root must be a TreeNode or None.")


def _validate_bst_value(value: Any) -> None:
    """
    Validate a value used by BST operations.

    Boolean values are rejected because bool is a subclass of int and
    accepting True/False as numerical tree values is usually unintended.
    """
    if isinstance(value, bool):
        raise ValueError("BST values cannot be boolean.")


# ============================================================================
# Binary Tree Traversals
# ============================================================================


def preorder_traversal(root: TreeNode | None) -> list[Any]:
    """
    Traverse a binary tree using preorder.

    Order:
        Root -> Left -> Right

    Returns
    -------
    list
        Values in preorder.
    """
    _validate_tree_root(root)

    result: list[Any] = []

    def traverse(node: TreeNode | None) -> None:
        if node is None:
            return

        result.append(node.value)
        traverse(node.left)
        traverse(node.right)

    traverse(root)

    return result


def inorder_traversal(root: TreeNode | None) -> list[Any]:
    """
    Traverse a binary tree using inorder.

    Order:
        Left -> Root -> Right

    Returns
    -------
    list
        Values in inorder.
    """
    _validate_tree_root(root)

    result: list[Any] = []

    def traverse(node: TreeNode | None) -> None:
        if node is None:
            return

        traverse(node.left)
        result.append(node.value)
        traverse(node.right)

    traverse(root)

    return result


def postorder_traversal(root: TreeNode | None) -> list[Any]:
    """
    Traverse a binary tree using postorder.

    Order:
        Left -> Right -> Root

    Returns
    -------
    list
        Values in postorder.
    """
    _validate_tree_root(root)

    result: list[Any] = []

    def traverse(node: TreeNode | None) -> None:
        if node is None:
            return

        traverse(node.left)
        traverse(node.right)
        result.append(node.value)

    traverse(root)

    return result


def level_order_traversal(root: TreeNode | None) -> list[Any]:
    """
    Traverse a binary tree level by level.

    Returns
    -------
    list
        Values in breadth-first / level-order sequence.
    """
    _validate_tree_root(root)

    if root is None:
        return []

    result: list[Any] = []
    queue: deque[TreeNode] = deque([root])

    while queue:
        node = queue.popleft()
        result.append(node.value)

        if node.left is not None:
            queue.append(node.left)

        if node.right is not None:
            queue.append(node.right)

    return result


# ============================================================================
# Binary Search Tree
# ============================================================================


def bst_search(root: TreeNode | None, value: Any) -> TreeNode | None:
    """
    Search for a value in a Binary Search Tree.

    Returns
    -------
    TreeNode | None
        The node containing the value, or None if not found.
    """
    _validate_tree_root(root)
    _validate_bst_value(value)

    current = root

    while current is not None:
        if value == current.value:
            return current

        if value < current.value:
            current = current.left
        else:
            current = current.right

    return None


def bst_insert(root: TreeNode | None, value: Any) -> TreeNode:
    """
    Insert a value into a Binary Search Tree.

    Duplicate values are ignored.

    The existing tree is modified in place.

    Returns
    -------
    TreeNode
        The root of the resulting BST.
    """
    _validate_tree_root(root)
    _validate_bst_value(value)

    if root is None:
        return TreeNode(value)

    current = root

    while True:
        if value == current.value:
            return root

        if value < current.value:
            if current.left is None:
                current.left = TreeNode(value)
                return root

            current = current.left

        else:
            if current.right is None:
                current.right = TreeNode(value)
                return root

            current = current.right


def bst_delete(root: TreeNode | None, value: Any) -> TreeNode | None:
    """
    Delete a value from a Binary Search Tree.

    Handles:
    - leaf nodes,
    - nodes with one child,
    - nodes with two children,
    - deletion of the root,
    - missing values.

    For a node with two children, the inorder successor is used.

    Returns
    -------
    TreeNode | None
        The root of the resulting BST.
    """
    _validate_tree_root(root)
    _validate_bst_value(value)

    def delete_node(
        node: TreeNode | None,
        target: Any,
    ) -> TreeNode | None:
        if node is None:
            return None

        if target < node.value:
            node.left = delete_node(node.left, target)
            return node

        if target > node.value:
            node.right = delete_node(node.right, target)
            return node

        # Node found.

        # No left child.
        if node.left is None:
            return node.right

        # No right child.
        if node.right is None:
            return node.left

        # Two children:
        # find inorder successor = minimum node in right subtree.
        successor = node.right

        while successor.left is not None:
            successor = successor.left

        node.value = successor.value
        node.right = delete_node(node.right, successor.value)

        return node

    return delete_node(root, value)


# ============================================================================
# Min Heap
# ============================================================================


class MinHeap:
    """
    Simple binary min-heap implementation.

    The smallest element is always available at the root.

    Supported operations:
    - push
    - pop
    - peek
    - is_empty
    - size
    """

    def __init__(self, values: list[Any] | None = None) -> None:
        self._data: list[Any] = []

        if values is not None:
            if not isinstance(values, list):
                raise ValueError("values must be a list.")

            self._data = values.copy()
            self._build_heap()

    def _build_heap(self) -> None:
        """Transform the internal list into a valid min-heap."""
        for index in range(len(self._data) // 2 - 1, -1, -1):
            self._sift_down(index)

    def _sift_up(self, index: int) -> None:
        """Move an element upward until the heap property is restored."""
        while index > 0:
            parent = (index - 1) // 2

            if self._data[parent] <= self._data[index]:
                break

            self._data[parent], self._data[index] = (
                self._data[index],
                self._data[parent],
            )

            index = parent

    def _sift_down(self, index: int) -> None:
        """Move an element downward until the heap property is restored."""
        size = len(self._data)

        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index

            if left < size and self._data[left] < self._data[smallest]:
                smallest = left

            if right < size and self._data[right] < self._data[smallest]:
                smallest = right

            if smallest == index:
                break

            self._data[index], self._data[smallest] = (
                self._data[smallest],
                self._data[index],
            )

            index = smallest

    def push(self, value: Any) -> None:
        """Insert a value into the heap."""
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> Any:
        """
        Remove and return the smallest value.

        Raises
        ------
        IndexError
            If the heap is empty.
        """
        if not self._data:
            raise IndexError("cannot pop from an empty heap.")

        if len(self._data) == 1:
            return self._data.pop()

        minimum = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)

        return minimum

    def peek(self) -> Any:
        """
        Return the smallest value without removing it.

        Raises
        ------
        IndexError
            If the heap is empty.
        """
        if not self._data:
            raise IndexError("cannot peek at an empty heap.")

        return self._data[0]

    def is_empty(self) -> bool:
        """Return True if the heap is empty."""
        return len(self._data) == 0

    def size(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._data)

    def to_list(self) -> list[Any]:
        """Return a copy of the internal heap representation."""
        return self._data.copy()


# ============================================================================
# Priority Queue
# ============================================================================


class PriorityQueue:
    """
    Min-priority queue implemented with a binary heap.

    Lower priority numbers are removed first.

    Each item is stored as:
        (priority, value)

    Example:
        queue.push("urgent", priority=1)
        queue.push("normal", priority=5)

    "urgent" will be removed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[Real, int, Any]] = []
        self._counter = 0

    def _sift_up(self, index: int) -> None:
        """Restore heap property upward."""
        while index > 0:
            parent = (index - 1) // 2

            if self._heap[parent][:2] <= self._heap[index][:2]:
                break

            self._heap[parent], self._heap[index] = (
                self._heap[index],
                self._heap[parent],
            )

            index = parent

    def _sift_down(self, index: int) -> None:
        """Restore heap property downward."""
        size = len(self._heap)

        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index

            if (
                left < size
                and self._heap[left][:2] < self._heap[smallest][:2]
            ):
                smallest = left

            if (
                right < size
                and self._heap[right][:2] < self._heap[smallest][:2]
            ):
                smallest = right

            if smallest == index:
                break

            self._heap[index], self._heap[smallest] = (
                self._heap[smallest],
                self._heap[index],
            )

            index = smallest

    def push(self, value: Any, priority: Real) -> None:
        """
        Add an item to the priority queue.

        Lower priority values are served first.
        """
        if isinstance(priority, bool) or not isinstance(priority, Real):
            raise ValueError("priority must be a real number.")

        item = (priority, self._counter, value)
        self._counter += 1

        self._heap.append(item)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> Any:
        """
        Remove and return the highest-priority item.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        if not self._heap:
            raise IndexError("cannot pop from an empty priority queue.")

        if len(self._heap) == 1:
            return self._heap.pop()[2]

        item = self._heap[0]
        self._heap[0] = self._heap.pop()
        self._sift_down(0)

        return item[2]

    def peek(self) -> Any:
        """
        Return the highest-priority item without removing it.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        if not self._heap:
            raise IndexError("cannot peek at an empty priority queue.")

        return self._heap[0][2]

    def is_empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._heap) == 0

    def size(self) -> int:
        """Return the number of queued items."""
        return len(self._heap)

