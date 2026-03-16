#!/usr/bin/env python3
"""Basic decision tree classes with printing support."""

import numpy as np


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
        """Initialize a node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth below this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node."""
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Add the display prefix for the left child subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |  " + line + "\n"
        return new_text[:-1]

    def right_child_add_prefix(self, text):
        """Add the display prefix for the right child subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "       " + line + "\n"
        return new_text[:-1]

    def __str__(self):
        """Return a string representation of the node and its children."""
        if self.is_root:
            node_text = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )
        else:
            node_text = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )

        left_text = self.left_child_add_prefix(str(self.left_child))
        right_text = self.right_child_add_prefix(str(self.right_child))

        return node_text + "\n" + left_text + "\n" + right_text


class Leaf(Node):
    """Represents a leaf of a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count the leaf as one node."""
        return 1

    def __str__(self):
        """Return a string representation of the leaf."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
        """Initialize the decision tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the whole tree."""
        return self.root.__str__()
