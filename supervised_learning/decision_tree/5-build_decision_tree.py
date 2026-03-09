#!/usr/bin/env python3
"""Module for building a basic decision tree structure."""

import numpy as np


class Node:
    """Class that represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
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
        """Return the maximum depth below the current node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below the current node."""
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
        """Add the prefix for the left child representation."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text[:-1]

    def right_child_add_prefix(self, text):
        """Add the prefix for the right child representation."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text[:-1]

    def __str__(self):
        """Return the string representation of the node."""
        if self.is_root:
            text = "root [feature={}, threshold={}]".format(
                self.feature, self.threshold
            )
        else:
            text = "-> node [feature={}, threshold={}]".format(
                self.feature, self.threshold
            )

        left_text = self.left_child_add_prefix(str(self.left_child))
        right_text = self.right_child_add_prefix(str(self.right_child))
        return text + "\n" + left_text + "\n" + right_text

    def get_leaves_below(self):
        """Return the list of all leaves below the current node."""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Update bounds dictionaries for all nodes below the current node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            if child is self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Compute and store the indicator function of the node."""

        def is_large_enough(x):
            return np.all(np.array([
                np.greater(x[:, key], self.lower[key])
                for key in self.lower.keys()
            ]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([
                np.less_equal(x[:, key], self.upper[key])
                for key in self.upper.keys()
            ]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )


class Leaf(Node):
    """Class that represents a leaf of a decision tree."""

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
        """Count the leaf node."""
        return 1

    def __str__(self):
        """Return the string representation of the leaf."""
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """Return the leaf as a single-element list."""
        return [self]

    def update_bounds_below(self):
        """Leaf has no children, so nothing to update."""
        pass


class Decision_Tree:
    """Class that represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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
        """Count the number of nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Return the list of all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def __str__(self):
        """Return the string representation of the tree."""
        return self.root.__str__() + "\n"
