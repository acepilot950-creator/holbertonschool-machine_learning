#!/usr/bin/env python3
"""Module for building and training a decision tree."""

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
            n_features = self.sub_population.size
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

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

    def pred(self, x):
        """Return the prediction for one individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


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
        """Leaf has no children."""
        pass

    def pred(self, x):
        """Return the prediction of the leaf."""
        return self.value


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

    def update_predict(self):
        """Compute and store the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(np.array([
            leaf.indicator(A) * leaf.value for leaf in leaves
        ]), axis=0)

    def pred(self, x):
        """Return the prediction for one individual."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return minimum and maximum of a numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Choose a random valid split for a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            vals = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(vals)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create and return a leaf child."""
        classes, counts = np.unique(
            self.target[sub_population], return_counts=True
        )
        value = classes[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create and return an internal node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fit a node and its children."""
        node.feature, node.threshold = self.split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        left_population = np.logical_and(
            node.sub_population,
            np.greater(feature_values, node.threshold)
        )
        right_population = np.logical_and(
            node.sub_population,
            np.less_equal(feature_values, node.threshold)
        )

        left_target = self.target[left_population]
        right_target = self.target[right_population]

        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(left_target).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(right_target).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                "Training finished.\n"
                f"Depth : {self.depth()}\n"
                f"Number of nodes : {self.count_nodes()}\n"
                f"Number of leaves : {self.count_nodes(only_leaves=True)}\n"
                f"Accuracy on training data : {self.accuracy(self.explanatory, self.target)}\n",
                end=""
            )

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy on a dataset."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size

    def __str__(self):
        """Return the string representation of the tree."""
        return self.root.__str__() + "\n"
