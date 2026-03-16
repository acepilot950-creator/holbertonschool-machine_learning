#!/usr/bin/env python3
"""Decision tree implementation with Gini split criterion."""

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
        """Add prefix to the left child subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix to the right child subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return a string representation of the node subtree."""
        if self.is_root:
            text = "root [feature={}, threshold={}]".format(
                self.feature, self.threshold
            )
        else:
            text = "-> node [feature={}, threshold={}]".format(
                self.feature, self.threshold
            )

        text += "\n"
        text += self.left_child_add_prefix(str(self.left_child))
        text += self.right_child_add_prefix(str(self.right_child))
        return text[:-1]

    def get_leaves_below(self):
        """Return the list of all leaves below this node."""
        return (
            self.left_child.get_leaves_below() +
            self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """Recursively compute lower and upper bounds below this node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        self.left_child.lower = self.lower.copy()
        self.left_child.upper = self.upper.copy()
        self.left_child.lower[self.feature] = self.threshold

        self.right_child.lower = self.lower.copy()
        self.right_child.upper = self.upper.copy()
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Compute and store the indicator function of the node."""

        def is_large_enough(x):
            return np.all(
                np.array([
                    np.greater(x[:, key], self.lower[key])
                    for key in list(self.lower.keys())
                ]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([
                    np.less_equal(x[:, key], self.upper[key])
                    for key in list(self.upper.keys())
                ]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Predict the value for one sample."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


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
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """Return the leaf itself in a list."""
        return [self]

    def update_bounds_below(self):
        """Do nothing for leaves."""
        pass

    def pred(self, x):
        """Predict the value for one sample."""
        return self.value


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
        """Return the string representation of the tree."""
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """Return the list of all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update lower and upper bounds for all nodes and leaves."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Compute and store the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([
                leaf.indicator(A) * leaf.value
                for leaf in leaves
            ]),
            axis=0
        )

    def pred(self, x):
        """Predict the value for one sample."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return the minimum and maximum of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Return a random feature and threshold for splitting."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree on the given dataset."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(
            self.target,
            dtype="bool"
        )

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                """  Training finished.
    - Depth                     : {}
    - Number of nodes           : {}
    - Number of leaves          : {}
    - Accuracy on training data : {}""".format(
                    self.depth(),
                    self.count_nodes(),
                    self.count_nodes(only_leaves=True),
                    self.accuracy(self.explanatory, self.target)
                )
            )

    def fit_node(self, node):
        """Recursively grow the tree below a node."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = np.logical_and(
            node.sub_population,
            np.greater(
                self.explanatory[:, node.feature],
                node.threshold
            )
        )

        right_population = np.logical_and(
            node.sub_population,
            np.less_equal(
                self.explanatory[:, node.feature],
                node.threshold
            )
        )

        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(
                node,
                left_population
            )
        else:
            node.left_child = self.get_node_child(
                node,
                left_population
            )
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(
                node,
                right_population
            )
        else:
            node.right_child = self.get_node_child(
                node,
                right_population
            )
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create and return a leaf child."""
        values, counts = np.unique(
            self.target[sub_population],
            return_counts=True
        )
        value = values[np.argmax(counts)]
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

    def accuracy(self, test_explanatory, test_target):
        """Return the prediction accuracy on a dataset."""
        return np.sum(
            np.equal(
                self.predict(test_explanatory),
                test_target
            )
        ) / test_target.size

    def possible_thresholds(self, node, feature):
        """Return all possible thresholds for one feature."""
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population]
        )
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Return best threshold and corresponding Gini score."""
        x = self.explanatory[:, feature][node.sub_population]
        y = self.target[node.sub_population]
        thresholds = self.possible_thresholds(node, feature)
        classes = np.unique(y)

        left_mask = np.greater(
            x[:, None],
            thresholds[None, :]
        )
        class_mask = np.equal(
            y[:, None],
            classes[None, :]
        )

        left_f = np.logical_and(
            left_mask[:, :, None],
            class_mask[:, None, :]
        )

        right_f = np.logical_and(
            np.logical_not(left_mask)[:, :, None],
            class_mask[:, None, :]
        )

        left_counts = np.sum(left_f, axis=0).astype(float)
        right_counts = np.sum(right_f, axis=0).astype(float)

        left_sizes = np.sum(left_counts, axis=1)
        right_sizes = np.sum(right_counts, axis=1)
        total_size = left_sizes + right_sizes

        left_proba_sq = np.divide(
            left_counts,
            left_sizes[:, None],
            out=np.zeros_like(left_counts),
            where=left_sizes[:, None] != 0
        ) ** 2

        right_proba_sq = np.divide(
            right_counts,
            right_sizes[:, None],
            out=np.zeros_like(right_counts),
            where=right_sizes[:, None] != 0
        ) ** 2

        left_gini = 1 - np.sum(left_proba_sq, axis=1)
        right_gini = 1 - np.sum(right_proba_sq, axis=1)

        gini_split = (
            (left_sizes / total_size) * left_gini +
            (right_sizes / total_size) * right_gini
        )

        i = np.argmin(gini_split)
        return thresholds[i], gini_split[i]

    def Gini_split_criterion(self, node):
        """Return best feature and threshold using Gini impurity."""
        x = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(x[:, 1])
        return i, x[i, 0]
