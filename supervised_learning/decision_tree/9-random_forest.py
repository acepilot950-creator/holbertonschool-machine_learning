#!/usr/bin/env python3
"""Random forest implementation based on decision trees."""

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
import numpy as np


class Random_Forest:
    """Represents a random forest classifier."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize the random forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict classes for all rows in explanatory."""
        predictions = np.array([
            pred(explanatory) for pred in self.numpy_preds
        ])
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x.astype(int))),
            axis=0,
            arr=predictions
        )

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the random forest on the given dataset."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            tree.fit(explanatory, target)
            self.numpy_preds.append(tree.predict)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            accuracies.append(tree.accuracy(tree.explanatory, tree.target))

        if verbose == 1:
            print(
                """  Training finished.
    - Mean depth                     : {}
    - Mean number of nodes           : {}
    - Mean number of leaves          : {}
    - Mean accuracy on training data : {}
    - Accuracy of the forest on td   : {}""".format(
                    np.array(depths).mean(),
                    np.array(nodes).mean(),
                    np.array(leaves).mean(),
                    np.array(accuracies).mean(),
                    self.accuracy(self.explanatory, self.target)
                )
            )

    def accuracy(self, test_explanatory, test_target):
        """Return the prediction accuracy on a dataset."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
