# Error Analysis

## Overview

This directory contains implementations related to **error analysis** in machine learning models.

Error analysis helps evaluate how well a model performs by examining its predictions and identifying patterns in its mistakes. This process is essential for improving model performance and understanding weaknesses in the model.

One of the most commonly used tools for error analysis in classification tasks is the **confusion matrix**.

---

## Confusion Matrix

A confusion matrix is a table used to evaluate the performance of a classification model.

It compares:

- the **actual labels** (ground truth)
- the **predicted labels** (model outputs)

The matrix structure is:

| Actual \ Predicted | Class 0 | Class 1 | ... |
|-------------------|--------|--------|-----|
| Class 0 | Correct predictions | Errors | ... |
| Class 1 | Errors | Correct predictions | ... |

Rows represent the **true labels**, while columns represent the **predicted labels**.

---
