#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))

    # Histogram: bin width = 10, black edges
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Match reference axes and ticks
    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))

    plt.ylim(0, 30)
    plt.yticks(range(0, 31, 5))

    plt.show()
