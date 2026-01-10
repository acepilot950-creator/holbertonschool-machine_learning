#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    # Set random seed for reproducibility
    np.random.seed(5)

    # Generate random student grades
    student_grades = np.random.normal(68, 15, 50)

    # Create the figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot the histogram with bins of size 10 and black edges
    plt.hist(student_grades,
             bins=range(0, 101, 10),
             edgecolor='black')

    # Label the axes
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')

    # Set the title of the plot
    plt.title('Project A')

    # Set x-axis ticks every 10 units
    plt.xticks(range(0, 101, 10))

    # Display the plot
    plt.show()
