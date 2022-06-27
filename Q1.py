# -*- coding = utf-8 -*-
# @Time: 31/03/2022 00:45
# @Author: Ziqi Han
# @Student ID: 201568748
# @File: Q1.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data from archive file, to avoid skipping header, set header = None
animals = pd.read_csv('animals', sep=" ", header=None)
countries = pd.read_csv('countries', sep=" ", header=None)
fruits = pd.read_csv('fruits', sep=" ", header=None)
veggies = pd.read_csv('veggies', sep=" ", header=None)

# Set a new label on the end of array
animals['Category'] = 'animals'
countries['Category'] = 'countries'
fruits['Category'] = 'fruits'
veggies['Category'] = 'veggies'

# Mix all data into a dataSet, then change the labels into numbers
dataSet = pd.concat([animals, countries, fruits, veggies], ignore_index=True)

# Change labels to serialized numbers starting from 0. 0 animals, 1 countries, 2 fruits, 3 veggies
labelNumber = pd.factorize(dataSet.Category)[0]
# Delete first line and "Category", pd.drop in default delete a row, to delete a line, should add "axis=1"
rawData = dataSet.drop([0, 'Category'], axis=1).values


def euclideanDistance(X, Y):
    # Return the Euclidean distance between X and Y
    return np.linalg.norm(X - Y)


def kMeans(rawData, k):
    # Initialise the first centroids, randomly generate k sets of numbers to produce k different clusters
    centroids = []
    ranClusters = np.random.randint(rawData.shape[0], size=k)
    # To avoid having the same number in k numbers, if happened just re-roll
    if len(ranClusters) > len(set(ranClusters)):
        ranClusters = np.random.randint(rawData.shape[0], size=k)
    for i in ranClusters:
        centroids.append(rawData[i])

    # Set Previous step and next step
    centroidsPrevious = np.zeros(np.shape(centroids))
    centroidsNext = np.copy(centroids)

    # Create a empty array to save & display results
    clusters = np.zeros(rawData.shape[0])
    # if there is a feasible step, count it
    feasible = euclideanDistance(centroidsNext, centroidsPrevious)
    feasibleNumber = 0

    # While-loop to optimise:
    while feasible != 0:
        distanceForNext = np.zeros([rawData.shape[0], k])
        feasibleNumber += 1
        for l in range(len(centroids)):
            distanceForNext[:, l] = np.linalg.norm(rawData - centroidsNext[l], axis=1)

        # Refreshing cluster
        clusters = np.argmin(distanceForNext, axis=1)

        # Switch Next into Previous and start next loop
        centroidsPrevious = np.copy(centroidsNext)
        for m in range(k):
            centroidsNext[m] = np.mean(rawData[clusters == m], axis=0)
        # Repeat calculate optimal value for next loop
        feasible = euclideanDistance(np.array(centroidsNext), np.array(centroidsPrevious))

    # Copy and save results for display
    clustersDisplay = clusters
    centroidsDisplay = np.array(centroidsNext)

    # If the user does not want P, R and F to be calculated, then display
    # the results of the k-means clustering

    print("=====There are", k, "clusters.================")
    print("=====The optimal after", feasibleNumber, "generations.=====")
    print("=====The optimal clusters is: \n", clustersDisplay)
    print("=====The optimal centroids is: \n", centroidsDisplay)



kMeans(rawData, 4)
