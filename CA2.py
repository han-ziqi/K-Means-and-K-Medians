# -*- coding = utf-8 -*-
# @Time: 21/03/2022 02:17
# @Author: Ziqi Han
# @Student ID: 201568748
# @File: CA2.py
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
# Unit L2 length to normalise data
rawDataL2Norm = rawData / np.linalg.norm(rawData)


def euclideanDistance(X, Y):
    # Return the Euclidean distance between X and Y
    return np.linalg.norm(X - Y)


def manhattanDistance(X, Y):
    return np.sum(np.abs(X - Y))


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
        for h in range(len(centroids)):
            distanceForNext[:, h] = np.linalg.norm(rawData - centroidsNext[h], axis=1)

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

    print("=====This is K-Means =====================")
    print("=====There are", k, "clusters.======= =========")
    print("=====The optimal after", feasibleNumber, "generations.=====")
    print("=====The optimal clusters is: \n", clustersDisplay)
    print("=====The optimal centroids is: \n", centroidsDisplay)
    return clustersDisplay, centroidsDisplay


def kMedians(rawData, k):
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
    feasible = manhattanDistance(centroidsNext, centroidsPrevious)
    feasibleNumber = 0

    # While-loop to optimise:
    while feasible != 0:
        distanceForNext = np.zeros([rawData.shape[0], k])
        feasibleNumber += 1
        for h in range(len(centroids)):
            distanceForNext[:, h] = np.sum(np.abs(rawData - centroidsNext[h]), axis=1)

        # Refreshing cluster
        clusters = np.argmin(distanceForNext, axis=1)

        # Switch Next into Previous and start next loop
        centroidsPrevious = np.copy(centroidsNext)
        for m in range(k):
            centroidsNext[m] = np.median(rawData[clusters == m], axis=0)
        # Repeat calculate optimal value for next loop
        feasible = manhattanDistance(np.array(centroidsNext), np.array(centroidsPrevious))

    # Copy and save results for display
    clustersDisplay = clusters
    centroidsDisplay = np.array(centroidsNext)

    # If the user does not want P, R and F to be calculated, then display
    # the results of the k-means clustering

    print("=====This is K-Medians ====================")
    print("=====There are ", k, "clusters.================")
    print("=====The optimal after", feasibleNumber, "generations.=====")
    print("=====The optimal clusters is: \n", clustersDisplay)
    print("=====The optimal centroids is: \n", centroidsDisplay)
    return clustersDisplay, centroidsDisplay


def calculatePRF(dataSet, clustersDisplay, centroidsDisplay):
    # Save the maximum index for each category for the P/R/F
    maxAnimalIndex = dataSet.index[dataSet['Category'] == 'animals'][-1]
    maxCountriesIndex = dataSet.index[dataSet['Category'] == 'countries'][-1]
    maxFruitIndex = dataSet.index[dataSet['Category'] == 'fruits'][-1]
    maxVeggiesIndex = dataSet.index[dataSet['Category'] == 'veggies'][-1]

    # Create objects of the index positioning of the different classes
    animalIndex = clustersDisplay[:maxAnimalIndex + 1]
    countriesIndex = clustersDisplay[maxAnimalIndex + 1:maxCountriesIndex + 1]
    fruitIndex = clustersDisplay[maxCountriesIndex + 1:maxFruitIndex + 1]
    veggiesIndex = clustersDisplay[maxFruitIndex + 1:maxVeggiesIndex + 1]

    # Initialising the parameter variables of confusion matrix
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    # A loop considering all animalIndex
    for i in range(len(animalIndex)):
        for j in range(len(animalIndex)):
            # TP and FN
            if j > i:
                if animalIndex[i] == animalIndex[j]:
                    TP += 1
                else:
                    FN += 1
        for j in range(len(countriesIndex)):
            # FP and TN
            if animalIndex[i] == countriesIndex[j]:
                FP += 1
            else:
                TN += 1
        for j in range(len(fruitIndex)):
            # FP and TN
            if animalIndex[i] == fruitIndex[j]:
                FP += 1
            else:
                TN += 1
        for j in range(len(veggiesIndex)):
            if animalIndex[i] == veggiesIndex[j]:
                FP += 1
            else:
                TN += 1
    # A loop considering all countriesIndex
    for i in range(len(countriesIndex)):
        for j in range(len(countriesIndex)):
            if j > i:
                if countriesIndex[i] == countriesIndex[j]:
                    TP += 1
                else:
                    FN += 1
        for j in range(len(fruitIndex)):
            if countriesIndex[i] == fruitIndex[j]:
                FP += 1
            else:
                TN += 1
        for j in range(len(veggiesIndex)):
            if countriesIndex[i] == veggiesIndex[j]:
                FP += 1
            else:
                TN += 1
    #  A loop considering all fruitIndex
    for i in range(len(fruitIndex)):
        for j in range(len(fruitIndex)):
            if j > i:
                if fruitIndex[i] == fruitIndex[j]:
                    TP += 1
                else:
                    FN += 1
        for j in range(len(veggiesIndex)):
            if fruitIndex[i] == veggiesIndex[j]:
                FP += 1
            else:
                TN += 1
    # A loop considering all veggiesIndex
    for i in range(len(veggiesIndex)):
        # For every row in veggiesIndex
        for j in range(len(veggiesIndex)):
            if j > i:
                if veggiesIndex[i] == veggiesIndex[j]:
                    TP += 1
                else:
                    FN += 1
    P = round((TP / (TP + FP)), 2)
    R = round((TP / (TP + FN)), 2)
    F = round((2 * (P * R) / (P + R)), 2)

    print("After this K, current P:", P, ", R:", R, ", F:", F)

    return P, R, F


def drawingDiagram(kList, PList, RList, FList, titleNumber):
    # Drawing P R F
    plt.plot(KList, PList, label="Precision")
    plt.plot(KList, RList, label="Recall")
    plt.plot(KList, FList, label="F-Score")
    if titleNumber == 1:
        plt.title("K-Means Clustering with K in range 1~10")
    elif titleNumber == 2:
        plt.title("K-Means Clustering with K in range 1~10 and L2 Normalisation")
    elif titleNumber == 3:
        plt.title("K-Medians Clustering with K in range 1~10")
    elif titleNumber == 4:
        plt.title("K-Medians Clustering with K in range 1~10 and L2 Normalisation")
    plt.xlabel('K value')
    plt.ylabel("Score")
    plt.legend()
    plt.show()

# ------------------------ Answer Zone -----------------------------------
# Q1
kMeans(rawData, 4)
# Q2
kMedians(rawData, 4)
'''
# Q3
PList = []
RList = []
FList = []
KList = []
for k in range(1, 11):
    KList.append(k)
    clustersDisplay, centroidsDisplay = kMeans(rawData, k)
    P, R, F = calculatePRF(dataSet, clustersDisplay, k)
    PList.append(P)
    RList.append(R)
    FList.append(F)
    print(PList)
    print(RList)
    print(FList)
drawingDiagram(KList, PList, RList, FList, 1)

# Q4
PList = []
RList = []
FList = []
KList = []
for k in range(1, 11):
    KList.append(k)
    clustersDisplay, centroidsDisplay = kMeans(rawDataL2Norm, k)
    P, R, F = calculatePRF(dataSet, clustersDisplay, k)
    PList.append(P)
    RList.append(R)
    FList.append(F)
    print(PList)
    print(RList)
    print(FList)
drawingDiagram(KList, PList, RList, FList, 2)
# Q5
PList = []
RList = []
FList = []
KList = []
for k in range(1, 11):
    KList.append(k)
    clustersDisplay, centroidsDisplay = kMedians(rawData, k)
    P, R, F = calculatePRF(dataSet, clustersDisplay, k)
    PList.append(P)
    RList.append(R)
    FList.append(F)
    print(PList)
    print(RList)
    print(FList)
drawingDiagram(KList, PList, RList, FList, 3)

# Q6
PList = []
RList = []
FList = []
KList = []
for k in range(1, 11):
    KList.append(k)
    clustersDisplay, centroidsDisplay = kMedians(rawDataL2Norm, k)
    P, R, F = calculatePRF(dataSet, clustersDisplay, k)
    PList.append(P)
    RList.append(R)
    FList.append(F)
    print(PList)
    print(RList)
    print(FList)
drawingDiagram(KList, PList, RList, FList, 4)
'''