# -*- coding = utf-8 -*-
# @Time: 29/03/2022 21:13
# @Author: Ziqi Han
# @Student ID: 201568748
# @File: test.py
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
    howMany = np.unique(clustersDisplay, return_counts=True)[1]
    print(howMany)
    # print("=====The optimal centroids is: \n", centroidsDisplay)
    return clustersDisplay,centroidsDisplay

def calculatePRF(dataSet,clustersDisplay,centroidsDisplay):

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

    # If the data was normalised, then print the distance measurement and
    # normalisation
    # if trueOrFalseNorm == True:
    #     print("\nFinal Results of K-Means Clustering with", distance_measure,
    #           "Distance Measurement and L2 Normalisation")
    # # Otherwise just print the distance measurement
    # else:
    #     print("\nFinal Results of K-Means Clustering with", distance_measure,
    #           "Distance Measurement")
    # Print the results
    print("\tP:", P, ", R:", R, ", F:", F)

    # Return the P, R and F values for plotting4
    return P, R, F


##############################################################################
"""
plotting allows the user to plot the results of the Precision (P), 
    Recall (R) and F-Scores (F) acquired from the K-Means clustering.

Formatting:
    k - list
    P - list
    R - list
    F - list
    distance_measure - string
    l2 - string

Returns:
    A line graph comparing the P, R and F across the number of clusters (k)

"""


def drawingDiagram(k, P, R, F):
    # Plot K against P
    plt.plot(K_list, P_list, label="Precision")
    # Plot K against R
    plt.plot(K_list, R_list, label="Recall")
    # Plot K against F
    plt.plot(K_list, F_list, label="F-Score")
    # Plot the title
    plt.title("K-Means Clustering in ")
    # Plot the x and y axis labels
    plt.xlabel('K value')
    plt.ylabel("Score")
    # Display the legend
    plt.legend()
    # Display the plot
    plt.show()


##############################################################################
# Question 1
clustersDisplay,centroidsDisplay = kMeans(rawData, 7)
P_list = []
R_list = []
F_list = []
K_list = []
for k in range(1,11):
    K_list.append(k)
    clustersDisplay, centroidsDisplay = kMeans(rawData, k)
    P,R,F=calculatePRF(dataSet,clustersDisplay,k)
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
    print(P_list)
    print(R_list)
    print(F_list)
drawingDiagram(k, P, R, F)
'''
# Questions 2-6
for question in range(2, 7):
    # Create an empty list for P, R, F and K
    P_list = []
    R_list = []
    F_list = []
    K_list = []
    # Create an empty string for the distance method
    distance_measure = ""

    # Question 2
    if question == 2:
        distance_measure = "Euclidean"
        normalisation = False
    # Question 3
    elif question == 3:
        distance_measure = "Euclidean"
        normalisation = True
    # Question 4
    elif question == 4:
        distance_measure = "Manhattan"
        normalisation = False
    # Question 5
    elif question == 5:
        distance_measure = "Manhattan"
        normalisation = True
    # Question 6
    else:
        distance_measure = "Cosine"
        normalisation = False
        
# Define k between 1 - 10
    for k in range(1, 11):
        # Append k to a list for plotting
        K_list.append(k)
        # Save the Precision, Recall and F-Scores
        P, R, F = kmeans_clustering(x, k, distance_measure, normalisation, True)
        # Append the Precision, Recall and F-Score to each list for plotting
        P_list.append(P)
        R_list.append(R)
        F_list.append(F)
    # If the data is normalised, edit the title to include 'and Normalisation'
    if normalisation:
        plotting(K_list, P_list, R_list, F_list, distance_measure, " and Normalisation")
    # If not normalised, then do not include additional title
    else:
        plotting(K_list, P_list, R_list, F_list, distance_measure, "")

'''
