# Assignment by Dorothea Kam and Dominik Schuberth
# Link to github: https://github.com/dominik-schuberth/Assignment3_Kam_Schuberth.git

# For this code's idea please see the flowchart attached.
# We created an additional outputfile: one csv as required, and one xlsx because the csv-file values are all in column when opened with excel.
# The xlsx file shows the values in separate values.

# it creates 2 Outputfiles (read below why)

# https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42import pandas as pd

import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter as xlsx
import csv


dataset = pd.read_csv('input.csv',delimiter=";", header = None, skiprows=2)
X = np.array(dataset)


init_centroids = rd.sample(range(0, len(dataset)), 3)


centroids = []
for i in init_centroids:
    centroids.append(dataset.loc[i])

centroids = np.array(centroids)
firstcentroids = centroids
print("centroids:", centroids)


#   calculate the manhattan distance
def CalcDistance(x1, x2):
    return sum(abs(x1 - x2))


#   calculate the closest centroid
def closestCentroid(centroids, X):
    nearestCentroid = []
    for i in X:
        distance = []
        for j in centroids:
            distance.append(CalcDistance(i, j))
            print("distance:", distance)
        nearestCentroid.append(np.argmin(distance)+1)  # determines the lowest distance to the 3 centroids; we want 1,2,3 and not 0,1,2 --> so +1
    return nearestCentroid


#   calculate the new centroid
def calc_new_centroids(clusters, X):
    newCentroids = []
    newDataset = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])], axis=1)
    print("newdata: \n", newDataset)
    print("newdatacluster: \n", newDataset['cluster'])


    for x in set(newDataset['cluster']):
        currentCluster = newDataset[newDataset['cluster'] == x][newDataset.columns[:-1]]
        print("currentcluster:", currentCluster)
        mean = currentCluster.mean(axis=0)
        print("mean:", mean)
        newCentroids.append(mean)
    return newCentroids


get_centroids = closestCentroid(centroids, X)
centroids = calc_new_centroids(get_centroids, X)
print(centroids)
changes = np.array([[0, 0], [0, 0], [0, 0]])
number_of_iterations = 0
equal_arrays = False

while not equal_arrays and number_of_iterations < 20:
    get_centroids = closestCentroid(centroids, X)
    centroids = calc_new_centroids(get_centroids, X)
    neu = np.array(centroids)

    comparison = neu == changes
    equal_arrays = comparison.all()
    print("Same or not?")
    print(equal_arrays)
    plt.figure(number_of_iterations)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='red')
    changes = np.array(centroids)
    plt.scatter(X[:, 0], X[:, 1], color='blue')
    plt.show()
    print("Runned: ")
    print(number_of_iterations)
    number_of_iterations += 1


#The CSV Output2file: when opened using excel, the values are in 1 column...
#thats why we created another Output version with xlsx, where the values in excel are in sepearte columns :)
import csv
with open('Output2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
#   [number of clusters]
    writer.writerow(["3"])
#   [list of the positions of the seed points]
    writer.writerow([centroids[0][0], centroids[0][1]])
    writer.writerow([centroids[1][0], centroids[1][1]])
    writer.writerow([centroids[2][0], centroids[2][1]])

#   [number of iterations]
    writer.writerow([number_of_iterations])

#   [number of data entries] [number of dimensions]
    writer.writerow([18,2])

    for i in range(18):
        writer.writerow([get_centroids[i],X[i][0],X[i][1]])

workbook = xlsx.Workbook('Output.xlsx')
worksheet = workbook.add_worksheet("Output")
row = 0
column = 0

#   [number of clusters]
worksheet.write(row, column, 3)

#   [list of the positions of the seed points]
worksheet.write(row+1, column, centroids[0][0])
worksheet.write(row+1, column+1, centroids[0][1])
worksheet.write(row+2, column, centroids[1][0])
worksheet.write(row+2, column+1, centroids[1][1])
worksheet.write(row+3, column, centroids[2][0])
worksheet.write(row+3, column+1, centroids[2][1])

#   [number of iterations]
worksheet.write(row+4, column, number_of_iterations)

#   [number of data entries] [number of dimensions]
worksheet.write(row+5, column, 18)
worksheet.write(row+5, column+1, 2)

for i in range(18):
    worksheet.write(row+6, column+1, X[i][0])
    row += 1

row = 6
for i in range(18):
    worksheet.write(row, column + 2, X[i][1])
    row += 1

row = 6
for i in range(18):
    worksheet.write(row, column, get_centroids[i])
    row += 1

workbook.close()






