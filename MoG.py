# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import random as rd

import numpy as np
import time
import sys
import os

from scipy.stats import multivariate_normal

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        if row[column].strip() == '?':
            row[column] = 0
        else:
            row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    for row in dataset:
        if row[column].strip() == '?':
            row[column] = 0
        else:
            row[column] = int(row[column].strip())

# Convert string column to integer
def str_column_to_cls(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


seed(2)
filename = 'breast-cancer-wisconsin.data'
dataset = load_csv(filename)
for i,rw in enumerate(dataset):
    rw = rw[1:]
    dataset[i] = rw

# convert string attributes to integers
for i in range(0, len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_cls(dataset, len(dataset[0]) - 1)

y22=np.array([np.array(xi) for xi in dataset])
yt = []
yf = []


# print(y)
for z11 in y22:
    if(z11[-1] == 0):
        yt.append(list(z11))
    else:
        yf.append(list(z11))

rd.shuffle(yt)
data1 = np.array(yt[0:400])
data = np.zeros((len(data1), 9))
for i,rw in enumerate(data1):
    rw = rw[0:-1]
    data[i] = rw


test_set = yt[400:]
test_set.extend(yf)

n = len(data)
k = int(sys.argv[1])
means = np.zeros((k,9))
covar = np.zeros((k,9,9))
gms = np.zeros((n,k))
ws = np.zeros(k)

ini = KMeans(n_clusters=k).fit(data)
means = ini.cluster_centers_
lab = ini.labels_
cnt = np.zeros(k)
for j in range(0, n):
    cl = lab[j]
    cnt[cl] = cnt[cl]+1
    covar[cl] = covar[cl] + np.outer(data[j],data[j])
for k1 in range(k):
    covar[k1] = covar[k1]/float(cnt[k1])
    ws[k1] = cnt[k1]/float(n)


for i in range(0, 7):
    for j in range(0, n):
        sm = 0
        for k1 in range(0, k):
            var = multivariate_normal(mean=means[k1], cov=covar[k1])
            gms[j][k1] = ws[k1]*var.pdf(data[j])
            sm += gms[j][k1]
        for k1 in range(0, k):
            gms[j][k1] = gms[j][k1]/float(sm)
    for k1 in range(0, k):
        m1 = 0
        means[k1] = np.zeros(9)
        covar[k1] = np.zeros((9,9))
        ws[k1] = 0
        for j in range(0, n):
            means[k1] = means[k1] + (gms[j][k1]*data[j])
            covar[k1] = covar[k1] + (gms[j][k1]*np.outer(data[j], data[j]))
            m1 = m1 + gms[j][k1]
        means[k1] = means[k1]/m1
        covar[k1] = covar[k1]/m1
        ws[k1] = m1/float(n)

predict = []
actual = [row[-1] for row in dataset]
thresh = 1
for tdat in data:
    prob = 0
    for k1 in range(k):
        var = multivariate_normal(mean=means[k1], cov=covar[k1])
        prob = prob + ws[k1]*var.pdf(tdat)
    thresh = min(thresh, prob)
for dat in dataset:
    prob = 0
    for k1 in range(k):
        var = multivariate_normal(mean=means[k1], cov=covar[k1])
        prob = prob + ws[k1]*var.pdf(dat[:-1])
        if prob >= 0.5*thresh:
            predict.append(0)
        else:
            predict.append(1)
accuracy = accuracy_metric(actual, predict)
print(ws[0])
print(means[0])
print(covar[0])
print(predict)
print(actual)
scores = []
scores.append(accuracy)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
