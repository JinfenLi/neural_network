# LVQ for the Ionosphere Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd


# Load a CSV file
def load_csv(filename):
    dataset = pd.read_csv(filename).astype(float).values
    return dataset


# Convert string column to float
# def str_column_to_float(dataset, column):
#     for row in dataset:
#         row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
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
def evaluate_algorithm(dataset,n_epochs, n_codebooks, learn_rate):
    # folds = cross_validation_split(dataset, n_folds)

    scores = list()
    train_set = dataset
    test_set = dataset

    for e in range(n_epochs):
        print("epoch {0}",e)
        train_set = dataset
        test_set = dataset
        codebooks = train_codebooks(train_set, n_codebooks, learn_rate, e)
        predicted,codebooks = learning_vector_quantization(train_set, test_set, n_codebooks, learn_rate,e,codebooks)
        actual = [row[-1] for row in dataset]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        print(accuracy)
    # for fold in folds:
    #     train_set = list(folds)
    #     print(fold)
    #     train_set.remove(fold)
    #     train_set = sum(train_set, [])
    #     test_set = list()
    #     for row in fold:
    #         row_copy = list(row)
    #         test_set.append(row_copy)
    #         row_copy[-1] = None
    #     predicted = algorithm(train_set, test_set, *args)
    #     actual = [row[-1] for row in fold]
    #     accuracy = accuracy_metric(actual, predicted)
    #     scores.append(accuracy)
    return scores


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
    # multiple codebooks, one row
    distances = list()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda tup: tup[1])
    # return that codebook
    return distances[0][0]


# Make a prediction with codebook vectors
def predict(codebooks, test_row):
    bmu = get_best_matching_unit(codebooks, test_row)
    return bmu[-1]


# Create a random codebook vector
def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    # print(codebook)
    # only one codebook each time
    return codebook


# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epoch):
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    import numpy as np
    # print(np.array(codebooks).shape)
    # for epoch in range(epochs):
    rate = lrate * (1.0 - (epoch / float(n_epochs)))
    for row in train:
        bmu = get_best_matching_unit(codebooks, row)
        for i in range(len(row) - 1):
            error = row[i] - bmu[i]
            if bmu[-1] == row[-1]:
                bmu[i] += rate * error
            else:
                bmu[i] -= rate * error
            # print(error)
    return codebooks


# LVQ Algorithm
def learning_vector_quantization(train_set, test_set, n_codebooks, learn_rate,epoch,codebooks):

    predictions = list()
    for row in test_set:
        output = predict(codebooks, row)
        predictions.append(output)
    return predictions,codebooks


# Test LVQ on Ionosphere dataset
seed(1)
# load and prepare data
filename = 'heart.csv'
dataset = load_csv(filename)
# for i in range(len(dataset[0]) - 1):
#     str_column_to_float(dataset, i)
# convert class column to integers
# str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 5
learn_rate = 0.5
n_epochs = 50
n_codebooks = 20
import time
starttime = time.time()
scores = evaluate_algorithm(dataset,n_epochs, n_codebooks, learn_rate)
endtime = time.time()
dtime = endtime - starttime

print("used time：%.8s s" % dtime)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))