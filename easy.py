#!/usr/bin/env python

## Ubuntu:
# sudo apt-get install python-sklearn
# sudo apt-get install python-numpy
## Otherwse:
# install scikit-learn
# install numpy


from sklearn import svm
import numpy as np
import sys

from csvdom import CSVDOM

TRAINING_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]

features = []
classes = []

print "Reading", TRAINING_FILE
trainingData = CSVDOM(TRAINING_FILE)
if trainingData.ok == False:
  print "Couldn't read training data"
  sys.exit(-1)

print "Reading", TEST_FILE
testData = CSVDOM(TEST_FILE)
if not testData.ok:
  print "Couldn't read test data"
  sys.exit(-1)

# Adjust input data
# Needed: don't normalize sparse data
# Optional:
#  Use PCA to make sure there is no linear correlation between components

assert len(testData.rows[0]) == len(trainingData.rows[0])

## Component-wise subtraction of the mean in vector entries

sparseness = np.array([[0,0,0,0,0,0]])


trainingVectors = np.matrix([[float(e) for e in row[3:-1]] for row in trainingData.rows])
trainingClasses = [int(row[-1]) for row in trainingData.rows]

testVectors = np.matrix([[float(e) for e in row[3:-1]] for row in testData.rows])
testClasses = [int(row[-1]) for row in testData.rows]

# Subtract out mean
mean = np.mean(trainingVectors, axis=0)
sig = np.std(trainingVectors, axis=0)

mean[sparseness == 1] = 0  # don't adjust for sparse data
sig[sparseness == 1] = 1.0 # don't adjust for sparse data
sig[sig == 0] = 1.0
trainingVectors -= mean
trainingVectors /= sig
testVectors -= mean
testVectors /= sig


print trainingVectors
print testVectors

clf = svm.SVC(kernel='poly', gamma=5)
clf.fit(trainingVectors, trainingClasses)
print "Correct training data:", clf.score(trainingVectors, trainingClasses)

print "Correct predictions:", clf.score(testVectors, testClasses)


