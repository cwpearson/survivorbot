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

assert len(testData.rows[0]) == len(trainingData.rows[0])

trainingVectors = [[float(e) for e in row[3:-1]] for row in trainingData.rows]
trainingClasses = [int(row[-1]) for row in trainingData.rows]

## FIXME: stretch / shift features so mean and variance are equal?

clf = svm.SVC(kernel='rbf')
clf.fit(trainingVectors, trainingClasses)
print "Correct training data:", clf.score(trainingVectors, trainingClasses)

testVectors = [[float(e) for e in row[3:-1]] for row in testData.rows]
testClasses = [int(row[-1]) for row in testData.rows]

predictions = clf.predict(testVectors)

correct = 0.0
for i in range(len(predictions)):
  if predictions[i] == testClasses[i]:
    correct += 1.0
print "Correct test data:", correct / len(testClasses)

