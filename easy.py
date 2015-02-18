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

DATA_FILE = sys.argv[1]

features = []
classes = []

with open(DATA_FILE, 'r') as f:
  for line in f:
    if line.startswith(";;"): continue
    fields = line.split(",")
    name = tuple(fields[1:3])
    #print name
    vector = [float(e) for e in fields[3:]]
    (f, c) = vector[:-1], vector[-1]
    features += [f]
    classes += [c]

#print "features:",features
#print "classes:", classes

## FIXME: stretch / shift features so mean and variance are equal?

clf = svm.SVC(kernel='rbf')
clf.fit(features, classes)
print "What fraction of training data was scored properly?", clf.score(features, classes)

#for result in clf.predict(features):
#  print result

#print clf.predict([[0, 0], [5, 5]])
