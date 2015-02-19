Geting Started
==============

Ubuntu:

sudo apt-get install python-sklearn

sudo apt-get install python-numpy

Otherwise:

Somehow install scikit-learn

Somehow install numpy

Data Layout
===============
data.csv is intended to be edited in a spreadsheet. Commas are the only
separator character. When data.csv is read by the python script, lines that
start with ";;" are ignored, so it is important to preserve those in the
csv file for things like column headers.

Right now, there are two classes that we want to predict: whether a player
made it into the last half of the game, or not. That is the final column of
the csv file.

How To Run
==========
`easy.py data.csv test.csv`

This runs the training. It then reports how many of the training vectors
were classified correctly. Finally it reports how much of the test data was
classified correctly.
