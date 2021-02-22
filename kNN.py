from numpy import *
import operator
import numpy as np

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


# this function runs the kNN on a piece of data inX
def classify0(inX, dataSet, labels, k):             # take the input vector inX, the full matrix of training examples, the labels, and k (positive integer)
    dataSetSize = dataSet.shape[0]                  # set to the number of rows in the dataset (group)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # repeat inX vector by the length of the dataSet group in rows
                                                    # then subtract the dataSet. This creates the difference matrix
    sqDiffMat = diffMat**2                          # square each difference
    sqDistances = sqDiffMat.sum(axis=1)             # add up the values in each row
    distances = sqDistances**0.5                    # distance formula ()^0.5
    sortedDistIndices = distances.argsort()         # each value gets assigned its position from lowest to greatest with argsort()
    
    classCount = {}
    for i in range(k):                                                  # vote 'k' times
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    # decompose classCount dictionary into list of tuples,
    # sort the tuples by second item in the tuple (itemgetter(1))
    # sorted largest to smallest with reverse = True
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    
    return sortedClassCount[0][0]

group, labels = createDataSet()

print(classify0([2,1], group, labels, 4))


