import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','A','B']
    return group,labels
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    print(dataSetSize)
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    print(diffMat)
    sqDiffMat = diffMat**2
    print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    print(sqDistances)
    distances = sqDistances**0.5
    print(distances)
    sortedDistIndicies = distances.argsort()
    print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    print(classCount)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]