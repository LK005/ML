import numpy as np
from os import listdir
import operator
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    with open(filename) as fp:
        for i in range(32):
            lineStr = fp.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i:] = img2vector(u'trainingDigits\%s'%fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(u'testDigits\%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,8)
        print("The classifier came back with:%d,The real answer is:%d"%(classifierResult,classNumStr))
        if (classifierResult != classNumStr):errorCount += 1
    print("\nThe total number of errors is :%d"%errorCount)
    print("\nThe total error rate is:%f"%(errorCount/float(mTest)))

handwritingClassTest()
# print(img2vector(r"C:\Users\cat\Desktop\MLLearning\ch02\KNNhand\trainingDigits\0_5.txt")[0,32:63])