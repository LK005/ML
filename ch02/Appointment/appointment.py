import numpy as np
import operator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','A','B']
    return group,labels
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
def file2matrix(filename):
    with open(filename) as fp:
        arrayOLines = fp.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = np.zeros((numberOfLines,3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat,classLabelVector
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print("The classifier came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount += 1.0
    print("the total error rate is: %f"%(errorCount/float(numTestVecs)))
# predict!
def classifyPerson():
    resultList = ['Not at all','In small doses','In large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent fliter miles earned per year?'))
    iceCream = float(input('Liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person:',resultList[classifierResult - 1])
    
classifyPerson()
# if __name__=='__main__':
#     datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#     # print(15.0*np.array(datingLabels))
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels),label="curve1")
#     datingLabels = np.array(datingLabels)
#     zhfont = FontProperties(fname='C:/Windows/Fonts/simsun.ttc',size=12)

#     # idx_1 = np.where(datingLabels==1)
#     # p1 = ax.scatter(datingDataMat[idx_1,0],datingDataMat[idx_1,1],marker = '*',color = 'r',label='1',s=10)
#     # idx_2 = np.where(datingLabels==2)
#     # p2 = ax.scatter(datingDataMat[idx_2,0],datingDataMat[idx_2,1],marker = 'o',color ='g',label='2',s=20)
#     # idx_3 = np.where(datingLabels==3)
#     # p3 = ax.scatter(datingDataMat[idx_3,0],datingDataMat[idx_3,1],marker = '+',color ='b',label='3',s=30)

#     type1_x = []
#     type1_y = []
#     type2_x = []
#     type2_y = []
#     type3_x = []
#     type3_y = []
#     for i in range(len(datingLabels)):
#         if datingLabels[i]==1:
#             type1_x.append(datingDataMat[i][0]) 
#             type1_y.append(datingDataMat[i][1])
#         if datingLabels[i]==2:
#             type2_x.append(datingDataMat[i][0]) 
#             type2_y.append(datingDataMat[i][1])
#         if datingLabels[i]==3:
#             type3_x.append(datingDataMat[i][0]) 
#             type3_y.append(datingDataMat[i][1])
#     p1 = ax.scatter(type1_x,type1_y,s=20,c='red')
#     p2 = ax.scatter(type2_x,type2_y,s=40,c='green')
#     p3 = ax.scatter(type3_x,type3_y,s=50,c='blue')

#     plt.xlabel(u'每年获取的飞行里程数',fontproperties=zhfont)
#     plt.ylabel(u'玩视频游戏所消耗的事件百分比',fontproperties=zhfont)
#     ax.legend((p1, p2, p3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2,prop=zhfont)
#     plt.show()