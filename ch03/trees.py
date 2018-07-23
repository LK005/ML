from math import log
import operator
import pickle
def createData():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannnonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannnonEnt -= prob*log(prob,2)
    return shannnonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])
            # print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方法
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1               #判定当前数据集包含多少特征属性
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) #集合
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
# 多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
# 创建树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(classList) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
# 获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth    
def retrieveTree(i):
    listOfTree = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},{'no surfacing': {0: 'no', 1: {'flippers': {0:{'head':{0: 'no', 1: 'yes'}},1:'no'}}}}]
    return listOfTree[i]
# 使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
# 使用pickle模块存储决策树
def storeTree(inputTree,filename):
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename,):
    fr = open(filename,'rb')
    return pickle.load(fr)
# dataSet = [[1,1,'maybe'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
# print(calcShannonEnt(dataSet))
# # print(splitDataSet(dataSet,0,0))
# print(chooseBestFeatureToSplit(dataSet))
# myDat,labels =  createData()
# print(createTree(myDat,labels))

# print(retrieveTree(1))
# myTree = retrieveTree(0)
# myDat,labels = createData()
# print(classify(myTree,labels,[1,0]))
# print(classify(myTree,labels,[1,1]))
# print(type(list(myTree.keys())))
# print(getNumLeafs(myTree))
# print(getTreeDepth(myTree))

# print(type(myTree))
# print(str(myTree))
# storeTree(myTree,'classifierStorage.txt')

# print(grabTree('classifierStorage.txt'))

with open('lenses.txt','r') as fp:
    lenses = [line.strip().split('\t') for line in fp.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    myTree = createTree(lenses,lensesLabels)
    print(myTree)