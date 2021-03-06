import matplotlib.pyplot as plt
from trees import *
decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy = parentPt,xycoords = 'axes fraction',xytext = centerPt,textcoords = 'axes fraction',va=  "center",ha = "center",bbox = nodeType,arrowprops = arrow_args)
# def createPlot():
#     fig = plt.figure(1,facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111,frameon = False)
#     plotNode(u'jucenode',(0.5,0.1),(0.1,0.5),decisionNode)
#     plotNode(u'yenode',(0.8,0.1),(0.3,0.8),leafNode)
#     # plotNode(u'yenode',(1,1),(1,1),leafNode)
#     plt.show()
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks = [])
    createPlot.ax1 = plt.subplot(111,frameon = False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
# createPlot()
with open('lenses.txt','r') as fp:
    lenses = [line.strip().split('\t') for line in fp.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    myTree = createTree(lenses,lensesLabels)
    print(myTree)
    createPlot(myTree)