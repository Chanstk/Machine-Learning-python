# coding=utf-8
from math import log
import operator
#决策树实例

#计算香农熵
def calcShannonEnt(dataSet):
    #样本集合的个数
    numEntries = len(dataSet)
    #构造一个字典用来对样本中每个类进行计数
    labelCounts = {}    
    for featVec in dataSet:
        #获取每个样本的类标签
        curLabel = featVec[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1
    #初始化香农熵
    shannoEnt = 0.0
    #利用公式计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt

#筛选出样本，要求：样本特征为axis（索引值），特征axis取值为value
#筛选出的样本不含特征axis
def splitDataSet(dataSet, axis, value):
    #目标样本
    retDataSet = []
    #遍历每个样本， 把符合要求的样本加入目标样本集合中（retDataSet）
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceVec = featVec[:axis]
            reduceVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceVec)
    return retDataSet

#初始化一个样本集合
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    #表示特征意义
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#选择筛选后香农熵最大的特征进行样本分割
def chooseBestFeatureToSplit(dataSet):
    #除去类别标签，减去1
    numFeatures = len(dataSet[0]) - 1
    #未筛选之前的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #香农熵增益，初始化为0
    bestInfoGain = 0.0
    #最好的特征，按该特征进行样本筛选，筛选后的样本香农熵最大，初始化为-1
    bestFeature = -1
    #对每个特征进行筛选后的样本香农熵计算，选取香农熵增益最大的特征
    for i in range(numFeatures):
        #将第i个特征的所有样本取值罗列出来
        featList = [example[i] for example in dataSet]
        #去除重复的取值
        uniqueVals = set(featList)
        #香农熵初始化为0
        newEntropy = 0.0
        #计算按特征筛选出的样本的香农熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        #计算香农熵增益
        infoGain = baseEntropy - newEntropy
        print "splited by No.%d feature ,infoGain is %f" % (i, infoGain)
        #选出最好的特征
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    print "choose No.%d feature to split the dataSet" % i
    return bestFeature


def mayjorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassList = sorted(classCount.iteritems(),
                             key=operator.itemgetter(1), reverse=True)
    return sortedClassList[0][0]

#构造决策树
def createTree(dataSet, labels):
    #选取样本的类标签
    classList = [example[-1] for example in dataSet]
    #如果样本同类，返回类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果样本没有特征可以筛选，返回数量最多的类标签
    if len(dataSet[0]) == 1:
        return mayjorityCnt(classList)
    #选取最好的特征（索引）筛选样本
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #根据索引获取特征的标签
    bestFeatLabel = labels[bestFeat]
    #将特征标签删除
    del(labels[bestFeat])
    myTree = {bestFeatLabel: {}}
    #对于最好特征的每个取值，进行递归建树
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(\
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def main():
    Mymat, labels = createDataSet()
    tree = createTree(Mymat, labels)
    print tree

if __name__ == '__main__':
    main()

