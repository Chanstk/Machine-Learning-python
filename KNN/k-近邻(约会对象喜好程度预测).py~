from numpy import *
import operator
#假设网站已对路人甲约会对象的数据进行了收集，放在datingTestSet.txt文件下，从左到右分别为：
#(1）约会对象每年获得飞行常客里数   （2)约会对象玩视频游戏所耗的时间百分比
#(3) 约会对象每周消费的冰淇淋公升数  (4) 路人甲对该约会对象的喜好程度（含不喜欢的人，有点喜欢，非常喜欢三个评价）
#本例程序用来预测路人甲对新约会对象的喜好程度，误差为5%


#分类器 inX（测试数据） dataSet（样本数据集） labels（样本的类别） k（选择距离最小的k个点）
def classify0(inX, dataSet, labels, k):
    #获取样本数据大小
    dataSetSize = dataSet.shape[0]
    #计算测试数据与各个样本数据之间的距离distances
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #distances进行排序，将索引值返回给sortedDistIndicies
    sortedDistIndicies = distances.argsort()
    classCount={} 
    #获取k个离测试数据最近的样本数据的labels，labels可能重复
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #将k个labels按个数进行排序，选择重复次数最大labels的返回
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#将文本数据转换成matrix
def FileToMatrix(fileName):
    f = open(fileName)
    arrayLines = f.readlines()
    numberOfLines = len(arrayLines)
    resultMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        resultMat[index, :] = listFromLine[0:3]
        #给不同label索引值
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        else:
            classLabelVector.append(1)
        index += 1
    return resultMat, classLabelVector

#由于不同的特征数值大小不一， 对结果会造成干扰，所以要对特征数值进行归一化 newValue = (oldValue - min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#测试分类器，选取10%的样本进行测试
def datingClassTest():
    hoRatio = 0.1
    #获取全部样本
    datingDateMat, datingLabels = FileToMatrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDateMat)
    m = normMat.shape[0]
    #选择前10%的样本数据
    numTestVecs = int(m*hoRatio)
    #错误分类样本数
    errorCount = 0
    for i in range(numTestVecs):
        #将用分类器获得的label与原样本中的label进行对比，错误的话计数加一
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],3)
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print "the total error rate is : %f" % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all' ,'in small doses' ,'in large doses']
    #随便输入一组测试数据
    percentTats = float(8.326976)
    ffMiles = float(40920)
    iceCream = float(0.953952)
    datingDataMat ,datingLabels = FileToMatrix('datingTestSet.txt')
    normMat ,ranges ,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles ,percentTats ,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges ,normMat ,datingLabels ,3)
    print "You will probably like this person:",resultList[classifierResult-1]

