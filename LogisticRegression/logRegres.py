# coding  = utf-8
from numpy import *

#加载testSet的样本，返回数据列表与标签列表
def loadDataSet():
    dataMat = []
    labelMat = []
    f = open('testSet.txt')
    for line in f.readlines():
        lineArr = line.strip().split()
        #将x0设置为1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#随机梯度上升算法
def stockGradAscent(dataMatrix, classLabels, numIter=150):
    #获取数据矩阵的行列长度
    m, n = shape(dataMatrix)
    #回归系数行矩阵，全部初始化为1
    weights = ones(n)
    #进行迭代计算系数行矩阵
    for j in range(numIter):
        #随机选取样本，更新回归系数
        for i in range(m):
            dataIndex = range(m)
            alpha = 4 / (1.0 + i + j) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#可视化显示分类效果，与loadDataSet搭配使用
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#对于任意一个新的数据行矩阵，计算他所属类别
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0.0

#测试算法
def colicTest():
    #训练数据
    frTrain = open('horseColicTraining.txt')
    #测试数据
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    #依次取出每个样本数据，放入trainingSet和trainingLabels
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    #根据训练数据，求出回归系数
    traningWeight = stockGradAscent(array(trainingSet), trainingLabels, 500)
    #错误分类计数
    errorCount = 0
    #测试次数训练
    numTestVec = 0.0
    #依次测试每个测试样本
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), traningWeight)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    #返回错误率
    return errorRate

#多次测试，求平均值
def MultTest():
    numTest = 10
    errorSum = 0.0
    for k in range(numTest):
        errorSum += colicTest()
    print "After %d iterations the average error rate is %f" % (numTest, errorSum/float(numTest))


MultTest()
