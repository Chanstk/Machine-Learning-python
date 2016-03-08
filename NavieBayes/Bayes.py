# coding=utf-8
from numpy import *
from math import log

#加载样本
def loadDataSet():
    #样本，6个分词后的字符组
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                       'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #6个分词后的字符组的类别
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

#根据样本建立单词表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#对一段字符串根据单词表建立向量
def setOfWordsToVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

#训练样本（向量）集合
def trainNB0(trainMatrix, trainCategory):
    #样本（向量表示）的个数
    numOfDocs = len(trainMatrix)
    #向量长度
    numOfWords = len(trainMatrix[0])
    #由于是二类问题，所以只需计算一个类别的样本占总体样本比例
    pAbusive = sum(trainCategory) / float(numOfDocs)
    #统计每个单词出现次数的向量，初始每个分量全部设为1（平滑处理）
    p0Num = ones(numOfWords)
    p1Num = ones(numOfWords)
    #统计每类样本的单词总数，初始全部设置为2（平滑处理）
    p0Denom = 2.0
    p1Denom = 2.0
    #求log的向量，便于计算
    p1Vect = ones(numOfWords)
    p0Vect = ones(numOfWords)
    #遍历样本集
    for i in range(numOfDocs):
        #如果不属于垃圾文本
        if trainCategory[i] == 0:
            #统计每个词出现的次数
            p0Num += trainMatrix[i]
            #统计词的个数
            p0Denom += sum(trainMatrix[i])
        #如果属于垃圾文本
        else:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
    #计算每个词的词频，并求log
    for i in range(numOfWords):
        p1Vect[i] = log(p1Num[i] / p1Denom)
        p0Vect[i] = log(p0Num[i] / p0Denom)
    return p0Vect, p1Vect, pAbusive

#给任意一个文本向量，进行类别判定
def classifyNB(vectWord, p0Vec, p1Vec, pClass1):
    #对经典公式左右两边取log
    p1 = sum(p1Vec * vectWord) + log(pClass1)
    p0 = sum(p0Vec * vectWord) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#文本分词
def textParse(bigString):
    import re
    listOfTokes = re.split(r'/W*', bigString)
    return [tok.lower() for tok in listOfTokes if len(tok) > 2]

#分类器测试
def spamTest():
    #分词后的文本列表
    docList = []
    #类别列表，跟文本列表一一对应
    classList = []
    fullText = []
    #对2个文件内的25个文本进行解析
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #建立单词表
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    #随机取10个样本进行测试，剩余40个对分类器进行训练
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    #将40个文本转化成向量
    for docIndex in trainingSet:
        trainMat.append(setOfWordsToVec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #得到词频向量
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    #统计分类错误的个数
    errorCount = 0
    #对10个测试样本进行测试
    for docIndex in testSet:
        wordVector = setOfWordsToVec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    #得出错误率
    print float(errorCount) / float(len(testSet))            


def main():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWordsToVec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWordsToVec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWordsToVec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    spamTest()

if __name__ == '__main__':
    main()
 
