from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

#转换文本数据
from pip._vendor.distlib.compat import raw_input


def fileMatrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLine = len(arrayLines)
    returnMat = zeros((numberOfLine,3)) #一个指定形状的数组
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip() #移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        listFromLine =line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
#        classLabelVector.append(int(listFromLine[-1]))
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

#matplotlib创建散点图
#fig = plt.figure()
#ax = fig.add_subplot(111)

#数值归一化 newVals = (oldVals - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
#    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

#knn算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #shape:矩阵的行列数 shape[0]:矩阵的行数
    #计算距离
    diffMat = tile(inX , (dataSetSize,1)) - dataSet  #tile(A, reps) 将A重复reps次 重复A的各个维度
    sqDiffMat = diffMat**2 #**是幂运算
    print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1) #axis=0对应着最外层的[]，axis=1对应第二外层的[]，以此类推，axis=n对应第n外层的[]。
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #argsort():对数组进行排序（默认是从小到大），并返回下标
    classCount = {}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #返回字典classCount中voteIlabel元素对应的值,若无，则进行初始化为0
    #排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

#测试约会网站
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = fileMatrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]) :errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

#预测函数
def classifyPerson():
    resultDict = {'didntLike':'not', 'smallDoses':'small', 'largelDoses':'large'}
    percenrTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = fileMatrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percenrTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % (resultDict[classifierResult]))

if __name__ == '__main__':
    classifyPerson()