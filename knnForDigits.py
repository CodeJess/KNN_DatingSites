from os import listdir

from numpy import *
import operator

#将32*32图像转为1*1024向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

#训练算法
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, classNumStr))
        if(classifierResult != classNumStr): errorCount += 1.0
        print("\nthe total num of errors is: %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


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

if __name__ == '__main__':
    handwritingClassTest()