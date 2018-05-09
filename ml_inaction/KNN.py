# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:05:12 2018

@author: huang_qi
"""

from numpy import *
    
import operator

from os import listdir






def classify0(inX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]     #number of line

    diffMat = tile(inX, (dataSetSize,1)) - dataSet   #copy

    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)    #sum of lines

    distances = sqDistances**0.5  

    sortedDistIndicies = distances.argsort()     #from numpy,return index

    classCount={}          

    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #items function produce an iterable
    #operator.itemgetter can get the first area number
    
    sortedClassCount = sorted(classCount.items(), 
                              key=operator.itemgetter(1), reverse=True)


    return sortedClassCount[0][0]
    

def file2matrix(filename):

    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}

    fr = open(filename)

    arrayOLines = fr.readlines()    

    numberOfLines = len(arrayOLines)            #get the number of lines in the file

    returnMat = zeros((numberOfLines,3))        #prepare matrix to return

    classLabelVector = []                       #prepare labels return   

    index = 0

    for line in arrayOLines:

        line = line.strip()

        listFromLine = line.split('\t')
        
        #two dimension 
        returnMat[index,:] = listFromLine[0:3]

        if(listFromLine[-1].isdigit()):

            classLabelVector.append(int(listFromLine[-1]))

        else:

            classLabelVector.append(love_dictionary.get(listFromLine[-1]))

        index += 1

    return returnMat,classLabelVector




def autoNorm(dataSet):

    minVals = dataSet.min(0)

    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))

    m = dataSet.shape[0]

    normDataSet = dataSet - tile(minVals, (m,1))

    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide

    return normDataSet, ranges, minVals




def datingClassTest():

    hoRatio = 0.10      #hold out 

    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file

    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]

    numTestVecs = int(m*hoRatio)

    errorCount = 0.0

    for i in range(numTestVecs):

        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)

        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))

        if (classifierResult != datingLabels[i]): errorCount += 1.0

    print( "the total error rate is: %f" % (errorCount/float(numTestVecs)))

    print (errorCount)



def classifyPerson():

    resultList = ['not at all', 'in small doses', 'in large doses']

    percentTats = float(input("percentage of time spent playing video games?"))

    ffMiles = float(input("frequent flier miles earned per year?"))

    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles, percentTats, iceCream, ])

    classifierResult = classify0((inArr - \

                                  minVals)/ranges, normMat, datingLabels, 3)

    print( "You will probably like this person: %s" \
          % resultList[classifierResult - 1])


        




























