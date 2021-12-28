import config
import numpy as np
import os
import cv2

def vectorized_result(j): #返回一个11行1列的向量，正确结果处为1，其余为0
    e = np.zeros((config.numClass,1))
    e[j] = 1.0
    return e

def load_data():
    trainImg = np.empty((config.trainNum,config.downSize,config.downSize,3))
    trainResult = np.empty((config.trainNum,config.numClass))

    testImg = np.empty((config.valNum,config.downSize,config.downSize,3))
    testResult = np.empty((config.valNum,config.numClass))

    countTrain = 0
    for line in open(config.trainPath):
        line = line.split()[0]
        if line != '':
            imgPath,label = line.split("_",1)
            img = cv2.imread(imgPath)[:,:,::-1]
            trainImg[countTrain] = img
            label = vectorized_result(config.classes[label])
            trainResult[countTrain,:] = label.T
            countTrain = countTrain + 1

    countTest = 0
    for line in open(config.valPath):
        line = line.split()[0]
        if line != '':
            imgPath,label = line.split("_",1)
            img = cv2.imread(imgPath)[:,:,::-1]
            testImg[countTest] = img
            label = vectorized_result(config.classes[label])
            testResult[countTest,:] = label.T
            countTest = countTest + 1

    training_data = (trainImg,trainResult)
    test_data = (testImg,testResult)

    return (training_data,test_data)

def load_data_wrapper():

    tr_d ,va_d, te_d = load_data()
    training_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]]

    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_results = np.array(training_results)
    training_results = training_results.reshape(training_results.shape[0],training_results.shape[1]).T
    print(training_results.shape)
    training_data = zip(training_inputs,training_results)

    validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = zip(validation_inputs,va_d[1])

    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = zip(test_inputs,te_d[1])

    return(training_data,validation_data,test_data)





