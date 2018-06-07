import _pickle as pickle
import gzip as gz
import numpy as np 

def LoadMNISTData():
    imagesFile = gz.open("./data/mnist_images.gz", "rb")
    #testImagesFiles = gz.open("./data/mnist_test_images.gz", "rb")

    (trainingData, validationData, testData) = pickle.load(imagesFile, encoding = 'latin-1')

    imagesFile.close()

    return (trainingData, validationData, testData)

def LoadDataIntoVectors():
    (trainingDataRaw, validationDataRaw, testDataRaw) = LoadMNISTData()

    trainingInputs = [np.reshape(x ,(784,1)) for x in trainingDataRaw[0]]
    trainingResults = [GetOutputVector(result) for result in trainingDataRaw[1]]
    trainingData = zip(trainingInputs, trainingResults)

    validationInputs = [np.reshape(x,(784,1)) for x in validationDataRaw[0]]
    validationData = zip(validationInputs, validationDataRaw[1])

    testInputs = [np.reshape(x,(784,1)) for x in testDataRaw[0]]
    testData = zip(testInputs, testDataRaw[1])

    return (trainingData, validationData, testData)

def GetOutputVector(i):
    vec = np.zeros(((10,1)))
    vec[i] = 1.0
    return vec