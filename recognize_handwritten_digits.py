import network
from mnist_data_loader import *
from scipy import ndimage 
import numpy as np
import cv2
import sys
import os
import math
import custom_data_loader

hiddenLayerSize = 25
learningRate = 2.5
epochs = 35
batchSize = 12
writeProcessedImage = True

(trainingData, validationData, testData) = LoadDataIntoVectors()

neuralNetwork = network.Network([784, hiddenLayerSize, 10], learningRate)

neuralNetwork.StochasticGradientDescent(epochs, batchSize, learningRate, trainingData, testData = testData)

while(True):
    customFileName = input("Enter the name of the file of the handwritten digit (include extension) (Enter \"Exit\" to exit): ")

    if(customFileName == "Exit") or (customFileName == "exit"):
        print("Exiting...")
        break;
    
    if(customFileName == "WriteToFile"):
        neuralNetwork.writeInfoToFiles()
        break

    filePath = "./customTestData/" + customFileName

    imageInputData = custom_data_loader.GetFormattedImage(filePath, customFileName, writeProcessedImage)

    result = neuralNetwork.EvaluateOneImage(imageInputData)

    print("The neural network recognized a {0}".format(result))
