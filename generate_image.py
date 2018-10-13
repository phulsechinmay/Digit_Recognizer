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
epochs = 30
batchSize = 12
writeProcessedImage = True;


result = np.loadtxt("TestResult.txt")
print(result)

(trainingData, validationData, testData) = LoadDataIntoVectors()

neuralNetwork = network.Network([784, hiddenLayerSize, 10], learningRate)

imageData = neuralNetwork.CreateImageFromNumber(9)

neuralNetwork.StochasticGradientDescent(epochs, batchSize, learningRate, trainingData, testData = testData)

while(True):
    customImageNum = input("Enter the number to generate:")

    if(customImageNum == "Exit") or (customImageNum == "exit"):
        print("Exiting...")
        break;

    imageData = neuralNetwork.CreateImageFromNumber(9)

    filePath = "./generatedImages/" + "GenImg" + customImageNum + ".png" 

    cv2.imwrite(filePath, imageData)