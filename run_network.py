import network
from mnist_data_loader import *

numFinalTests = 10
hiddenLayerSize = 40
learningRate = 3.0
epochs = 30
batchSize = 10

(trainingData, validationData, testData) = LoadDataIntoVectors()

testData = testData

neuralNetwork = network.Network([784, hiddenLayerSize, 10])

neuralNetwork.StochasticGradientDescent(trainingData, epochs, batchSize, learningRate, testData = None)

i = 0
for (x,y) in testData:
    if(i >= numFinalTests):
        break

    i += 1

    result = neuralNetwork.EvaluateOneImage(testData = x)

    if (result != (-999)):
        print("Input was {0} and network recognized {1}".format(y, result))