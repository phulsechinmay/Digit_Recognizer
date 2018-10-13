import random
import numpy as np
import cv2

class Network(object):

    def __init__(self, sizes, eta):
        self.learningRate = eta
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def Feedforward(self, activation):
        for w, b in zip(self.weights, self.biases):
            activation = Sigmoid(np.dot(w, activation)+b)
        return activation

    def StochasticGradientDescent(self, epochs, miniBatchSize, eta, trainingData, testData=None):
        self.learningRate = eta
        trainingData = list(trainingData)
        if testData: 
            testData = list(testData)
            numTest = len(testData)
        numTraining = len(trainingData)
        for j in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [
                trainingData[k:k+miniBatchSize] for k in range(0, numTraining, miniBatchSize)]
            for miniBatch in miniBatches:
                self.ProcessMiniBatch(miniBatch, eta)
            if testData:
                print ("Epoch {0}: {1}% Accuracy".format(j, (self.Evaluate(testData) / numTest) * 100.00))
            else:
                print ("Epoch {0} Finished".format(j))

    def ProcessMiniBatch(self, miniBatch, eta):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        for data, correctResult in miniBatch:
            delta_grad_w, delta_grad_b = self.Backpropagate(data, correctResult)
            grad_b = [gb+dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, delta_grad_w)]
        self.weights = [w - ( eta / len(miniBatch))* gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - ( eta / len(miniBatch)) * gb for b, gb in zip(self.biases, grad_b)]

    def Backpropagate(self, data, correctResult):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        # Feedforward
        curActivation = data
        activations = [data]
        vec_z = []
        for b, w in zip(self.biases, self.weights):
            curZ = np.dot(w, curActivation)+b
            vec_z.append(curZ)
            curActivation = Sigmoid(curZ)
            activations.append(curActivation)

        # Backpropagation
        curDelta = self.costFunction(activations[-1], correctResult) * SigmoidDerivative(vec_z[-1])
        grad_b[-1] = curDelta
        grad_w[-1] = np.dot(curDelta, activations[-2].transpose())

        for l in range(2, self.numLayers):
            z = vec_z[-l]
            sp = SigmoidDerivative(z)
            curDelta = np.dot(self.weights[-l+1].transpose(), curDelta) * sp
            grad_b[-l] = curDelta
            grad_w[-l] = np.dot(curDelta, activations[-l-1].transpose())
        return (grad_w, grad_b)

    def Evaluate(self, testData):
        results = [(np.argmax(self.Feedforward(data)), correctResult) for (data, correctResult) in testData]
        return sum(int(networkResult == testResult) for (networkResult, testResult) in results)

    def costFunction(self, trainingResult, givenResult):
        return (trainingResult-givenResult)

    def EvaluateOneImage(self, testData):
        result = self.Feedforward(testData)
        np.savetxt("TestResult.txt", result)
        return (np.argmax(result))

    def CreateImageFromNumber(self, num):
        testLearningRate = 8
        inputShape = (784,1)
        inputImage = np.random.rand(784,1) # Creates random samples from uniform distribution
        inputImage = np.random.randn(784,1) # Creates random samples from normal distribution
        #inputImage = np.zeros(inputShape);
        threshold = 999.999
        writeCount = 1
        result = np.loadtxt("TestResult.txt")
        result.reshape((10,1))
        while((threshold > 0.2) or (threshold < -0.2)):
            # Feedforward
            curActivation = inputImage
            activations = [inputImage]
            vec_z = [curActivation]
            for w, b in zip(self.weights, self.biases):
                curZ = np.dot(w, curActivation) + b
                vec_z.append(curZ)
                curActivation = Sigmoid(curZ)
                activations.append(curActivation)
            
            # Backpropagation
            outputShape = (10,1)
            #result = np.zeros(outputShape)
            #result[num-1] = 1
            #threshold = (self.costFunction(activations[-1], result))[num-1]
            curDelta = self.costFunction(activations[-1], result) * SigmoidDerivative(vec_z[-1]) # The -1 indexing returns the last element of the list
            i = len(vec_z) - 2
            while (i >= 1):
                curZ = vec_z[i]
                curGrad = SigmoidDerivative(curZ)
                curDelta = np.dot(self.weights[i].transpose(), curDelta) * curGrad
                i = i - 1
            
            curDelta = np.dot(self.weights[0].transpose(), curDelta)
            inputImage = inputImage - (np.power(10, testLearningRate)) * curDelta

            writeCount = writeCount + 1
            if(writeCount == 10000):
                writeCount = 1
                writeImageShape = (28,28)
                newImage = inputImage * 255.0
                newImageData = np.reshape(newImage, writeImageShape)
                filePath = "./generatedImages/" + "GenImg" + str(num) + ".png" 
                cv2.imwrite(filePath, newImageData)

        
        generatedImageShape = (28,28)
        inputImage = inputImage * 255.0
        generatedImage = np.reshape(inputImage, generatedImageShape)

        return generatedImage


# Neuron Functions
def Sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

def SigmoidDerivative(z):
    return Sigmoid(z) * (1-Sigmoid(z))
