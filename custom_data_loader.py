from scipy import ndimage 
import numpy as np
import cv2
import sys
import os
import math

def GetShiftForBestFit(image):
    (centerY, centerX) = ndimage.measurements.center_of_mass(image)
    (rows, columns) = image.shape
    xShift = int(np.round( columns / 2.0 - centerX))
    yShift = int(np.round( rows / 2.0 - centerY))

    return (xShift, yShift)

def Shift(image, xShift, yShift):
    (rows, columns) = image.shape
    Mat = np.float32([[1, 0, xShift], [0, 1, yShift]])
    shiftedImage = cv2.warpAffine(image, Mat, (columns, rows))

    return shiftedImage

def GetFormattedImage(imagePath, fileName, writeToFile = False):
    imageData = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    imageData = cv2.resize(255-imageData, (28, 28))

    (thresh, imageData) = cv2.threshold(imageData, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(imageData[0]) == 0: # These 4 loop remove the totally black outer rows
        imageData = imageData[1:]
    
    while np.sum(imageData[:,0]) == 0:
        imageData = np.delete(imageData,0,1)

    while np.sum(imageData[-1]) == 0:
        imageData = imageData[:-1]

    while np.sum(imageData[:,-1]) == 0:
        imageData = np.delete(imageData,-1,1)

    (rows,columns) = imageData.shape

    if(rows > columns): # This will bring the image down to 20x20
        multFactor = 20.0/rows
        rows = 20
        columns = int(round(columns * multFactor))
        imageData = cv2.resize(imageData, (columns, rows))
    else:
        multFactor = 20.0/columns
        columns = 20
        rows = int(round(rows * multFactor))
        imageData = cv2.resize(imageData, (columns, rows))

    columnPadding = ((int(math.ceil(((28-columns)/2.0)))), (int(math.floor(((28-columns)/2.0))))) # This part resizes the image back to 28x28 by adding black rows around the 20x20 image
    rowPadding = ((int(math.ceil(((28-rows)/2.0)))), (int(math.floor(((28-rows)/2.0)))))

    imageData = np.lib.pad(imageData, (rowPadding, columnPadding), 'constant')

    (xShift, yShift) = GetShiftForBestFit(imageData)
    shiftedImage = Shift(imageData, xShift, yShift)

    imageData = shiftedImage
    if(writeToFile):
        cv2.imwrite("./customTestDataModified/"+fileName, imageData)

    flatten = imageData.flatten() / 255.0

    imageInputData = np.reshape(flatten, (784,1))

    return imageInputData