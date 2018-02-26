# coding=utf-8
import numpy as np
from image_processing import getImageVectors

trainMatrix, testMatrix = getImageVectors()
#np.set_printoptions(threshold=np.nan)
print("Train Matrix")
print(trainMatrix.shape)
print(trainMatrix)

print('------------------')
print("Test Matrix")
print(testMatrix.shape)
print(testMatrix)
