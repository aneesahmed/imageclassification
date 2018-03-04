# coding=utf-8
import numpy as np
from image_processing import *
from sklearn import preprocessing

trainMatrix, testMatrix = getImageVectors()

#np.set_printoptions(threshold=np.nan)
print("Train Matrix")
print(trainMatrix.shape)
print(trainMatrix)
print('------------------')
print("Test Matrix")
print(testMatrix.shape)
print(testMatrix)
print('------------------')

### =================
## Image centering and scalling
centered_matrix_train = np.zeros(trainMatrix.shape, dtype=float)
centered_matrix_test = np.zeros(testMatrix.shape, dtype=float)

#print(centered_vector)
meanVector = getMeanVector(trainMatrix)
print("Mean Vector total Size \n", meanVector.shape)

centered_matrix = trainMatrix - meanVector
for i in range(0,10):
    print("Normal Data = ", trainMatrix[i][0], " Centered Data  =", np.round(centered_matrix[i][0],2) ," Mean of Row = ",np.round(meanVector[i][0],2))
print(" centered Mean ( after rounding to 8 digits) = ", round(centered_matrix.mean(), 8))
# multiplying X_transpose with X to get surrogate
surrogate_matrix = np.matmul( centered_matrix.transpose(), centered_matrix )