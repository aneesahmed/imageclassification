# coding=utf-8
import numpy as np
from image_processing import *
from sklearn import preprocessing

trainMatrix, testMatrix = getImageVectors()
#np.set_printoptions(threshold=np.nan)
# print("Train Matrix")
# print(trainMatrix.shape)
# print(trainMatrix)
# print('------------------')
# print("Test Matrix")
# print(testMatrix.shape)
# print(testMatrix)
# print('------------------')

### =================
## Image centering and scalling
centered_matrix = np.zeros(trainMatrix.shape, dtype=float)
scalled_matrix = np.zeros(trainMatrix.shape, dtype=float)
#print(centered_vector)
centerize(trainMatrix, centered_matrix)
print(centered_matrix[0].mean(), "centered data", centered_matrix[0],"\n\n" )
#scalled = preprocessing.scale(trainMatrix)
#print(scalled[0].mean(), "scalled data", scalled[0])

scalling(centered_matrix, scalled_matrix)
print(scalled_matrix[0].mean(), "scalled data", scalled_matrix[0],"\n\n")

#centered_vector = preprocessing.scale(trainMatrix)
#print(centered_vector[0])
coVar_train = np.cov(scalled_matrix.transpose())
# 1/(N-1) * Sum (x_i - m)(x_i - m)^T (where m is the mean)
print(coVar_train[0].mean(), "co variance", coVar_train,"\n\n")
