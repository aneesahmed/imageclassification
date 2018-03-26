# coding=utf-8
import numpy as np
from image_processing import *
from numpy import linalg as la
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=380)
# both matrix are transposed to keep 2500 bits of an image
# in 320 columns with 1 column per image vector
# it means a column contain an image
# for example trainMrix(:, 0) will cotain first image as a vector
#of the image matrix
trainMatrix, testMatrix = getImageVectors()

#np.set_printoptions(threshold=np.nan)
print("Train Matrix")
print(trainMatrix.shape)
print(trainMatrix)
print("Sample Column vector")
print(trainMatrix[:,0])
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
 #
 #
meanVector =  getMeanVector(trainMatrix)
print("Mean Vector  first 10 sample \n", meanVector[:10])
#1 converting meanvector from single dim row shaped to a column vector
print("Mean Vector total Size \n", meanVector.shape, "\n", )

# at the below pint both matrix are in matrix of column vectors
centered_matrix_train = trainMatrix -  meanVector
print("centered matrix\n", centered_matrix_train.shape ,"\n", centered_matrix_train)
#for i in range(0,10):
    #print("Normal Data = ", trainMatrix[i][0], " Centered Data  =", np.round(centered_matrix[i][0],2) ," Mean of Row = ",np.round(meanVector[i][0],2))
print(" mean of centered matrix ( after rounding to 8 digits) = ", round(centered_matrix_train.mean(), 8))
# multiplying X_transpose with X to get surrogate

surrogate_matrix = np.matmul( centered_matrix_train.transpose(), centered_matrix_train )
#surrogate_matrix = np.cov(centered_matrix)
print("surrogate_matrix[:10,:10]\n", surrogate_matrix[:10,:10])
# ###############################3
# finding eignvectors and eignvalues of our suggrogate matrix
U, S, V = la.svd(surrogate_matrix)
# U = np.array(U)
# S = np.array(S)
# V = np.array(V)
print ("shape U, s, v")
print(U.shape, S.shape, V.shape)
print("U before Sorting")
print(U[:5, :10])

#U = sorted(U.all(), reverse=True)
#sortedU = np.array(sorted( U,  key=lambda x:x[2], reverse=True))
sortedU = np.sort(U, axis=0)
print("sorted u \n", sortedU[:5,:10])
### to reverse
sortedU = sortedU[::-1]
#print("shape ", sortedU.shape)
print("after reverse")
print(sortedU[:5, :10])
print("S")
print(S[:10])
sortedS = sorted(S, reverse=True)
sortedS = np.array(sortedS)
#sortedS = sortedS[:-1]
print("after sorting of S and reversing \n",sortedS[:10])
#######################
# getting Z = Ureduced matrix
k = int(10)
#print(sortedU.shape, sortedS.shape)
#W = np.matmul(U, S )
print("sorted U shape", sortedU.shape, "sorted S shape", sortedS.shape, "Train_x shape", centered_matrix_train.shape)
# W = sortedU[:,:10]
# Ztrain = np.array([])
# Ztrain = np.matmul(W.T, centered_matrix_train.T)
#print("w",W.T.shape, W.shape, "X_train", centered_matrix_train.T.shape, Ztrain.shape )
# Ztrain = np.matmul(W.transpose(), centered_matrix_train)
#Ztest = np.matmul(W, centered_matrix_test.T)
# for i, testVector in enumerate(Ztest):
#     for j,trainVector in enumerate(Ztrain):
#         distance = testVector - trainMatrix
#         norm2 = la.norm(distance, ord==2)
#         print("distance", distance, "norm", norm)
