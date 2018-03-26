# coding=utf-8
import numpy as np
import HelperFunctions as hf
from numpy import linalg as la
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=380)
# for branch rowmatrix-pca-working we are not going to
# store images into column vector, instead each image will be
# saved into rows  so there will be 320 * 2500 trainmatrix
# and 80 * 2500 test matrix
# m x n = 320 * 2500
trainMatrix, testMatrix = hf.getImageVectors()

#np.set_printoptions(threshold=np.nan)
print("Train Matrix")
print("shape", trainMatrix.shape)
print("matrix sample=10", trainMatrix[:10,:])
print('------------------')
print("Test Matrix")
print("shape", testMatrix.shape)
print("matrix sample=10", testMatrix[:10,:])
print('------------------')

### =================
 ## Image centering and scalling
Xtrain = np.zeros(trainMatrix.shape, dtype=float)
Xtest = np.zeros(testMatrix.shape, dtype=float)
meanVector =  hf.getMeanVector(trainMatrix)
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
