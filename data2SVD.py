# coding=utf-8
import numpy as np
import HelperFunctions as hf
from numpy import linalg as la
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=250)
######################33
def calculateSVD():
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
    #Xtrain = np.zeros(trainMatrix.shape, dtype=float)
    #Xtest = np.zeros(testMatrix.shape, dtype=float)
    #meanVector = np.mean(trainMatrix, axis=1)
    # train matrix will be centered
    count = hf.centered(trainMatrix)
    print("count", count)
    print("centered total Size \n", trainMatrix.shape, "\n", trainMatrix[:10,:10]  )

    # computing surrogate_matrix
    C = hf.getCovariance(trainMatrix)
    ###surrogate_matrix = np.matmul( centered_matrix_train.transpose(), centered_matrix_train )
    # ###############################3
    # finding eignvectors and eignvalues of our suggrogate matrix
    U, S, V = la.svd(C)
    #m,n = trainMatrix.shape
    return U,S,V, trainMatrix, testMatrix