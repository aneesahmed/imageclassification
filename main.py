# coding=utf-8
import numpy as np
import HelperFunctions as hf
from numpy import linalg as la
from matplotlib import pyplot as plt
import data2SVD as ds
import math
np.set_printoptions(linewidth=250)
################################
outfile = "SVD.npz"
try:
    data = np.load(outfile)
    U = data['U']
    S = data['S']
    V = data['V']
    trainMatrix = data['trainMatrix']
    testMatrix = data['testMatrix']
    print("Data loaded from saved binary file")

except FileNotFoundError:
    print("Computing S V D")
    U,S,V, trainMatrix, testMatrix = ds.calculateSVD()
    np.savez(outfile, U=U, S=S,V=V, trainMatrix=trainMatrix, testMatrix= testMatrix)

#######################
# m,n = trainMatrix.shape
# projectedX = np.zeros([m,n])
# for k in range(10,n,10):
#     norm, norm2, pX = hf.sampleProjection(U,k,trainMatrix[0])
#     print("k:", k, " ,norm:", norm, ", norm2:", norm2)
#     if norm2 <=1:
#         break
# print("k", k, "pX vs X\n", pX, "\n", trainMatrix[0])

########################################
## finding best K values using S
requiredVariance = .99
variance, k = hf.findingK(S, requiredVariance)
print("variance", variance, "k", k)

## validating k
norm, norm2 ,projectedx = hf.sampleProjection(U, k, trainMatrix[0])
print("norm", norm, "norm2", norm2, "projectedX vs orginal x", "\n", projectedx[:10],"\n",   trainMatrix[0][:10])
######################
################
# m,n = trainMatrix.shape
# projectedTest = np.zeros([m,n])
# accuracies = []
# kCount = []
#
# for k in range(1000,n, 100):
#     hf.transform(U, k, testMatrix, projectedTest)
#     accuracy = 0
#     for i, testX in enumerate(testMatrix):
#         norm =0
#         bestNorm =9999999999
#         index = 0
#         for j, trainX in enumerate(trainMatrix):
#             # euclidean norm or distance
#             #print(testX.shape, trainX.shape, "norm", norm)
#             norm = hf.getNorm(testX, trainX)
#             if bestNorm > norm:
#                 bestNorm = norm
#                 index = j
#
#         # now check if picture is matched with correct one or not
#         if math.ceil(index/2) == math.ceil(index/8):
#              accuracy = accuracy +1
#     print(" K value", k,  "acuracy level", accuracy,  "index matched", index, "best norm", bestNorm)
#         ########## appending accuracy to the list
#     # store for each value of K, accuracy is what
#     kCount.append(k)
#     accuracies.append(accuracy)
#
# plt.plot(kCount, accuracies)
# plt.show()