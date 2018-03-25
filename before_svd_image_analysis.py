# coding=utf-8
import numpy as np
from image_processing import *
from numpy import linalg as la
from matplotlib import pyplot as plt

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
centered_matrix = trainMatrix -  meanVector
print("centered matrix\n", centered_matrix.shape ,"\n", centered_matrix)
#for i in range(0,10):
    #print("Normal Data = ", trainMatrix[i][0], " Centered Data  =", np.round(centered_matrix[i][0],2) ," Mean of Row = ",np.round(meanVector[i][0],2))
print(" mean of centered matrix ( after rounding to 8 digits) = ", round(centered_matrix.mean(), 8))
# multiplying X_transpose with X to get surrogate

surrogate_matrix = np.matmul( centered_matrix.transpose(), centered_matrix )
#surrogate_matrix = np.cov(centered_matrix)
print("surrogate_matrix[:10,:10]\n", surrogate_matrix[:10,:10])
# ###############################3
# finding eignvectors and eignvalues of our suggrogate matrix
U, S, V = la.svd(surrogate_matrix)
eignvalues, eignVectors = la.eig(surrogate_matrix)
print("Size of eignMatrix", eignVectors.shape)

print("the eignVectors[:10, :10] of surrogate Covariance\n", eignVectors[:10, :10])
print("Vector  of eignvalues[:10]", eignvalues[:10],"\n\n")

# ######################3333
#eignvalues=np.sort(eignvalues, kind='heapsort', )
#eignvalues=np.argsort( eignvalues)
# in place X[::-1].sort()
print("shape of eignvalues\n",eignvalues.shape)
eignvalues = sorted(eignvalues, reverse=True)
#for i  in range(10):
#    print(i, eignvalues[i], eignvalues[-1 * i],"comparer ", eignvalues[i] > eignvalues[i * -1] )

#print("sorted eign values shape \n",  eignvalues.shape)
print("Vector  of sorted eignvalues[:10]\n", eignvalues[:10],"\n\n")
print("first 2 values %age \n", eignvalues[0] / sum(eignvalues), "\n" , eignvalues[1] / sum(eignvalues))
#idx = np.argsort(eignvalues )
# no need to do above, eignvalue already sorted revierse
#var_exp = [(i / tot) for i in sorted(eignvalues, reverse=True)]
reduced_eignvalues = eignvalues[:20]
tot = sum(reduced_eignvalues)
var_exp = [(i / tot) for i in reduced_eignvalues]
print(var_exp)
cum_var_exp = np.cumsum(var_exp)

plt.bar(np.arange(len(reduced_eignvalues)), alpha=0.5, align='center', height=var_exp,  label='explained variance', color="blue")
plt.step(np.arange(len(reduced_eignvalues)), cum_var_exp, where='mid',label='cumulative explained variance', color="green")
#plt.step(np.arange(len(var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
#plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance')
# # plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
exp = [round(x ,1) for x  in var_exp]
plt.xticks(np.arange(len(reduced_eignvalues)), exp)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
#plt.show()
###############3
net = 0
#for i,x in enumerate(var_exp):
#    net = net + x
#    print("i ", i, "x ", x, " net ", net)
##################33333
#### projection
newMatrix = centered_matrix.dot(eignVectors[:,:10])
print("new matrix shape ", newMatrix.shape, "old matrix shape", centered_matrix.shape)
print("newMatrix  and old one ")
for x1, x2 in zip(newMatrix[:, 1], centered_matrix[:,1]):
    print("new matrix=", x1, " real one=", x2)

#####################
#Prediction working (25 march 2018)
# transformed = matrix_w.T.dot(all_samples)
# assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."

