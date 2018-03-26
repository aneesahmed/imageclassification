#coding=utf-8
from numpy import array
from numpy import mean
import numpy as np
from numpy import linalg as la
from numpy.linalg import eig
from image_processing import getMeanVector
np.set_printoptions(linewidth=380)
# define a matrix
A = array([[1, 2,1,2,1,2,3,4], [21,22,21,22,21,22,23,24], [31, 32,31,32,31,32,33,34], [41,42,41,42,41,42,43,44],
           [51,52,51,52,51,52,53,54]])
print("A\n", A.shape, A)
A = A.T
print("A\n", A)
# calculate the mean of each column
M = getMeanVector(A)
print('M\n' ,M)
# center columns by subtracting column means
C = A - M
print("C\n", C)
SM = np.matmul( C.T, C )
print("SM\n", SM)
# calculate covariance matrix of centered matrix
xU, xS, xV = la.svd(SM)
print ("shape xU, xs, xv")
print(xU.shape, xS.shape, xV.shape)
print("U before Sorting")
print(xU[:5, :10])

#U = sorted(U.all(), reverse=True)
#sortedU = np.array(sorted( U,  key=lambda x:x[2], reverse=True))
sxU = np.sort(xU, axis=0)
print("sorted u \n", sxU[:5,:10])
### to reverse
sxU = sxU[::-1]
#print("shape ", sortedU.shape)
print("after reverse")
print(sxU[:5, :10])
print("S")
print(xS[:10])
sxS = sorted(xS, reverse=True)
#sortedS = sortedS[:-1]
print("after sorting of S and reversing \n",sxS[:10])

tot = np.sum(xS)
print("tot\n", tot)
var_exp = [(i / tot) for i in sxS]
print("var exp \n", var_exp)
cum_var_exp = np.cumsum(var_exp)
print("cumsumvarexp\n", cum_var_exp)
#projected_C =  np.matmul(sxU, C[0])
print("C matrix shape and sxU\n", C[0,:].shape, sxU.shape)
Z = np.matmul(sxU[:2,:],C[10])
print("x of sxU, projected_C , actualC of zero\n")
for a,b,c in zip(sxU[0],Z, C[:,:]):
    print( a, ",",b, ",",c )

# eigendecomposition of covariance matrix
# values, vectors = eig(V)
# print('vector\n', vectors)
# print('values\n', values)
# # project data
# P = vectors[:,:1].T.dot(C.T)
# P = vectors.T.dot(C.T)
# print(' P.T\n', P.T)
