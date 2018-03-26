# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import HelperFunctions as hf
debug = 1 # debug
np.set_printoptions(linewidth=380)
X = np.array([[2,3,2,2,3, 2, 2 ], [5, 5, 7, 12, 12,  12, 12], [5, 9, 9, 9 , 9 ,  10, 10] ], dtype=float)
if debug: print("X.shape", X.shape)

if debug: print("X\n", X)
means = np.mean(X, axis=1)
var = np.var(X, axis=1)
if debug: print('mean, var')
if debug: print(means, var)
for j, mean in enumerate(means):
    X[j]= X[j] - mean
if debug: print("xCenter\n", X)
# means = np.mean(X, axis=1)
# var = np.var(X, axis=1)
# print('mean, var')
# print(means, var)
C = []
m,n = X.shape
if debug: print("m, n", m,n)
if debug: print(np.reshape(X[0],(n,1) ).shape)
tot = 0
C= np.zeros([n,n])
for i, x in enumerate(X):
    x= np.reshape(X[i],(n,1) )
    #xt= np.reshape(X[i],(1,n) )

    cov = np.array(( np.matmul(x, x.T) ) )

    if debug: print("Cov", cov.shape, np.sum(cov))
    C = C + cov

if debug: print("C\n", C.shape, "\n", C)
S  = np.zeros([n,n])
U, S, V = la.svd(C)
sortedS = sorted(S, reverse=True)
sortedS = np.array(sortedS)
# no S is not in use and data is stored in sortedS
S =  np.zeros([n,n])
for i in range(n):
    S[i,i]= sortedS[i]
if debug: print("U, S , V")
if debug: print(U.shape, S. shape, V.shape)
if debug: print(S)
sortedU = np.sort(U, axis=0)
### to reverse
sortedU = sortedU[::-1]
if debug: print("sorted u \n", sortedU[:5,:10])
accuracy = 0
k = 3
projectedX = np.zeros(X.shape)
print("projectedx\n", projectedX)
hf.project(U,k,X, projectedX)
for x1, x2 in (zip(projectedX, X)):
    print("\n", x1,"\n", x2)

#print("norm", norm)
#print("X2, Xv","\n", projection, "\n", X[0])
# for k in range(0,n):
#     norm ,prediction= hf.predict(debug,U,k)
#     if norm <1:
#         accuracy= accuracy + 1
#         break





