#coding=utf-8
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print("A\n", A)
# calculate the mean of each column
M = mean(A.T , axis=1)
print('M\n' ,M)
# center columns by subtracting column means
C = A - M
print('C\n', C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print('V\n', V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print('vector\n', vectors)
print('values\n', values)
# project data
P = vectors[:,:1].T.dot(C.T)
P = vectors.T.dot(C.T)
print(' P.T\n', P.T)
