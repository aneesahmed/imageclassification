#coding=utf-8
from numpy import array
from sklearn.decomposition import PCA
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print("A\n", A)
# create the PCA instance
pca = PCA(2)
# fit on data
pca.fit(A)
# access values and vectors
print("pca.components_", pca.components_)
print("pca.explained_variance_", pca.explained_variance_)
# transform data
B = pca.transform(A)
print("B", B)