# coding=utf-8
import numpy as np
import math
def centerlized(vector):

    sum= np.sum(vector)
    ln = len(vector)
    mean = sum/ln
    v2 = np.zeros(len(vector))
    v3 = np.zeros(len(vector))
    #print(v2)
    for  i in range(0, vector.shape[0]):
        v2[i] = mean - vector[i]
        v3[i] = v2[i] * v2[i]
    var = np.sum(v3) / ln
    #print("sum", sum,"len", ln, "mean", mean, "v2", v2, v2.mean(),var, v2.var( ddof=True))
    return v2.mean()
###########
#X = np.asarray([[-1, 0, 1], [0, 1, 2]], dtype=np.float)  # Float is needed.
#X = X.flatten()
#print(X)
# Before-normalization.

#X_normalized= centerlized(X)
#X_normalized = preprocessing.normalize(X, norm='l2')
#print(X_normalized)
#-----------------
#a = [[1,2], [3,4]]
#a = np.transpose((a))
#x = np.cov(a)
#print(a)
#print(x)
a = np.array([[1, 2], [3, 4]])
print(a)
b = np.mean(a, axis=1)
print("mean", b)
b = b.reshape(b.shape[0],1)
print("b", b)
c = np.array(a - b)
print(c)
print(c.mean())
