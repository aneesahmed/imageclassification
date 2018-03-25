# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def linmap(vin,rout):
    # function for linear mapping between two ranges
    # inputs:
    # vin: the input vector you want to map, range [min(vin),max(vin)]
    # rout: the range of the resulting vector
    # output:
    # the resulting vector in range rout
    # usage:
    # >>> v1 = np.linspace(-2,9,100);
    # >>> rin = np.array([-3,5])
    # >>> v2 = linmap(v1,rin);
    # *** (this function uses numpy module)
    #
    a = np.amin(vin);
    b = np.amax(vin);
    c = rout[0];
    d = rout[1];
    return ((c+d) + (d-c)*((2*vin - (a+b))/(b-a)))/2;

# v1 = np.linspace(-2,9,20)
# rin = np.array([-3,5])
# print(v1)
# print("v2")
# print( rin)
# print("linmap\n")
# print(linmap(v1,rin))

matrix = np.array([[5,4,3,2,1], [1,2,3,4,5]])
print(matrix, matrix.shape)
matrix = matrix.transpose()
print(matrix, matrix.shape)
meanvector = np.mean(matrix, axis=1)
print("meanvector\n", meanvector.shape,"\n" ,meanvector)
meanvector = meanvector.reshape(meanvector.shape[0], 1)
print("meanvector\n", meanvector.shape,"\n" ,meanvector)

print(matrix - meanvector )

meanvector.sort()
print("meanvector\n", meanvector)
a = np.array([3,2,1,4,5])
a =sorted(a, reverse=True)
print(a)
tot = sum(a)
print("tot", tot)
var_exp = [(i / tot) for i in a]
print(var_exp)
idx = np.argsort(a)
print(idx, "idx")
height = [2,2,2,2,2]
plt.bar(np.arange(len(a)), height= var_exp )
cum_var_exp = np.cumsum(var_exp)
plt.step(np.arange(len(var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.show()

#plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance')