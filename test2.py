# coding=utf-8
import numpy as np


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

v1 = np.linspace(-2,9,20)
rin = np.array([-3,5])
print(v1)
print("v2")
print( rin)
print("linmap\n")
print(linmap(v1,rin))
