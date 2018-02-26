# coding=utf-8
import numpy as np
from random import randint
N = 2
x=[randint(0, 9) for _ in range(N)]
print(x)
trRows = int(401* (80/100) ) #80% pic for training
tsRows = 401 - trRows    # 20% pic for testing

#print(trRows,tsRows)
# a = np.zeros((3,5))
#a = np.array ( [[1,2,3,4], [1,2,3,4], [1,2,3,4]])
#print(a)

#a[1,:] = np.array([1,2,3,4,5])
#print(a[:,[1,2]])
#print(a)