import numpy as np
from numpy import linalg as la
from random import randint
# coding=utf-8
import random
import os
import numpy as np
from numpy import linalg as la
from random import randint
from   loggingInitializer import logger

from image2vector import image_to_vector
np.set_printoptions(linewidth=380)

######################################3
def sampleProjection(U,k, Xvector):
    Z = np.matmul(U[:,:k].T, Xvector)
    pX =  np.matmul(U[:,:k], Z)
    # sum = 0
    # for x1, x2 in zip(Xvector, pX):
    #     sum = sum + (round(x1 -x2, 8 ) **2)
    # norm = sum/len(Xvector)
    # #x3 = x1 -x2
    norm2 = np.mean(np.exp2( np.round(np.subtract( Xvector, pX ), 8) ) )
    norm1 = np.mean(np.exp2(Xvector))
    norm =  norm2/norm1
    #norm = round(la.norm(distance), 10)
    return norm, norm2 , pX
###################
def findingK(S, requiredVariance):
    variance = 0
    k = 0
    for k in range(10,len(S), 10):
        variance = np.sum(S[:k])/np.sum(S)
        if variance >= requiredVariance:
            break

    return variance, k

##################33333
def transform(U, k, inputMatrix, outputMatrix):
    maxnorm = 0
    reducedU = (U[:, :k])
    #print("reducedU\n", reducedU)
    for i,x in enumerate(inputMatrix):
        Z = np.matmul(reducedU.T,x)
        outputMatrix[i][:] = np.matmul(reducedU, Z)
    #return maxnorm

#####################3
def getNorm(trainVector, testVector):
    # print("projectedX\n", projectedX[i], "\n", X[i])
    distance = trainVector - testVector
     # print("distance", distance)
    norm = round(la.norm(distance), 10)
    return norm

##################33333
def infoLog(message):
    logger.debug(message)

####################
def scalling(s,t):
    #t = (s - s.mean()) / s.var()
    # s= source matrix
    # t = centerized matrix
    counter = 0
    for i in range(0, s.shape[0]):
        counter += 1
        # print(counter)
        t[i]  = (s[i] - s[i].mean()) / s[i].var()
        # print(counter, mean, t[i].mean(), m2.mean())
        # no need to return because arrays are pass as reference
#####################3333
def getCovariance(X):
    C = []
    m, n = X.shape

    #print("m, n", m, n)
    #print(np.reshape(X[0], (n, 1)).shape)
    tot = 0
    C = np.zeros([n, n])
    for i, x in enumerate(X):
        x = np.reshape(X[i], (n, 1))

        # xt= np.reshape(X[i],(1,n) )
        #print("x after reshape ", x)
        cov = np.array((np.matmul(x,x.T)))
        #cov = np.array((np.matmul(x.T, x)))
        print("Cov", i, np.sum(cov))
        C = C + cov
    return C

###########################33333
def centered(X):
    means = np.mean(X, axis=1)
    #var = np.var(X, axis=1)
    for j, mean in enumerate(means):
        X[j] = X[j] - mean
    return j

#####################33
def getImageVectors():
    train = np.zeros([320, 2500]) #80% pic for training
    test = np.zeros([80,2500])   # 20% pic for testing
    #print(train.shape)
    #print(test.shape)
    vector = np.zeros(2500)
    trCount = 0
    tsCount = 0
    imgCount = 0
    oldFolder = ''
    for folder, dirs, files in os.walk("./att_faces"):
        for filename in files:
            #print(filename, folder) #9.pgm ./att_faces/s5

            shape, vector = image_to_vector(folder+'/' + filename)
            # from each folder take two files and store into testmatrix
            if oldFolder != folder:
                oldFolder = folder
                count = 0
                test1 = randint(0,8)
                test2 = test1 + 1

            if vector[0] > -1:
                imgCount = imgCount + 1
                #print(len(vector))
                vct = vector
                if (  count <2  and ( imgCount % 10 == test1 or imgCount % 10 == test2)  ): # add in the testing matrix
                    #print(imgCount, count, imgCount %10)
                    test[tsCount][:] = vector
                    count = count + 1
                    tsCount = tsCount + 1
                    #print("tst", imgCount % 10, tsCount)
                else:
                    train[trCount][:] = vector.transpose()
                    trCount  = trCount + 1
                #if imgCount % 10 == 0:
                #    print(imgCount, trCount,  tsCount, test1, test2)
    #print(folder,vector[0], tsCount, trCount, imgCount)
        #print(shape)
    #print(train.shape)
    #print(test.shape)
    #return train.transpose(), test.transpose()
    return train, test