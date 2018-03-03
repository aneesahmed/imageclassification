# coding=utf-8
import random
import os
import numpy as np
from random import randint
from image2vector import image_to_vector
#from scaling_testing import centerlized
# two function for task 2, centerlized and scalling
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


def centerize(s, t):
    # s= source matrix
    # t = centerized matrix
    counter = 0
    for i in range(0,s.shape[0]):
        counter += 1
        #print(counter)
        mean = (s[i].mean())
        #m2 = centerlized(s[i]) ## ok
        t[i] =  mean - s[i]
        #print(counter, mean, t[i].mean(), m2.mean())
    # no need to return because arrays are pass as reference

def getImageVectors():
    tsRows = int(400* (20/100) )  #80% pic for training
    trRows = 400 - tsRows   # 20% pic for testing, adding 5 due to int() issue
    #print(trRows, tsRows)
    train = np.zeros([trRows,10304]) #80% pic for training
    test = np.zeros([tsRows,10304])   # 20% pic for testing
    #print(train.shape)
    #print(test.shape)
    vector = np.zeros(10304)
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
                    train[trCount][:] = vector
                    trCount  = trCount + 1
                #if imgCount % 10 == 0:
                #    print(imgCount, trCount,  tsCount, test1, test2)
    #print(folder,vector[0], tsCount, trCount, imgCount)
        #print(shape)
    #print(train.shape)
    #print(test.shape)
    return train, test