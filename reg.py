import numpy as np #to use the vector functions
import pandas as pd #to read the csv
import csv
import math
from random import random
import matplotlib.pyplot as plt

#Author: Monynich Kiem
#Date: Feb. 21. 2022
#Desc: Linear regression and polynomial expansion program using csv data being read in

#hyperparameters are the alpha and epoch
alpha = 0.001 #dr harrison said smaller numbers are better
max_epoch = 100 #42 just bc of hitchikers guide to the galaxy
num_features = 11 #because there are 11 in this specific dataset
weights = []
for x in range(num_features + 1): #+1 is to account for the size being increased since the first input is always 1
    weights.append(random()) #ranadomize float between 0 to 1
weights = np.array(weights)
#print(type(weights))
#print(weights)
MSE = []

#pt 1--------------------------------------------------------------------------
wine = np.loadtxt('winequality-red.csv', delimiter = ',', skiprows = 1)
epoch = 1
#print(wine[:,0]) #prints first column
#print(wine[0,:]) #prints first row
#print(type(wine))

while epoch <= max_epoch:
    ex = 0
    for x in wine:
        entry = np.array(wine[ex])
        #print(entry)
        feat = entry[:-1] #only the features doesnt include the quality rating
        #print(feat)
        input = np.insert(feat, 0, 1.0) #x0 is always 1
        #print(input)
        #print(np.shape(weights))
        #print(np.shape(input))
        #print(type(input))
        weights = []
        for x in range(num_features + 1): #+1 is to account for the size being increased since the first input is always 1
            weights.append(random()) #ranadomize float between 0 to 1
        weights = np.array(weights)
        hypothesis = np.dot((np.transpose(weights)), input) #scalar
        #print(hypothesis)
        #print(type(hypothesis))
        
        gt = wine[ex][(num_features-1)]
        #print(gt)
        
        raw_err = hypothesis - gt #scalar
        #print(raw_err)
        
        gradient = raw_err * input #1 x n
        #update weights:
        
        temp = weights
        feat = 0
        for y in weights:
            weights[feat] = temp[feat] - alpha * gradient[feat]
            feat += 1
        mse = np.dot(np.transpose(weights), input)
        mse = mse - gt
        mse = abs(mse)
        MSE.append(mse)
        ex += 1
    avgSE = sum(MSE)/ len(MSE)
    
    if epoch % 10 == 0:
        print(avgSE)
    epoch += 1

print(weights)

'''
x1 = [i[0] for i in wine]
x2 = x1
y1 = [i[num_features] for i in wine]
y2 = []
ex = 0
for j in wine:
    entry = np.array(wine[ex])
    #print(entry)
    feat = entry[:-1] #only the features doesnt include the quality rating
    input = np.insert(feat, 0, 1.0) #x0 is always 1
    y2.append(np.dot((np.transpose(weights)), input))
    ex += 1
    
plt.xlabel("Inputs")
plt.ylabel("Outputs")
plt.scatter(x1, y1, color="pink", label="Original Data")
plt.scatter(x2, y2, color="blue", label="Model Prediction")
plt.legend(loc="best")
plt.show()
'''
