import numpy as np #to use the vector functions
import pandas as pd #to read the csv
import csv
import math
from random import random
import matplotlib.pyplot as plt

#Author: Monynich Kiem
#Date: Feb. 21. 2022
#Desc: Linear regression and polynomial expansion program using csv data being read in
'''
def hypothesis(theta, dataset_x):
    return theta*dataset_x
#hypothesis(theta, dataset_x)

def cost(dataset_x, dataset_y, theta):
    y_h = hypothesis(theta, dataset_x)
    #print(y_h)
    cost = np.sum((y_h - dataset_y**2)/(2*m))
    #print(cost)
    return cost
#cost(x,y, theta)

def gradient(dataset_x, dataset_y, theta, alpha, epoch):
    cost_iter = []
    curr_epoch = 0
    while curr_epoch < epoch:
        y_h = hypothesis(theta, dataset_x)
        y_h = np.sum(y_h, axis = 1)
        for c in range(0, len(dataset_x.columns)):
            theta[c] = theta[c] - alpha * sum((y_h-dataset_y)*dataset_x.iloc[:,c])/len(dataset_x)
            j = cost(dataset_x, dataset_y, theta)
            cost_iter.append(j)
    return cost_iter, j, theta

#print(theta[0])
cost_iter, j, theta = gradient(x, y, theta, alpha, epoch)
#print(theta)
'''

#hyperparameters are the alpha and epoch
alpha = 0.001 #dr harrison said smaller numbers are better
max_epoch = 10 #epochs are just the number of iterations
num_features = 11 #because there are 11 in this specific dataset
weights = []
#for x in range(num_features + 1): #+1 is to account for the size being increased since the first input is always 1
#    weights.append(random()) #ranadomize float between 0 to 1
#weights = np.array(weights)
#print(type(weights))
#print(weights)
MSE = []
'''
wine = pd.read_csv('winequality-red.csv')
wine = pd.concat([pd.Series(1, index = wine.index, name = 'theta0'), wine], axis = 1)
print(wine.head())
m = len(wine)
x = wine.drop(columns = 'quality') #input variables
#print(x.to_numpy())
y = wine.iloc[:,12] #output variable
#print(y.to_numpy())
'''
#pt 1--------------------------------------------------------------------------
wine = np.loadtxt('winequality-red.csv', delimiter = ',', skiprows = 1)
epoch = 1
#print(wine[:,0]) #prints first column
#print(wine[0,:]) #prints first row
#print(type(wine))

def normalize(dataset):
    columns = dataset.shape[1]
    print(columns-1)
    for i in range(columns-1):
        dataset[:,i] = dataset[:,i]/np.max(dataset[:,i])
normalize(wine)
print(wine)

while epoch <= max_epoch:
    ex = 0
    for i in wine:
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
        for j in range(num_features + 1): #+1 is to account for the size being increased since the first input is always 1
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

