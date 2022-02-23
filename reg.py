import numpy as np #to use the vector functions
import pandas as pd #to read the csv
import csv
import math
from random import random
import matplotlib.pyplot as plt

#Author: Monynich Kiem
#Date: Feb. 21. 2022
#Desc: Linear regression and polynomial expansion program using csv data being read in
#using SGD since it's allegedly faster

#HYPERPARAMETERS are the alpha and epoch ---------------------------------------------
alpha = 0.001 #dr harrison said smaller numbers are better
max_epoch = 42 #epochs are just the number of iterations, chose 42 b/c nerd stuff
#-------------------------------------------------------------------------------------

#num_features = 11 #because there are 11 in this specific dataset
#weights = []
#for x in range(num_features + 1): #+1 is to account for the size being increased since the first input is always 1
#    weights.append(random()) #ranadomize float between 0 to 1
#weights = np.array(weights)
#print(type(weights))
#print(weights)
MSE = []

#attempt at pandas usage but kept getting errors b/c unable to iterate a row at a time?
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
curr_epoch = 0
#num_features = len(wine[0,:])-1 #does not include the output/target class label
#print(num_features)
#print(wine[:,0]) #prints first column
#print(wine[0,:]) #prints first row
#print(type(wine))
#https://www.kite.com/python/answers/how-to-iterate-through-columns-of-a-numpy-array-in-python
def normalize(dataset):
    columns = dataset.shape[1]
    #print(columns-1)
    for i in range(columns-1):
        dataset[:,i] = dataset[:,i]/np.max(dataset[:,i])
normalize(wine)
#print(wine)

def sgd(dataset, max_epoch, alpha):
    curr_epoch = 0
    while curr_epoch <= max_epoch:
        ex = 0
        for i in wine:
            all_data = np.array(dataset[ex])
            inputs = all_data[:-1] #only the features doesnt include the quality rating
            #print(inputs)
            input = np.insert(inputs, 0, 1.0) #x0 is always 1
            #sgd uses randomized weights and handles a vector at a time
            #print(input)
            #print(np.shape(weights))
            #print(np.shape(input))
            #print(type(input))
            
            #initializing the random weights
            weights = []
            for j in range(len(wine[0,:])): #+1 is to account for the size being increased since the first input is always 1 which i inserted
                weights.append(random()) #ranadomize float between 0 to 1
            weights = np.array(weights)
            hypothesis = np.dot((np.transpose(weights)), input) #scalar
            #print(hypothesis)
            #print(type(hypothesis))
            
            pred = dataset[ex][(len(dataset[0,:])-1)]
            #print(pred)
        
            raw_err = hypothesis - pred #scalar
            #print(raw_err)
        
            gradient = raw_err * input #1 x n
            #update weights:
        
            temp = weights
            feat = 0
            for y in weights:
                weights[feat] = temp[feat] - alpha * gradient[feat]
                feat += 1
            mse = np.dot(np.transpose(weights), input)
            mse = mse - pred
            mse = abs(mse)
            MSE.append(mse)
            
            ex += 1
        
        avgSE = sum(MSE)/ len(MSE)
    
        if curr_epoch % 7 == 0:
            print(avgSE)
        curr_epoch += 1

    print(weights)

sgd(wine, max_epoch, alpha)
#end pt 1 -----------------------------------------------------------------------------
