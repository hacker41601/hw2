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

#pt 1--------------------------------------------------------------------------
wine = pd.read_csv('winequality-red.csv')
#print(wine.head())
wine=(wine-wine.min())/(wine.max()-wine.min())
wine = pd.concat([pd.Series(1, index = wine.index, name = 'x0'), wine], axis = 1)
#print(normalized_wine)
#print(wine)
#m = len(wine)
#num_cols = len(wine.columns)
#print(len(wine.columns))
#x = wine.drop(columns = 'quality') #input variables
#print(x)
#y = wine.iloc[:,12] #output variable
#print(y)

def sgd(dataset, max_epoch, alpha):
    curr_epoch = 0
    
    while curr_epoch <= max_epoch:
        ex = 0
        
        for i in dataset:
            x = dataset.drop(columns = 'quality') #input variables
            input = x.loc[ex]
            #sgd uses randomized weights and handles a vector/datapoint at a time
            #print(input)
            #print(np.shape(weights))
            #print(np.shape(input))
            #print(type(input))
            
            #initializing the random weights
            weights = []
            for j in range(len(input)):
                weights.append(random()) #ranadomize float between 0 to 1
            weights = np.array(weights)
            
            hypothesis = np.dot((np.transpose(weights)), input) #scalar
            #print(hypothesis)
            #print(type(hypothesis))
            y = dataset.iloc[:,(len(dataset.columns)-1)] #output variable
            pred = y.loc[ex]
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
            
            #print(weights)
            mse = np.dot(np.transpose(weights), input)
            mse = abs(mse - pred)
            MSE.append(mse)
            
            ex += 1
        
        avgSE = sum(MSE)/ len(MSE)
        print(avgSE)
        curr_epoch += 1
    
    print("AVG SE: ", avgSE)
    print("WEIGHTS: ", weights)

sgd(wine, max_epoch, alpha)

