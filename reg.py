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
#MSE = []
dp = [] #place the array of dot products

#pt 1--------------------------------------------------------------------------
wine = pd.read_csv('winequality-red.csv')
#print(wine.head())
wine=(wine-wine.min())/(wine.max()-wine.min()) #normalizing ALL of the data frame using max min method from stack overflow
wine = pd.concat([pd.Series(1, index = wine.index, name = 'x0'), wine], axis = 1) #adding in the bias columns
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
        ex = 0 #the examples are to iterator through each row at a time
        
        for data in dataset:
            x = dataset.drop(columns = 'quality') #input variables only
            input = x.loc[ex]
            y = dataset.iloc[:,(len(dataset.columns)-1)] #output variable only
            pred = y.loc[ex] #this is the prediction/output
            #print(pred)
            #sgd uses randomized weights and handles a vector/datapoint at a time
            #print(input)
            #print(np.shape(weights))
            #print(np.shape(input))
            #print(type(input))
            
            #initializing the random weights
            weights = []
            for w in range(len(input)):
                weights.append(random()) #ranadomize float between 0 to 1 b/c sigmoid or smth
            weights = np.array(weights) #turn into an array so i can use the dot and transpose functions in numpy
            
            hypothesis = np.dot((np.transpose(weights)), input) #scalar operation
            #print(hypothesis)
            #print(type(hypothesis))
        
            raw_err = hypothesis - pred #scalar operation
            #print(raw_err)
        
            gradient = raw_err * input #multiply raw error by all the inputs/x_i
            
            #update weights:
            update = weights
            feat = 0
            for w in weights:
                weights[feat] = update[feat] - alpha * gradient[feat]
                feat += 1
            
            #print(weights)
            dot_prod = np.dot(np.transpose(weights), input)
            dot_prod = abs(dot_prod - pred)
            dp.append(dot_prod)
            
            ex += 1
        
        #1/m not 1/2m
        MSE = sum(dp)/ len(dp)
        print(MSE)
        curr_epoch += 1
    
    print("MEAN SQUARED ERROR: ", MSE)
    print("WEIGHTS: ", weights)

sgd(wine, max_epoch, alpha)

