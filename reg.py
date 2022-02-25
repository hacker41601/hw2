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
wine = (wine - wine.min())/(wine.max() - wine.min()) #normalizing ALL of the data frame using max min method from stack overflow
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
#weight = np.random.uniform(0,1,12)
#print(weight)

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
            #print(np.shape(weights)) #12x1
            #print(np.shape(input)) #12x1
            #print(type(input))
            
            #initializing the random weights
            weights = np.random.uniform(0,1,len(input))
            #1x12 * 12x1
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
        
        m = len(dp)
        #1/m not 1/2m
        MSE = sum(dp)/ m
        #print(MSE)
        curr_epoch += 1
    
    print("MEAN SQUARED ERROR: ", MSE)
    print("WEIGHTS: ", weights)
    print(" ")

sgd(wine, max_epoch, alpha)

#pt 2 --------------------------------------------------------------------------------------------------------
#this part is using polynomial regression with basis expansion
synth1 = pd.read_csv('synthetic-1.csv', header = None)
synth2 = pd.read_csv('synthetic-2.csv', header = None)

synth1 = (synth1 - synth1.min())/(synth1.max() - synth1.min())
synth2 = (synth2 - synth2.min())/(synth2.max() - synth2.min())

synth1 = pd.concat([pd.Series(1, index = synth1.index, name = 'x0'), synth1], axis = 1)
synth2 = pd.concat([pd.Series(1, index = synth2.index, name = 'x0'), synth2], axis = 1)

synth1.columns = ['x0', 'input', 'quality']
synth2.columns = ['x0', 'input', 'quality']

#print(synth1)
#print(synth2)

def basis_exp(dataset, order): #the orders or 2, 3, and 5
    ex = 0
    exp_ind = 0
    for x in range(order - 1):
        exp_ind+=1
        dataset[2 + exp_ind] = 0
        #use pandas concat instead and see if it works
    for m in dataset:
        ind = 0
        og = dataset.iloc[ex][ind+1]
        for x in range(order - 1):
            ind+=1
            cast = float(og) ** (ind + 1)
            dataset[ex][ind] = cast
        ex += 1
        
sgd(synth1, max_epoch, alpha)
sgd(synth2, max_epoch, alpha)

print(synth1.iloc[0][1])

#lines 139 - 147 work
for m in synth1:
    ex = 0
    ind = 0
    og = synth1.iloc[ex][ind+1]
    print(og)

#inserts into 2nd column, named 1, filled with 0s
synth1.insert(2, 1, 0)
print(synth1)
#basis_exp(synth1, 5)

