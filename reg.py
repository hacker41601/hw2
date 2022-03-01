#Author: Monynich Kiem
#Date: Feb. 21. 2022
#Desc: Linear regression and polynomial expansion program using csv data being read in
import numpy as np #to use the vector functions
import pandas as pd #to read the csv
from random import random
import matplotlib.pyplot as plt
#using SGD since it's allegedly faster
#HYPERPARAMETERS are the alpha and epoch-------------------------------------------
alpha = 0.001 #dr harrison said smaller numbers are better
max_epoch = 42 #epochs are just the number of iterations, chose 42 b/c nerd stuff
#-----------------------------------------------------------------------------------
dp = [] #place the array of dot products
#pt 1--------------------------------------------------------------------------
wine = pd.read_csv('winequality-red.csv')
#print(wine.head())
wine = (wine - wine.min())/(wine.max() - wine.min()) #normalizing ALL of the data frame using max min method from stack overflow
wine = pd.concat([pd.Series(1, index = wine.index, name = 'x0'), wine], axis = 1) #adding in the bias columns
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
            #print(np.shape(weights))
            hypothesis = np.dot((np.transpose(weights)), input) #scalar operation
            #print(hypothesis) maybe put hypothesis on y axis and use that
            #print(type(hypothesis))
            #dataset.insert(exp_ind, newcol, 0)
            raw_err = hypothesis - pred #scalar operation
            #print(raw_err)
        
            gradient = raw_err * input #multiply raw error by all the inputs/x_i
            #print("GRADIENT: ", gradient)
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

print("----------------------------~WINE~:---------------------------- \n")
sgd(wine, max_epoch, alpha)
#pt 2 --------------------------------------------------------------------------------------------------------
#this part is using polynomial regression with basis expansion
synth12 = pd.read_csv('synthetic-1.csv', header = None)
synth13 = pd.read_csv('synthetic-1.csv', header = None)
synth15 = pd.read_csv('synthetic-1.csv', header = None)

synth22 = pd.read_csv('synthetic-2.csv', header = None)
synth23 = pd.read_csv('synthetic-2.csv', header = None)
synth25 = pd.read_csv('synthetic-2.csv', header = None)

synth12 = pd.concat([pd.Series(1, index = synth12.index, name = 'x0'), synth12], axis = 1)
synth13 = pd.concat([pd.Series(1, index = synth13.index, name = 'x0'), synth13], axis = 1)
synth15 = pd.concat([pd.Series(1, index = synth15.index, name = 'x0'), synth15], axis = 1)

synth22 = pd.concat([pd.Series(1, index = synth22.index, name = 'x0'), synth22], axis = 1)
synth23 = pd.concat([pd.Series(1, index = synth23.index, name = 'x0'), synth23], axis = 1)
synth25 = pd.concat([pd.Series(1, index = synth25.index, name = 'x0'), synth25], axis = 1)

synth12.columns = ['x0', 'input', 'quality']
synth13.columns = ['x0', 'input', 'quality']
synth15.columns = ['x0', 'input', 'quality']

synth22.columns = ['x0', 'input', 'quality']
synth23.columns = ['x0', 'input', 'quality']
synth25.columns = ['x0', 'input', 'quality']

#print(synth1)
#print(synth2)

def basis_exp(dataset, order): #the orders or 2, 3, and 5
#expand dataset to their respective orders
#use exp_ind = 1 bc i already inserted a thing of 1's into my dataframe 
    ex = 0
    exp_ind = 1
    newcol = 1
    for x in range(order - 1):
        exp_ind+=1
        #dataset[2 + exp_ind] = 0
        newcol += 1
        dataset.insert(exp_ind, newcol, 0)
        #print(dataset)
        #expanding in accordance to order
        #use pandas concat instead and see if it works
    for m in range(len(dataset)):
        ind = 1
        og = dataset.iloc[ex, ind] #first input piece of data
        #print(og)
        for x in range(order - 1):
            ind+=1
            ordered = og ** (ind)
            dataset.iloc[ex,ind] = ordered
        ex += 1
        
    return dataset

basis_exp(synth12, 2)
basis_exp(synth22, 2)

basis_exp(synth13, 3)
basis_exp(synth23, 3)

basis_exp(synth15, 5)
basis_exp(synth25, 5)

#print(type(synth15))
#print(synth15)

synth12.to_csv('newSynth1-2.csv', index = False)
synth12 = pd.read_csv('newSynth1-2.csv')
synth22.to_csv('newSynth2-2.csv', index = False)
synth22 = pd.read_csv('newSynth2-2.csv')
    
print("----------------------------Synth1-2:---------------------------- \n")
#under 35
sgd(synth12, max_epoch, alpha)
print("----------------------------Synth2-2:---------------------------- \n")
#.5 and under
sgd(synth22, max_epoch, alpha)
#print(synth12)
#print(synth22)
synth13.to_csv('newSynth1-3.csv', index = False)
synth13 = pd.read_csv('newSynth1-3.csv')
synth23.to_csv('newSynth2-3.csv', index = False)
synth23 = pd.read_csv('newSynth2-3.csv')

print("----------------------------Synth1-3:---------------------------- \n")
#under 10
sgd(synth13, max_epoch, alpha)
print("----------------------------Synth2-3:---------------------------- \n")
#.5 and under
sgd(synth23, max_epoch, alpha)
#print(synth13)
#print(synth23)
#print(synth1)
#benchmarks for part 1: 35, 10, 10 got this down
#becnhmarks for part 2: .5, .5, .5 do not got this down
synth15.to_csv('newSynth1-5.csv', index = False)
synth15 = pd.read_csv('newSynth1-5.csv')
synth25.to_csv('newSynth2-5.csv', index = False)
synth25 = pd.read_csv('newSynth2-5.csv')


print("----------------------------Synth1-5:---------------------------- \n")
#under 10
sgd(synth15, max_epoch, alpha)
print("----------------------------Synth2-5:---------------------------- \n")
#.5 and under
sgd(synth25, max_epoch, alpha)

#print(synth15)
#print(synth25)
