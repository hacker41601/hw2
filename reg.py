#Author: Monynich Kiem
#Date: Feb. 21. 2022
#Desc: Linear regression and polynomial expansion program using csv data being read in
import numpy as np #to use the vector functions
import pandas as pd #to read the csv
from random import random
import matplotlib.pyplot as plt
#using SGD since it's allegedly faster
#HYPERPARAMETERS are the alpha and epoch-------------------------------------------
alpha = 0.1 #dr harrison said smaller numbers are better
max_epoch = 126
#-----------------------------------------------------------------------------------
#pt 1--------------------------------------------------------------------------
wine = pd.read_csv('winequality-red.csv')
#print(wine.head())
wine = (wine - wine.min())/(wine.max() - wine.min()) #normalizing ALL of the data frame using max min method from stack overflow
wine.insert(0, 'x0', 1) #adding in the bias columns

def sgd(dataset, max_epoch, alpha):
    curr_epoch = 0
    
    while curr_epoch <= max_epoch:
        ex = 0 #the examples are to iterator through each row at a time
        x = dataset.drop(columns = 'quality') #input variables only
        y = dataset.iloc[:,(len(dataset.columns)-1)] #output variable only
        for data in dataset:
            input = x.loc[ex]
            pred = y.loc[ex]
            #sgd uses randomized weights and handles a vector/datapoint at a time
            #initializing the random weights
            #weights = np.random.uniform(0,1,len(input))
            #1x12 * 12x1
            weights = np.random.uniform(0,1,len(input))
            #print(np.shape(weights))
            hypothesis = np.dot((np.transpose(weights)), input) #scalar operation
            raw_err = hypothesis - pred #scalar operation
        
            gradient = raw_err * input #multiply raw error by all the inputs/x_i
            #update weights:
            gradient = gradient.to_numpy() #fixed key error!!
            update = weights
            feat = 0
            for w in weights:
                weights[feat] = update[feat] - alpha * gradient[feat]
                feat += 1
            
            #print(weights)
            dp = [] #place the array of dot products
            dot_prod = np.dot(np.transpose(weights), input)
            dot_prod = (dot_prod - pred)**2
            dp.append(dot_prod)
            #print(dp)
            
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
sgd(wine, max_epoch, .15)
#pt 2 --------------------------------------------------------------------------------------------------------
#this part is using polynomial regression with basis expansion
synth1 = pd.read_csv('synthetic-1.csv', header = None)
synth2 = pd.read_csv('synthetic-2.csv', header = None)

synth1.insert(0, 'x0', 1)
synth2.insert(0, 'x0', 1)

synth1.columns = ['x0', 'input', 'quality']
synth2.columns = ['x0', 'input', 'quality']

#print(synth1)

def basis_exp(df, order): #the orders or 2, 3, and 5
#expand dataset to their respective orders
#use exp_ind = 1 bc i already inserted a thing of 1's into my dataframe 
    ex = 0
    newcol = 2
    dataset = df.copy(deep = True)
    for x in range(order - 1):
        #dataset[2 + exp_ind] = 0
        dataset.insert(newcol, newcol, 0)
        newcol += 1
        #print(dataset)
        #expanding in accordance to order
        #use pandas concat instead and see if it works
    for m in range(len(dataset)):
        ind = 1
        og = dataset.iloc[ex, ind] #first input piece of data
        #print(og)
        for x in range(order - 1):
            ind+=1
            ordered = (og) ** (ind)
            dataset.iloc[ex,ind] = ordered
        ex += 1
        
    return dataset

b12 = basis_exp(synth1, 2)
b22 = basis_exp(synth2, 2)

b13 = basis_exp(synth1, 3)
b23 = basis_exp(synth2, 3)

b15 = basis_exp(synth1, 5)
b25 = basis_exp(synth2, 5)
#print(type(synth15))
#print(synth15)
#print(b15)
#print(b25)

#SYNTHETIC DATA 2
print("----------------------------Synth1-2:---------------------------- \n")
#under 35
sgd(b12, max_epoch, alpha)
print("----------------------------Synth1-3:---------------------------- \n")
#under 10
sgd(b13, max_epoch, alpha)
print("----------------------------Synth1-5:---------------------------- \n")
#under 10
sgd(b15, max_epoch, .3)

#SYNTHETIC DATA 2
print("----------------------------Synth2-2:---------------------------- \n")
#.5 and under
sgd(b22, max_epoch, alpha)
#print(synth12)
#print(synth22)
#b22 has a larger MSE for some reason will need to adjust alpha
print("----------------------------Synth2-3:---------------------------- \n")
#.5 and under
sgd(b23, max_epoch, alpha)
#benchmarks for part 1: 35, 10, 10 got this down
#becnhmarks for part 2: .5, .5, .5 do not got this down
print("----------------------------Synth2-5:---------------------------- \n")
#.5 and under
sgd(b25, max_epoch, .15)

'''
b12.to_csv('synth12.csv', index = False)
b13.to_csv('synth13.csv', index = False)
b15.to_csv('synth15.csv', index = False)

b22.to_csv('synth22.csv', index = False)
b23.to_csv('synth23.csv', index = False)
b25.to_csv('synth25.csv', index = False)
'''
'''
x = b15[5]
y = b15['quality']

mymodel = np.poly1d(np.polyfit(x, y, 5))
myline = np.linspace(-30, 30, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
'''
