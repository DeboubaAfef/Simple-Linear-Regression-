# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#SIMPLE LINEAR REGRESSION
# Importing libraries
import matplotlib.pyplot as plt #using for show the plots of variables
import pandas as pd #using for reading and know some informations about the data

# Read the data 
data = pd.read_csv('Heights_weights.csv') #'read'funtion for read the data and 'csv" means it's an Excel forme(i downloded the data from kaggle)
#data.info() #informations about the data :type,shape,...
#print('data.describe =' ,data.describe()) #description of the data : mean,mmax , min , ...
# separate the variables into two matrices with the command "iloc"
x = data.iloc[:,:-1].values #valeurs de toutes les lignes et la colonne (data sous forme de dictionnaire)
y = data.iloc[:, -1].values #valeurs de toutes les lignes et la dernière colonne 

#importing module " train_test_split" from the librairie "sklearn.model_selection" for split the data into sets :training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0) #Two sets(train and test) from x&y
#the shape of datatest is about 1/3 of the main data.

# Training the Simple Linear Regression model on the Training set:
from sklearn.linear_model import LinearRegression #choose type of regression as a simple linear regression 
regressor = LinearRegression() #variable plays the role of a Linear regression
regressor.fit(x_train, y_train) #affet the train sets 

# Predicting the Test set results by using the function "predict" from LinearRegression to know the learning rate of the program 
y_pred = regressor.predict(x_test)



# Visualising the Training set results by using the librairie "matplotlib.pyplot"
plt.scatter(x_train, y_train, color = 'red') #plotting the sets as scatter form and red color 
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #plotting the linear line which represente the x_train and her prediction
plt.title('Heights vs Weights (Training set)') #named the plot 
plt.xlabel('Heights') #name the labels
plt.ylabel('Weights')
plt.show() #show the plot


# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Heights vs Weights(Test set)')
plt.xlabel('Heigths')
plt.ylabel('Weights')
plt.show()


#testing my code by some values in cm
y_pred2= regressor.predict([[1.59]])
print("the predictive value of weigth in Kg is :",y_pred2)
print("----------------------")
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test)) 