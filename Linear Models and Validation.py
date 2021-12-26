#!/usr/bin/env python
# coding: utf-8

# # PS2: Linear Models and Validation

# **Your Name:** [Sara Honarvar]
# 
# **People I discussed this assignment with:** [None]

# # Preamble
# We'll be loading some CO2 concentration data that is a commonly used dataset for model building of time series prediction. You will build a few baseline linear models and assess them using some of the tools we discussed in class. Which model is best? Let's find out.
# 
# First let's just load the data and take a look at it:

# In[281]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from datetime import datetime, timedelta
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set_context('notebook')

# Fetch the data 
mauna_lao = fetch_openml('mauna-loa-atmospheric-co2')
print(mauna_lao.DESCR)
data = mauna_lao.data
# Assemble the day/time from the data columns so we can plot it
d1958 = datetime(year=1958,month=1,day=1)
time = [datetime(int(d[0]),int(d[1]),int(d[2])) for d in data] 
X = np.array([1958+(t-d1958)/timedelta(days=365.2425) for t in time]).T
X = X.reshape(-1,1)  # Make it a column to make scikit happy
y = np.array(mauna_lao.target)
X


# In[282]:


X[:,np.newaxis]


# In[283]:


d = 2
polynomial_features = PolynomialFeatures(degree=d,
                                             include_bias=False)
# Construct linear regression model
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
# Now fit the data first through the 
# polynomial basis, then do regression
pipeline.fit(X, y)

# Get the accuracy score of the trained model
# on the original training data
score = pipeline.score(X,y)


# In[284]:


score


# In[285]:


#Plot the results
X_plot = X
plt.plot(X_plot, pipeline.predict(X_plot), label="Model")
plt.scatter(X, y, label="Samples")
plt.xlabel("x")
plt.ylabel("y")

plt.legend(loc="best")

# Print the polynomial degree and the training
# accuracy in the title of the graph
plt.title("Degree {}\nTrain score = {:.3f}".format(
    d, score))
plt.show()  


# In[286]:


# Plot the data
plt.figure(figsize=(10,5))    # Initialize empty figure
plt.scatter(X, y, c='k',s=1) # Scatterplot of data
plt.xlabel("Year")
plt.ylabel(r"CO$_2$ in ppm")
plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
plt.tight_layout()
plt.show()


# # Linear Models

# Construct the following linear models:
# 1. Model 1: "Vanilla" Linear Regression, that is, where $CO_2 = a+b \cdot time$
# 2. Model 2: Quadratic Regression, where $CO_2 = a+b \cdot t + c\cdot t^2$
# 3. Model 3: A more complex "linear" model with the following additive terms $CO_2=a+b\cdot t+c\cdot sin(\omega\cdot t)$:
#   * a linear (in time) term
#   * a sinusoidal additive term with period such that the peak-to-peak of the sinsusoid is roughly ~1 year and phase shift of zero (set $\omega$ as appropriate to match the peaks)
# 4. Model 4: A "linear" model with the following additive terms ($CO_2=a+b\cdot t+c\cdot t^2+d\cdot sin(\omega\cdot t)$:
#   * a quadratic (in time) polynomial
#   * a sinusoidal additive term with period such that the peak-to-peak of the sinsusoid is roughly ~1 year and phase shift of zero (set $\omega$ as appropriate to match the peaks)
#   
# Evauate these models using **the appropriate kind of Cross Validation** for each of the following amounts of Training data:
# 1. N=50 Training Data Points
# 2. N=100
# 3. N=200
# 4. N=500
# 5. N=1000
# 6. N=2000

# **Question**: Before you even construct the models or do any coding below, what is your initial guess or intuition behind how each of those four models will perform? Note: there is no right or wrong answer to this part of the assignment and this question will only be graded on completeness, not accuracy. It's intent is to get you to think about and write down your preliminary intuition regarding what you think will happen before you actually implement anything, based on your approximate understanding of how functions of the above complexity *should* perform as N increases.
# 
# **Student Response:** [By just looking at the visualization for CO$_2$ in ppm over years, we can see an upward trend as well as periodicity structural pattern in the data. More particularly, we can see a periodic motion trend with a period approximately 1 year. So, for building a model we can use a line to predict future but it might be better to add a sinosoidal trend to the line. With this explanation, I guess with low number of training data points, a line would be fine, but with low values of N, I believe the third model could capture the trend better than the first and second models. I guess the 4th model would get better as N increases but it might lead to overfitting with high number of N. Overall, I think the third model will have the best performance. ]

# **Question**: What is the appropriate kind of Cross Validation to perform in this case if we want a correct Out of Sample estimate of our Test MSE?
# 
# **Student Response:** [Since we have time dependency in the data, and we are interested in predicting CO$_2$ as a function of time, time series split would sound a good CV method for an unbiased estimate of out of sample error.]

# Now, for each of the above models and training data sizes:
# * Plot the predicted CO2 as a function of time, including the actual data, for each of the N=X training data examples. This should correspond to six plots (one for each amount of training data) if you plot all models on the same plot, or 6x4 = 24 plots if you plot each model and training data plot separately.
# * Create a [Learning Curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) plot for the model which plots its Training and Test MSE as a function of training data. That is, plot how Training and Testing MSE change as you increase the training data for each model. This could be a single plot for all four models (8 lines on the plot) or four different plots corresponding to the learning curve of each model separately.

# In[287]:


# 

# Insert Modeling Building or Plotting code here
# Note, you may implement these however you see fit
# Ex: using an existing library, solving the Normal Eqns
#     implementing your own SGD solver for them. Your Choice.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve


# # Model 1: "Vanilla" Linear Regression, that is, where $CO_2 = a+b \cdot time$

# In[288]:


N = [50,100,200,500,1000,2000]

XX = X #features
yy = y 

MSE_train = np.zeros(len(N))
MSE_test = np.zeros(len(N))
STD_train = np.zeros(len(N))
STD_test = np.zeros(len(N))
predicted_CO2 = np.zeros((6,len(XX)))
num_splits = 20
for k, train_num in enumerate(N):
    tscv = TimeSeriesSplit(max_train_size=train_num, n_splits=num_splits)
    training_score = np.zeros(num_splits)
    testing_score = np.zeros(num_splits)
    
    i=0
    for train_index, test_index in tscv.split(XX):
        X_train, X_test = XX[train_index], XX[test_index] 
        y_train, y_test = yy[train_index], yy[test_index] 

        polynomial_features = PolynomialFeatures(degree=1,
                                             include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

        pipeline.fit(X_train, y_train)
        
        training_score[i] = mean_squared_error(y_train,pipeline.predict(X_train))
        testing_score[i]  = mean_squared_error(y_test,pipeline.predict(X_test))
    
        i = i+1

        pipeline.predict(X_test)
        
    # For calculating MSE values, I got the average across num of splits to smooth out the values. However,
    #for predicting CO2, I just plotted the last one to better visualize the data
    MSE_train[k] = np.mean(training_score) 
    MSE_test[k] = np.mean(testing_score)
    STD_train[k] = np.std(training_score) 
    STD_test[k] = np.std(testing_score)
    predicted_CO2[k,:] = pipeline.predict(XX)

    plt.figure(figsize=(5, 5))
    #plt.subplot(3, 3, k+1)
    plt.plot(X, pipeline.predict(XX), label="Model1",color='black')
    plt.scatter(X, y, label="actual data",color='g')
    plt.xlabel("year")
    plt.ylabel(r"CO$_2$ in ppm")

    plt.legend(loc="best")
    
    plt.title("\nN = {:}".format(train_num))

    plt.show()


# In[289]:


MSE_test


# Learning curve for Model 1

# In[290]:


MSE_test
plt.figure(figsize=(5, 5))
plt.semilogy(N, MSE_train, label="training_score",color='red')
plt.semilogy(N, MSE_test, label="testing_score",color='green')
plt.legend(loc="best")
plt.title("Learning curve for Model 1")
plt.xlabel("Training number")
plt.ylabel("score")
plt.show()


# # Model 2: Quadratic Regression, where $CO_2 = a+b \cdot t + c\cdot t^2$

# In[291]:


N = [50,100,200,500,1000,2000]

XX = np.hstack([X,np.square(X)]) #features
yy = y 

MSE_train = np.zeros(len(N))
MSE_test = np.zeros(len(N))
STD_train = np.zeros(len(N))
STD_test = np.zeros(len(N))
predicted_CO2 = np.zeros((6,len(XX)))
num_splits = 20
for k, train_num in enumerate(N):
    tscv = TimeSeriesSplit(max_train_size=train_num, n_splits=num_splits)
    training_score = np.zeros(num_splits)
    testing_score = np.zeros(num_splits)
    
    i=0
    for train_index, test_index in tscv.split(XX):
        X_train, X_test = XX[train_index], XX[test_index] 
        y_train, y_test = yy[train_index], yy[test_index] 

        polynomial_features = PolynomialFeatures(degree=1,
                                             include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

        pipeline.fit(X_train, y_train)
        
        training_score[i] = mean_squared_error(y_train,pipeline.predict(X_train))
        testing_score[i]  = mean_squared_error(y_test,pipeline.predict(X_test))
    
        i = i+1

        pipeline.predict(X_test)
        
    # For calculating MSE values, I got the average across num of splits to smooth out the values. However,
    #for predicting CO2, I just plotted the last one to better visualize the data
    MSE_train[k] = np.mean(training_score) 
    MSE_test[k] = np.mean(testing_score)
    STD_train[k] = np.std(training_score) 
    STD_test[k] = np.std(testing_score)
    predicted_CO2[k,:] = pipeline.predict(XX)

    plt.figure(figsize=(5, 5))
    #plt.subplot(3, 3, k+1)
    plt.plot(X, pipeline.predict(XX), label="Model2",color='black')
    plt.scatter(X, y, label="actual data",color='g')
    plt.xlabel("year")
    plt.ylabel(r"CO$_2$ in ppm")

    plt.legend(loc="best")
    
    plt.title("\nN = {:}".format(train_num))

    plt.show()


# In[292]:


MSE_test


# Learning curve for Model 2

# In[293]:


plt.figure(figsize=(5, 5))
plt.semilogy(N, MSE_train, label="training_score",color='red')
plt.semilogy(N, MSE_test, label="testing_score",color='green')
plt.legend(loc="best")
plt.title("Learning curve for Model 2")
plt.xlabel("Training number")
plt.ylabel("score")
plt.show()


# # Model 3: A more complex "linear" model with the following additive terms $CO_2=a+b\cdot t+c\cdot sin(\omega\cdot t)$:

# In[294]:


N = [50,100,200,500,1000,2000]
w = 2*np.pi
XX = np.hstack([X,np.sin(w*X)]) #features
yy = y 

MSE_train = np.zeros(len(N))
MSE_test = np.zeros(len(N))
STD_train = np.zeros(len(N))
STD_test = np.zeros(len(N))
predicted_CO2 = np.zeros((6,len(XX)))
num_splits = 20
for k, train_num in enumerate(N):
    tscv = TimeSeriesSplit(max_train_size=train_num, n_splits=num_splits)
    training_score = np.zeros(num_splits)
    testing_score = np.zeros(num_splits)
    
    i=0
    for train_index, test_index in tscv.split(XX):
        X_train, X_test = XX[train_index], XX[test_index] 
        y_train, y_test = yy[train_index], yy[test_index] 

        polynomial_features = PolynomialFeatures(degree=1,
                                             include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

        pipeline.fit(X_train, y_train)
        
        training_score[i] = mean_squared_error(y_train,pipeline.predict(X_train))
        testing_score[i]  = mean_squared_error(y_test,pipeline.predict(X_test))
    
        i = i+1

        pipeline.predict(X_test)
        
    # For calculating MSE values, I got the average across num of splits to smooth out the values. However,
    #for predicting CO2, I just plotted the last one to better visualize the data
    MSE_train[k] = np.mean(training_score) 
    MSE_test[k] = np.mean(testing_score)
    STD_train[k] = np.std(training_score) 
    STD_test[k] = np.std(testing_score)
    predicted_CO2[k,:] = pipeline.predict(XX)

    plt.figure(figsize=(5, 5))
    #plt.subplot(3, 3, k+1)
    plt.plot(X, pipeline.predict(XX), label="Model3",color='black')
    plt.scatter(X, y, label="actual data",color='g')
    plt.xlabel("year")
    plt.ylabel(r"CO$_2$ in ppm")

    plt.legend(loc="best")
    
    plt.title("\nN = {:}".format(train_num))

    plt.show()


# In[295]:


MSE_test


# Learning curve for model 3

# In[296]:


plt.figure(figsize=(5, 5))
plt.semilogy(N, MSE_train, label="training_score",color='red')
plt.semilogy(N, MSE_test, label="testing_score",color='green')
plt.legend(loc="best")
plt.title("Learning curve for Model 3")
plt.xlabel("Training number")
plt.ylabel("score")
plt.show()


# # Model 4: A "linear" model with the following additive terms ($CO_2=a+b\cdot t+c\cdot t^2+d\cdot sin(\omega\cdot t)$:
# 

# In[297]:


N = [50,100,200,500,1000,2000]
w = 2*np.pi
XX = np.hstack([X,np.square(X),np.sin(w*X)]) #features
yy = y 

MSE_train = np.zeros(len(N))
MSE_test = np.zeros(len(N))
STD_train = np.zeros(len(N))
STD_test = np.zeros(len(N))
predicted_CO2 = np.zeros((6,len(XX)))
num_splits = 20
for k, train_num in enumerate(N):
    tscv = TimeSeriesSplit(max_train_size=train_num, n_splits=num_splits)
    training_score = np.zeros(num_splits)
    testing_score = np.zeros(num_splits)
    
    i=0
    for train_index, test_index in tscv.split(XX):
        X_train, X_test = XX[train_index], XX[test_index] 
        y_train, y_test = yy[train_index], yy[test_index] 

        polynomial_features = PolynomialFeatures(degree=1,
                                             include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

        pipeline.fit(X_train, y_train)
        
        training_score[i] = mean_squared_error(y_train,pipeline.predict(X_train))
        testing_score[i]  = mean_squared_error(y_test,pipeline.predict(X_test))
    
        i = i+1

        pipeline.predict(X_test)
        
    # For calculating MSE values, I got the average across num of splits to smooth out the values. However,
    #for predicting CO2, I just plotted the last one to better visualize the data
    MSE_train[k] = np.mean(training_score) 
    MSE_test[k] = np.mean(testing_score)
    STD_train[k] = np.std(training_score) 
    STD_test[k] = np.std(testing_score)
    predicted_CO2[k,:] = pipeline.predict(XX)

    plt.figure(figsize=(5, 5))
    #plt.subplot(3, 3, k+1)
    plt.plot(X, pipeline.predict(XX), label="Model4",color='black')
    plt.scatter(X, y, label="actual data",color='g')
    plt.xlabel("year")
    plt.ylabel(r"CO$_2$ in ppm")

    plt.legend(loc="best")
    
    plt.title("\nN = {:}".format(train_num))

    plt.show()


# In[298]:


MSE_test


# Learning curve for model 4

# In[299]:


plt.figure(figsize=(5, 5))
plt.semilogy(N, MSE_train, label="training_score",color='red')
plt.semilogy(N, MSE_test, label="testing_score",color='green')
plt.legend(loc="best")
plt.title("Learning curve for Model 4")
plt.xlabel("Training number")
plt.ylabel("score")
plt.show()


# **Question**: Which Model appears to perform best in the N=50 or N=100 Condition? Why is this?
# 
# **Student Response:** [Model 3 performs the best in N=50 or N=100. Because it has the closest form to the actual data. The initial visualization gives us an idea that the data has an upwarding trend with a periodic behavior. That's why even with low number of training, the 3rd model performs better than the other. ]
# 
# **Question**: Which Model appears to perform best as the N=200 to 500? Why is this?
# 
# **Student Response:** [The second model! The MSE_test for the third model is still lower than all of them, and the 4th model has a decreasing trend and low values of MSE_test (3.6 to 1.94) as N goes from 200 to 500. However, for the second model the error decreases from around 8 to 5. This could be because of the quadratic term that allows for capturing the data better.   ]
# 
# **Question**: Which Model appears to perform best as N = 2000? Why is this?
# 
# **Student Response:** [The fourth model. Since it has all the required components to capture all the features of the data and since the training data is large, the coefficients of linear regression model can be found with better accuracy. However, this model with this many training data (almost all data) is prone to overfitting.]

# In[ ]:




