# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:44:46 2023

@author: Vinu
"""
# Reading the file 
import numpy as np
import pandas as pd
df0 = pd.read_csv("delivery_time.csv")
df0

# Log Transformation
df = np.log(df0)

# scatter plot (EDA) Without Log Transformation
import matplotlib.pyplot as plt
plt.scatter(x=df0["Sorting Time"], y=df0["Delivery Time"])
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Scatter Plot Without Log Transformation")
plt.show()

# scatter plot (EDA) With Log Transformation
import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"], y=df["Delivery Time"],color='red')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Scatter Plot After Log Transformation")
plt.show()
df.corr()

# split the variable as X and Y
Y = df["Delivery Time"]
X = df[["Sorting Time"]]

# fitting the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y) # Bo + B1x1

LR.intercept_ # Bo
LR.coef_   #B1

# calc y_pred
Y_pred = LR.predict(X)
Y_pred

# plt the scatter plot with y_pred
import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"], y=df["Delivery Time"])
plt.scatter(x=df["Sorting Time"], y=Y_pred)
plt.plot(df["Sorting Time"], Y_pred, color='green')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Scatter Plot After Model Fitting")
plt.show()

# metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean squared Error:", mse.round(3))
print("Root Mean squared Error:", np.sqrt(mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
print("R square:", r2.round(3))


