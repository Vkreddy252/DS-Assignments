# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:44:46 2023

@author: Vinu
"""
# Reading the file 
import numpy as np
import pandas as pd
df0 = pd.read_csv("Salary_Data.csv")
df0

# Sqrt Transformation
df = np.sqrt(df0)

# scatter plot (EDA) Without Sqrt Transformation
import matplotlib.pyplot as plt
plt.scatter(x=df0["YearsExperience"], y=df0["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Scatter Plot before Sqrt Transformation")
plt.show()

# scatter plot (EDA) With Sqrt Transformation
import matplotlib.pyplot as plt
plt.scatter(x=df["YearsExperience"], y=df["Salary"],color='green')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Scatter Plot After Sqrt Transformation")
plt.show()

df.corr()

# split the variable as X and Y
Y = df["Salary"]
X = df[["YearsExperience"]]

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
plt.scatter(x=df["YearsExperience"], y=df["Salary"])
plt.scatter(x=df["YearsExperience"], y=Y_pred)
plt.plot(df["YearsExperience"], Y_pred, color='black')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
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


