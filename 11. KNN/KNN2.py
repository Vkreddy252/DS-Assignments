# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:25:41 2023

@author: Vinu
"""
#==============================================================================
# Step-1:  Importing the Required Libraries and file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Zoo.csv")
df.head()
df.shape
#==============================================================================
# Step-2:  EDA

df.duplicated().sum()
df.isnull().sum()
df.describe().round(2) 
df.info()

# Correlation   
cor = df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()
    
# Box Plots and Histograms for Independent Variables

cols = ['hair','feathers', 'eggs','milk', 'airborne','aquatic','predator','toothed',
        'backbone','breathes','venomous','fins','legs','tail','domestic','catsize','type']

# Box Plots
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(6,3,i)
    sns.boxplot(y=col, data=df,width=0.2)
    i+=1
plt.show()

# Histograms
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(6,3,i)
    sns.histplot(x=col, data=df,kde=True,color='blue')
    i+=1
plt.show()

#==============================================================================
# Step-3: Data Transformation

X = df.iloc[:,1:16]
# Standardization for X 

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
X1 = pd.DataFrame(SS_X)
X1.columns =list(X)
X1.head()

Y = df["type"]

#==============================================================================
# Step-4:  Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y,train_size=0.7,random_state=1)

#==============================================================
# Step-5:  Model Fitting

# Finding Best Parameters Using Grid Search CV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 

param_grid = ({'n_neighbors':range(1,40),'p':range(1,10)})
Model = KNeighborsClassifier()
grid=GridSearchCV(Model,param_grid)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)

# Fitting the Model with Best Parameters

KNN = KNeighborsClassifier(n_neighbors=1,p=1)
KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train) 
Y_pred_test = KNN.predict(X_test) 

#==============================================================
# Step-6: Metrics
from sklearn.metrics import accuracy_score
ac1= accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:", ac1.round(4))
ac2= accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:", ac2.round(4))

#==============================================================
# Step-7 Visualizing Results
    
k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    train_accuracy.append(knn.score(X_train, Y_train))
    test_accuracy.append(knn.score(X_test, Y_test))
    
# Plot
plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
    
    



