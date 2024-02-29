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

df = pd.read_csv("Glass.csv")
df.head()
df.shape
#==============================================================================
# Step-2:  EDA

df.duplicated().sum()
df1=df.drop_duplicates().reset_index(drop=True)
df1.duplicated().sum()
df1.isnull().sum()
df1.describe().round(2) 
df1.info()

# Correlation   
cor = df1.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()
    
# Box Plots and Histograms for Independent Variables

cols = ['RI','Na', 'Mg','Al', 'Si','K','Ca','Ba','Fe','Type']

# Box Plots
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(4,3,i)
    sns.boxplot(y=col, data=df1,width=0.2)
    i+=1
plt.show()

# Histograms
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(4,3,i)
    sns.histplot(x=col, data=df1,kde=True,color='blue')
    i+=1
plt.show()

# Pair plot
sns.pairplot(data=df1, hue = 'Type')

#==============================================================================
# Step-3: Data Transformation

X = df1.iloc[:,0:9]

# Standardization for X 

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
X1 = pd.DataFrame(SS_X)
X1.columns =list(X)
X1.head()

Y = df1["Type"]

#==============================================================================
# Step-4:  Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y,train_size=0.75,random_state=10)

#==============================================================
# Step-5:  Model Fitting

# Finding Best Parameters Using Grid Search CV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 

param_grid = ({'n_neighbors':range(1,20),'p':range(1,10)})
Model = KNeighborsClassifier()
grid=GridSearchCV(Model,param_grid)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)

# Fitting the Model with Best Parameters

KNN = KNeighborsClassifier(n_neighbors=4,p=5)
KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train) 
Y_pred_test = KNN.predict(X_test) 

#==============================================================
# Step-6: Metrics
from sklearn.metrics import accuracy_score
ac1= accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:", ac1.round(3))
ac2= accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:", ac2.round(3))

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
    
    



