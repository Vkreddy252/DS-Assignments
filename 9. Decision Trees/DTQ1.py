# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:41:46 2023

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

df = pd.read_csv("Company_Data.csv")
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
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()
    
# Box Plots and Histograms for Independent Variables

cols = ['Sales','CompPrice', 'Income','Advertising', 'Population','Price','Age','Education']

# Box Plots
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(3,3,i)
    sns.boxplot(y=col, data=df,width=0.2)
    i+=1
plt.show()

# Histograms
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(3,3,i)
    sns.histplot(x=col, data=df,kde=True,color='blue')
    i+=1
plt.show()

# Pair plot
sns.pairplot(data=df, hue = 'ShelveLoc')

#==============================================================================
# Step-3: Data Transformation
# Coverting Sales into Categorical Variable

df1=df.copy()
df1['Sale'] = pd.cut(x = df1['Sales'], bins = [0,6.03,8.67,16.27], labels=['Low','Medium','High'], right = False)
df1.head()
df1.info()

# Label Encoding For Categorical Columns

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df1["Urban"] = LE.fit_transform(df1["Urban"])
df1["US"] = LE.fit_transform(df1["US"])
df1["ShelveLoc"] = LE.fit_transform(df1["ShelveLoc"])
df1["Sale"] = LE.fit_transform(df1["Sale"])
X = df1.iloc[:,0:11]

# Standardization for X 

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
X1 = pd.DataFrame(SS_X)
X1.columns =list(X)
X1.head()

Y = df1["Sale"]

#==============================================================================
# Step-4:  Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y,train_size=0.75)

#==============================================================================
# Step-5 Model Fitting
# Finding Best Parameters Using Grid Search CV

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV 

param_grid = ({'max_depth':range(1,10),'criterion':['gini','entropy'],'max_leaf_nodes':range(1,10),'min_samples_leaf':range(1,10)})
Model = DecisionTreeClassifier()
grid=GridSearchCV(Model,param_grid)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)

# Applying Parameters to Model

DT = DecisionTreeClassifier(criterion='gini', max_depth=2,max_leaf_nodes=3,min_samples_leaf=1)
DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train) 
Y_pred_test = DT.predict(X_test) 

#==============================================================================
# Step-6: # Metrics
from sklearn.metrics import accuracy_score
ac1= accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:", ac1.round(4)*100)
ac2= accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:", ac2.round(4)*100)

print("Number of Nodes",DT.tree_.node_count)
print("Level of Depth",DT.tree_.max_depth)


