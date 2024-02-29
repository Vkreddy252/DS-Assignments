# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:41:46 2023

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 

param_test = ({'n_estimators':range(100,500,100),'criterion':['gini','entropy'],'n_jobs':range(1,10)})
Model = RandomForestClassifier
grid=GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=param_test,
                        scoring='accuracy', cv=5)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)

# Applying Parameters to Model
RF = RandomForestClassifier(n_estimators=100,n_jobs=9,criterion="gini",)
RF.fit(X_train,Y_train)
Y_pred_train = RF.predict(X_train) 
Y_pred_test = RF.predict(X_test) 

#==============================================================================
# Step-6: # Metrics
from sklearn.metrics import accuracy_score
ac1= accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:", ac1.round(4)*100)
ac2= accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:", ac2.round(4)*100)




