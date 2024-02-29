# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:32:55 2023

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

df = pd.read_csv("Fraud_check.csv")
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
plt.figure(figsize=(6,4))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()
    
# Box Plots and Histograms for Independent Variables

cols = ['Taxable.Income','City.Population', 'Work.Experience']

# Box Plots

i=1
for col in cols:
    sns.boxplot(y=col, data=df,width=0.2)
    i+=1
    plt.figure(figsize=(6,8))
    plt.show()

# Histograms

i=1
for col in cols:
    sns.histplot(x=col, data=df,kde=True,color='blue')
    i+=1
    plt.figure(figsize=(10,15))
    plt.show()

#==============================================================================
# Step-3: Data Transformation
# Coverting Sales into Categorical Variable

df1=df.copy()
df1["TaxInc"] = pd.cut(df1["Taxable.Income"], bins = [10000,30000,100000], labels = ["Risky", "Good"])
df1.head()
df1.info()

# Pair plot
sns.pairplot(data=df1, hue = 'TaxInc')

# Label Encoding For Categorical Columns

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df1["Undergrad"] = LE.fit_transform(df1["Undergrad"])
df1["Marital.Status"] = LE.fit_transform(df1["Marital.Status"])
df1["Urban"] = LE.fit_transform(df1["Urban"])
df1["TaxInc"] = LE.fit_transform(df1["TaxInc"])
X = df1.iloc[:,0:6]

# Standardization for X 

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
X1 = pd.DataFrame(SS_X)
X1.columns =list(X)
X1.head()

Y = df1["TaxInc"]

#==============================================================================
# Step-4:  Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y,test_size=0.2)

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
RF = RandomForestClassifier(n_estimators=100,n_jobs=1,criterion="gini",)
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

