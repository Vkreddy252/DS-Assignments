# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:17:56 2023

@author: Vinu
"""

# Step-1: Importing the Required Libraries and file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("forestfires.csv")
df.head()

#==============================================================================

# Step-2: EDA
df.duplicated().sum()
df1=df.drop_duplicates().reset_index(drop=True)
df1
df1.isnull().sum()
df1.info()
df1.describe().round(2)

# Correlation
cor = df1.corr()
plt.figure(figsize=(25,15))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()

# Box Plots and Histograms for Independent Variables

cols = ['FFMC','DMC', 'DC','ISI', 'temp','RH','wind','rain','area']

# Box Plots
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(3,3,i)
    sns.boxplot(y=col, data=df1,width=0.2)
    i+=1
plt.show()

# Histograms
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(3,3,i)
    sns.histplot(x=col, data=df1,kde=True,color='blue')
    i+=1
plt.show()

# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(df1)
plt.savefig('Pair Plot.png', bbox_inches='tight', dpi=100)

#==============================================================================

# Step-3: Data transformation

# Label Encoding For Y
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y =  LE.fit_transform(df1["size_category"])
Y1=pd.DataFrame(Y)
Y1

X = df1.iloc[:,2:30]
list(X)
    
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
X1 = pd.DataFrame(SS_X)
X1.columns =list(X)
X1.head()

#==============================================================================

# Step-4:  Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y1,train_size=0.75,random_state=50)

#==============================================================================

# Step-5:  SVM

# Finding Best Paramters Using Grid Search CV

# SVM Model Fitting **** Kernel = rbf ****
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
gs = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[1,4,8,12,16,20],'C':[12,14,0.1,0.01,10,20,17,0.001] }]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=0.1,gamma= 1,kernel='rbf')
svc.fit(X_train,Y_train)
Y_pred_trainrbf = svc.predict(X_train) 
Y_pred_testrbf = svc.predict(X_test) 
#------------------------------------------------------------------------------

# SVM Model Fitting **** Kernel = Linear ****
gs = SVC()
param_grid = [{'kernel':['linear'],'gamma':[1,4,8,12,16,20],'C':[12,14,0.1,0.01,10,20,17,0.001] }]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=12,gamma= 0.1,kernel='linear')
svc.fit(X_train,Y_train)
Y_pred_trainlin = svc.predict(X_train) 
Y_pred_testlin = svc.predict(X_test) 
#------------------------------------------------------------------------------

# SVM Model Fitting **** Kernel = Poly ****
gs = SVC()
param_grid = [{'kernel':['poly'],'gamma':[1,4,8,12,16,20],'C':[12,14,0.1,0.01,10,20,17,0.001] }]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=12,gamma= 1,kernel='poly')
svc.fit(X_train,Y_train)
Y_pred_trainpoly = svc.predict(X_train) 
Y_pred_testpoly = svc.predict(X_test) 
#------------------------------------------------------------------------------

# SVM Model Fitting **** Kernel = Sigmoid ****
gs = SVC()
param_grid = [{'kernel':['sigmoid'],'gamma':[1,4,8,12,16,20],'C':[12,14,0.1,0.01,10,20,17,0.001] }]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=0.01,gamma= 1,kernel='sigmoid')
svc.fit(X_train,Y_train)
Y_pred_trainsig = svc.predict(X_train) 
Y_pred_testsig = svc.predict(X_test) 

#==============================================================================
# Step -6 Accuracy Table

from sklearn.metrics import accuracy_score
ac1= accuracy_score(Y_train,Y_pred_trainrbf)
ac2= accuracy_score(Y_test,Y_pred_testrbf)

ac3= accuracy_score(Y_train,Y_pred_trainlin)
ac4= accuracy_score(Y_test,Y_pred_testlin)

ac5= accuracy_score(Y_train,Y_pred_trainpoly)
ac6= accuracy_score(Y_test,Y_pred_testpoly)

ac7= accuracy_score(Y_train,Y_pred_trainsig)
ac8= accuracy_score(Y_test,Y_pred_testsig)

print("Training Accuracy score by rbf:", ac1.round(4)*100)
print("Test Accuracy score by rbf:", ac2.round(4)*100)

print("Training Accuracy score by Linear:", ac3.round(4)*100)
print("Test Accuracy score by Linear:", ac4.round(4)*100)

print("Training Accuracy score by Polynomial:", ac5.round(4)*100)
print("Test Accuracy score by Polynomial:", ac6.round(4)*100)

print("Training Accuracy score by Sigmoid:", ac7.round(4)*100)
print("Test Accuracy score by Sigmoid:", ac8.round(4)*100)

# Accuracy Table

Table = {"MODEL":pd.Series(["RBF","Linear","Polynomial","Sigmoid"]),"Accuracy":pd.Series([ac2,ac4,ac6,ac8])}
Table1=pd.DataFrame(Table)
Table1.sort_values(['Accuracy'])

'''
      ***** Results *****
      
S.No        MODEL           Accuracy
1           RBF             0.742188
2           Sigmoid         0.742188
3           Polynomial      0.812500
4           Linear          0.976562

'''
