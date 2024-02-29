# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:43:17 2023

@author: Vinu
"""

# Step-1: Importing the Required Libraries and file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("SalaryData_Train.csv")
df_train.head()

df_test = pd.read_csv("SalaryData_Test.csv")
df_test.head()

#==============================================================================

# Step-2: EDA

# Train Data
df_train.duplicated().sum()
df_train1=df_train.drop_duplicates().reset_index(drop=True)
df_train1
df_train1.isnull().sum()
df_train1.info()
df_train1.describe().round(2)

# Test Data
df_test.duplicated().sum()
df_test1=df_test.drop_duplicates().reset_index(drop=True)
df_test1
df_test1.isnull().sum()
df_test1.info()
df_test1.describe().round(2)

#------------------------------------------------------------------------------

# Correlation

# Train Data
cor = df_train1.corr()
plt.figure(figsize=(10,6))
sns.heatmap(cor,annot=True,cmap='Blues')
plt.show()

# Test Data
cor = df_test1.corr()
plt.figure(figsize=(10,6))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()

#------------------------------------------------------------------------------

# Box Plots and Histograms for Independent Variables

cols = ['age','educationno', 'capitalgain','capitalloss', 'hoursperweek']

# Box Plots

# Train Data
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(2,3,i)
    sns.boxplot(y=col, data=df_train1,width=0.2)
    i+=1
plt.show()

# Test Data
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(2,3,i)
    sns.boxplot(y=col, data=df_test1,width=0.2,color='green')
    i+=1
plt.show()

#------------------------------------------------------------------------------

# Histograms

# Train Data
plt.figure(figsize=(15,8))
i=1
for col in cols:
    plt.subplot(2,3,i)
    sns.histplot(x=col, data=df_train1,kde=True,color='blue')
    i+=1
plt.show()

# Test Data
plt.figure(figsize=(15,8))
i=1
for col in cols:
    plt.subplot(2,3,i)
    sns.histplot(x=col, data=df_test1,kde=True,color='green')
    i+=1
plt.show()

#------------------------------------------------------------------------------

# Pie Plot

# Train Data
label_data=df_train1['Salary'].value_counts()

explode=(0.1,0.1)
plt.figure(figsize=(12,7))
patches, texts, pcts= plt.pie(label_data,labels=label_data.index,colors=['green','orange'],pctdistance=0.65,shadow=True,
                             startangle=90,explode=explode,autopct='%1.1f%%',
                             textprops={'fontsize':17,'color':'black','weight':'bold','family':'serif'})
plt.setp(pcts,color='white')
hfont={'weight':'bold','family':'serif'}
plt.title('Salary Comparision of Train Data',size=20,**hfont)

centre_circle=plt.Circle((0,0),0.40,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre_circle)
plt.legend(['Less Salary','More Salary'],loc="upper right")
plt.show()

# Test Data
label_data=df_test1['Salary'].value_counts()

explode=(0.1,0.1)
plt.figure(figsize=(12,7))
patches, texts, pcts= plt.pie(label_data,labels=label_data.index,colors=['blue','red'],pctdistance=0.65,shadow=True,
                             startangle=90,explode=explode,autopct='%1.1f%%',
                             textprops={'fontsize':17,'color':'black','weight':'bold','family':'serif'})
plt.setp(pcts,color='white')
hfont={'weight':'bold','family':'serif'}
plt.title('Salary Comparision of Test Data',size=20,**hfont)

centre_circle=plt.Circle((0,0),0.40,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre_circle)
plt.legend(['Less Salary','More Salary'],loc="upper right")
plt.show()

#------------------------------------------------------------------------------

# Cross Tab for Salary Comparision for Occupation Wise

# Train Data
pd.crosstab(df_train1['Salary'], df_train1['occupation']).plot(kind="bar", figsize=(15, 8));
plt.title('Occupation wise Salary Comparision of Train Data',)
plt.plot()

# Test Data
pd.crosstab(df_test1['Salary'], df_test1['occupation']).plot(kind="bar", figsize=(15, 8));
plt.title('Occupation wise Salary Comparision of Test Data',)
plt.plot()

#------------------------------------------------------------------------------
# Pair Plots

# Train Data
sns.pairplot(df_train1,hue='Salary',
             kind='reg',diag_kind='kde')
plt.show()

# Test Data
sns.pairplot(df_test1,hue='Salary',
             kind='reg',diag_kind='kde',palette='rocket')
plt.show()

#==============================================================================
# Step-3: Data transformation

# Train Data

# Seperating Categorical and Numerical Data
df_cat_train = df_train1.select_dtypes(object)
df_num_train = df_train1.select_dtypes(int)
df_cat_train.info()

# Label Encoding for Categorical Values in X
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,9):
    df_cat_train.iloc[:,i] = LE.fit_transform(df_cat_train.iloc[:,i])
df_train2 = pd.concat([df_num_train,df_cat_train],axis=1)    
df_train2.head()
df_train2.info()  
  
Xtr = df_train2.iloc[:,0:13]
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(Xtr)
X_train = pd.DataFrame(SS_X)
X_train.columns =list(Xtr)
X_train.head()

Y_train = df_train2.iloc[:,13]

# Test Data

# Seperating Categorical and Numerical Data
df_cat_test = df_test1.select_dtypes(object)
df_num_test = df_test1.select_dtypes(int)
df_cat_test.info()

# Label Encoding for Categorical Values in X
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,9):
    df_cat_test.iloc[:,i] = LE.fit_transform(df_cat_test.iloc[:,i])
df_test2 = pd.concat([df_num_test,df_cat_test],axis=1)    
df_test2.head()
df_test2.info()  
  
Xts = df_test2.iloc[:,0:13]
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(Xts)
X_test = pd.DataFrame(SS_X)
X_test.columns =list(Xts)
X_test.head()

Y_test = df_test2.iloc[:,13]
#==============================================================================
'''
X_train = df

# Step-4:  Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y,train_size=0.75,random_state=50)

#==============================================================================

# Step-5:  SVM

# Finding Best Paramters Using Grid Search CV

# SVM Model Fitting **** Kernel = rbf ****
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
gs = SVC()
param_grid = [{'kernel' : ["rbf"],'random_state':[40],'gamma':[0.1],'C':[1]}]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=1,gamma= 0.1,kernel='rbf',random_state=40)
svc.fit(X_train,Y_train)
Y_pred_trainrbf = svc.predict(X_train) 
Y_pred_testrbf = svc.predict(X_test) 
#------------------------------------------------------------------------------

# SVM Model Fitting **** Kernel = Linear ****
gs = SVC()
param_grid = [{'kernel' : ["linear"],'random_state':[40],'gamma':[0.1],'C':[1]}]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=1,gamma= 0.1,kernel='linear',random_state=40)
svc.fit(X_train,Y_train)
Y_pred_trainlin = svc.predict(X_train) 
Y_pred_testlin = svc.predict(X_test) 
#------------------------------------------------------------------------------

# SVM Model Fitting **** Kernel = Poly ****
gs = SVC()
param_grid = [{'kernel' : ["poly"],'random_state':[40],'gamma':[0.1],'C':[1]}]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=1,gamma= 0.1,kernel='poly',random_state=40)
svc.fit(X_train,Y_train)
Y_pred_trainpoly = svc.predict(X_train) 
Y_pred_testpoly = svc.predict(X_test) 
#------------------------------------------------------------------------------

# SVM Model Fitting **** Kernel = Sigmoid ****
gs = SVC()
param_grid = [{'kernel' : ["sigmoid"],'random_state':[40],'gamma':[0.1],'C':[1]}]
gsv = GridSearchCV(gs,param_grid,cv=5)
gsv.fit(X_train,Y_train)
gsv.best_params_ , gsv.best_score_ 

svc = SVC(C=1,gamma= 0.1,kernel='sigmoid',random_state=40)
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

         ***** Results *****
         
S.NO        MODEL               Accuracy
1           Sigmoid             0.744425
2           Linear              0.802557
3           Polynomial          0.832144
4           RBF                 0.835861
'''