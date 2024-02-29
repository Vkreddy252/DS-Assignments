# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:43:17 2023

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
# Step-3: Data transformation and Data Splitting

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
# Step-4:  Model Fitting:  NaiveBayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred_train = gnb.predict(X_train) 
Y_pred_test = gnb.predict(X_test) 

#==============================================================================
# Step-5: # Metrics & Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred_test)
cm
ac= accuracy_score(Y_test,Y_pred_test)
print("Accuracy score:", ac.round(4)*100)

from sklearn.metrics import recall_score,precision_score,f1_score
print("Sensitivity score:", recall_score(Y_test,Y_pred_test).round(3))
print("Precision score:", precision_score(Y_test,Y_pred_test).round(3))
print("F1 score:", f1_score(Y_test,Y_pred_test).round(3))

TN = cm[0,0]
FP = cm[1,0]
TNR = TN/(TN + FP)
print("Specificity:", TNR.round(3))

#==============================================================================
# Step-6: ROC Curve

from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds=roc_curve(Y_test,gnb.predict_proba(X_test)[:, 1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y_test,Y_pred_test)
plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

