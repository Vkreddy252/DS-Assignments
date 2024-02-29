# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:48:25 2023

@author: Vinu
"""

#step-1: Importing the Required Libraries and file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bank-full.csv")
df.head()

#==============================================================================
#step-2: EDA
df.info()

# Correlation
cor = df.corr()
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()
# Histograms
# age
df['age'].hist()
plt.ylabel('Count')
plt.xlabel('age')
plt.show()

# balance
df['balance'].hist()
plt.ylabel('Count')
plt.xlabel('balance')
plt.show()

# day
df['day'].hist()
plt.ylabel('Count')
plt.xlabel('day')
plt.show()

# duration
df['duration'].hist()
plt.ylabel('Count')
plt.xlabel('duration')
plt.show()

# campaign
df['campaign'].hist()
plt.ylabel('Count')
plt.xlabel('campaign')
plt.show()

# Pdays
df['pdays'].hist()
plt.ylabel('Count')
plt.xlabel('pdays')
plt.show()

# previous
df['previous'].hist()
plt.ylabel('Count')
plt.xlabel('previous')
plt.show()

# Box Plot
fig, axes=plt.subplots(7,1,figsize=(14,12),sharex=False,sharey=False)
sns.boxplot(x='age',data=df,color='blue',ax=axes[0])
sns.boxplot(x='balance',data=df,color='green',ax=axes[1])
sns.boxplot(x='day',data=df,color='orange',ax=axes[2])
sns.boxplot(x='duration',data=df,color='yellow',ax=axes[3])
sns.boxplot(x='campaign',data=df,color='brown',ax=axes[4])
sns.boxplot(x='pdays',data=df,color='black',ax=axes[5])
sns.boxplot(x='previous',data=df,color='red',ax=axes[6])
plt.tight_layout(pad=2.0)
plt.savefig('Box Plots.png', bbox_inches='tight', dpi=100)

#pair plot
sns.set_style(style='darkgrid')
sns.pairplot(df)
plt.savefig('Pair Plot.png', bbox_inches='tight', dpi=100)

# Seperating Categorical and Numerical Data
df_cat = df.select_dtypes(object)
df_num = df.select_dtypes(int)
df_cat.info()

# Label Encoding for Categorical Values in X
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,9):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])
    
df.head()    
X = df_cat.iloc[:,0:9]
df1 = pd.concat([df_num,X],axis=1)
df1.info()

# Label Encoding For Y
Y =  LE.fit_transform(df["y"])
Y2=pd.DataFrame(Y)
Y2

#==============================================================================
# step3: Data transformation
X1 = df1.iloc[:,0:16]
list(X1)
    
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X1)
X2 = pd.DataFrame(SS_X)
X2.columns =list(X1)
X2.head()

#==============================================================================
# step4:  Model fitting
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X2,Y2) 
Y_pred = logreg.predict(X2) 

#==============================================================================
# step5: # Metrics & Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y2,Y_pred)
cm
ac= accuracy_score(Y2,Y_pred)
print("Accuracy score:", ac.round(3)*100)

from sklearn.metrics import recall_score,precision_score,f1_score
print("Sensitivity score:", recall_score(Y2,Y_pred).round(3))
print("Precision score:", precision_score(Y2,Y_pred).round(3))
print("F1 score:", f1_score(Y2,Y_pred).round(3))

TN = cm[0,0]
FP = cm[1,0]
TNR = TN/(TN + FP)
print("Specificity:", TNR.round(3))

#==============================================================================
# Step6: ROC Curve

from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds=roc_curve(Y2,logreg.predict_proba(X2)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y2,Y_pred)
plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()



