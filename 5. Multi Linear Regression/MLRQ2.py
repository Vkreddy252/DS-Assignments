# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:50:46 2023

@author: Vinu
"""
# Step-1 *** Importing Libraries and File ***

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot

df=pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
df.head()

# Step -2 *** EDA ***

df1=df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
df1.shape
df1.describe().round(2)
df1.duplicated().sum()
df2=df1.drop_duplicates().reset_index(drop=True)
df2
df2.isnull().sum()
df2.info()

# Correlation 
cor = df2.corr()
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()

# Pair Plot
sns.set_style(style='darkgrid')
sns.pairplot(df2)

# Step-3 *** Model Fitting and Validation ***
model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df2).fit()
model.params
model.tvalues , np.round(model.pvalues,5)
print("The accuracy of Model Before Improving is:", model.rsquared.round(4)*100)

sm.qqplot(model.resid,line='q') 
plt.show()

list(np.where(model.resid>6000))
list(np.where(model.resid<-4000))

def standard_values(vals) : return (vals-vals.mean())/vals.std()
plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('Fitted values')
plt.ylabel('Residual values')
plt.show()

# Step-4 *** Removing Influencers from the Model ***

(c,_)=model.get_influence().cooks_distance
c

# Stem Plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(df2)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()

np.argmax(c) , np.max(c)
df2[df2.index.isin([80])] 
df2_copy=df2.copy()
df2_copy

df3=df2_copy.drop(df2_copy.index[[80]],axis=0).reset_index(drop=True)
df3

# Step-5 *** Creating Final model ***

while np.max(c)>0.5 :
    model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df3).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    df3=df3.drop(df3.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    df3
else:
    final_model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df3).fit()
    final_model.rsquared 
print("The Model Accuracy improved to",final_model.rsquared.round(4)*100)

# Y Predicted Values
y_pred=final_model.predict(df3)
y_pred.round(2)