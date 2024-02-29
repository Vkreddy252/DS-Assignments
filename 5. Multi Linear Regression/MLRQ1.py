# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:52:34 2023

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

df=pd.read_csv("50_Startups.csv")
df

# Step-2 *** EDA ***
df.info()
df1=df.rename({'R&D Spend':'RD','Administration':'AD','Marketing Spend':'MS'},axis=1)
df1
df1[df1.duplicated()]
df1.describe()

# Correlation 
cor = df1.corr()
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()

#pair plot
sns.set_style(style='darkgrid')
sns.pairplot(df1)

#Step-3 *** Model Fitting and Validation ***
model=smf.ols("Profit~RD+AD+MS",data=df1).fit()
model.params
model.tvalues , np.round(model.pvalues,5)
print("The accuracy of Model Before Improving is:", model.rsquared.round(4)*100)

sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()
list(np.where(model.resid<-20000))

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
plt.stem(np.arange(len(df1)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()
np.argmax(c) , np.max(c) #Influencer Value

# Influence Plot
influence_plot(model)
plt.show()

df1[df1.index.isin([49])] # Dropping the Influencer
df2=df1.drop(df1.index[[49]],axis=0).reset_index(drop=True)
df2

# Step-5 *** Creating Final model ***

model2=smf.ols("Profit~RD+AD+MS",data=df2).fit()
while model2.rsquared < 0.99:
    for c in [np.max(c)>1]:
        model2=smf.ols("Profit~RD+AD+MS",data=df2).fit()
        (c,_)=model2.get_influence().cooks_distance
        c
        np.argmax(c) , np.max(c)
        df2=df2.drop(df2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        df2
    else:
        final_model=smf.ols("Profit~RD+AD+MS",data=df2).fit()
        final_model.rsquared , final_model.aic
print("Model accuracy is Finally improved to",final_model.rsquared.round(4)*100)

# R**2 Table
d2={'Name':['Model','Final_Model'],'Rsquared':[model.rsquared.round(4)*100,final_model.rsquared.round(4)*100]}
table=pd.DataFrame(d2)
table