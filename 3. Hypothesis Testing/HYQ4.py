# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 19:05:26 2023

@author: Vinu
"""
import pandas as pd
df=pd.read_csv('Costomer+OrderForm.csv')
df.head()

df.isnull().sum()
df.info()
x1=df['Phillippines'].value_counts()
x2=df['Indonesia'].value_counts()
x3=df['Malta'].value_counts()
x4=df['India'].value_counts()

print(x1,'\n',x2,'\n',x3,'\n',x4)

from scipy.stats import chi2_contingency
 
data = [[271,267,269,280], [29,33,31,20]]
stat, pval, dof, expected = chi2_contingency(data)
 
alpha = 0.05
print("p value is ",pval.round(4))

if pval <= alpha:
    print('Ho is rejected and H1 is accepted')
else:
    print('H1 is rejected and H0 is accepted')
