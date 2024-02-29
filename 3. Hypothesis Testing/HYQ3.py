# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 18:54:23 2023

@author: Vinu
"""
import pandas as pd
df=pd.read_csv('BuyerRatio.csv')
df

from scipy.stats import chi2_contingency
 
data = [[50,142,131,70], [435, 1523, 1356,750]]
stat, pval, dof, expected = chi2_contingency(data)
 
alpha = 0.05
print("p value is ",pval.round(4))

if pval < alpha:
    print('Ho is rejected and H1 is accepted')
else:
    print('H1 is rejected and H0 is accepted')