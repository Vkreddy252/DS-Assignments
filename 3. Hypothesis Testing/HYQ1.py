# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:42:08 2023

@author: Vinu
"""

import pandas as pd
df = pd.read_csv("Cutlets.csv")
df

df["Unit A"].mean()
df["Unit B"].mean()

from scipy import stats

zcal ,pval = stats.ttest_ind( df["Unit A"] , df["Unit B"] ) 

print("Z calculated value:", zcal.round(3))
print("P- value:", pval.round(3))
 
alpha = 0.05

if pval < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and Ho is accepted")