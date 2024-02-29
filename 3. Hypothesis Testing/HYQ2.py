# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 16:25:56 2023

@author: Vinu
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

df=pd.read_csv("LabTAT.csv")
df.head()

test_statistic , p_value = stats.f_oneway(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3])
print('p_value =',p_value)

alpha = 0.05 
if p_value < alpha:
    print('Ho is rejected and H1 is accepted')
else:
    print('H1 is rejected and H0 is accepted')