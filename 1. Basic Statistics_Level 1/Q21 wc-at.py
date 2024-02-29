# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:25:12 2023

@author: Vinu
"""

# Normality Test for Cars

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("wc-at.csv")
df.head()

from statsmodels.stats import weightstats as ztests
zcal,pval = ztests.ztest(x1=df["AT"],value=8,alternative='smaller')
alpha=0.05
if pval > alpha:
    print("In the Given Data, Adipose Tissue Follows Normal Distribution")
else:
    print("In the Given Data, Adipose Tissue Doesnot Follow Normal Distribution")
    