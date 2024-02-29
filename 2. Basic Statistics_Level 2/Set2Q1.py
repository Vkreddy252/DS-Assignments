# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:39:33 2023

@author: Vinu
"""
from scipy import stats
X= 1-stats.norm.cdf(60,loc=55,scale=8)
print("Required Probability is: ",X.round(4))
