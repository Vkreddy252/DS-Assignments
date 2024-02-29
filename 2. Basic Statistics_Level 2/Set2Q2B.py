# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:39:33 2023

@author: Vinu
"""
# Z Score for 30
from scipy import stats
X= 1-stats.norm.cdf(30,loc=38,scale=6)
print("Probability of Employees age is 30 is: ",X.round(2))
