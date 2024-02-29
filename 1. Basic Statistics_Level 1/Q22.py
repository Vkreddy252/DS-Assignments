# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:16:28 2023

@author: Vinu
"""
from scipy import stats
from scipy.stats import norm 
z1=stats.norm.ppf(0.90)
z2=stats.norm.ppf(0.94)
z3=stats.norm.ppf(0.60)

print("The Z score value for the Confidence Interval at 90% is: ",z1.round(3))
print("The Z score value for the Confidence Interval at 94% is: ",z2.round(3))
print("The Z score value for the Confidence Interval at 60% is: ",z3.round(3))