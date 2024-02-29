# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:39:33 2023

@author: Vinu
"""
# Z Score for 44
from scipy import stats
X= 1-stats.norm.cdf(44,loc=38,scale=6)
print("Probability of Employees age Older than 44 is: ",X.round(4))

# Z Score between 38 and 44

X1=1-stats.norm.cdf(44,loc=38,scale=6)
X2=1-stats.norm.cdf(38,loc=38,scale=6)
X3=X2-X1
print("Probability of Employees age in between 38 and 44 is:",X3.round(4))