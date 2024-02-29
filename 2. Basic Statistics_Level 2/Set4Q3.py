# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:07:30 2023

@author: Vinu
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
n=100
M=50 
S=40 
X1=45
X2=55
# z-scores at x=45
z_45=(X1-M)/(S/n**0.5)
z_45
# find z-scores at x=55
z_55=(X2-M)/(S/n**0.5)
z_55
# For No investigation P(45<X<55) using z_scores = P(X<50)-P(X<45)
NI = stats.norm.cdf(z_55)-stats.norm.cdf(z_45)
# For Investigation 1-P(45<X<55)
I = 1-NI
print("Required Probability is: ",I.round(4)*100,'%')