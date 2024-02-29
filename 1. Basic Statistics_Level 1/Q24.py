# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:04:39 2023

@author: Vinu
"""

from scipy import stats
from scipy.stats import norm
# Calculating t value
t=(260-270)/(90/18**0.5)
p=1-stats.t.cdf(abs(-0.4714),df=17)
print("The Required probability is: ",p)