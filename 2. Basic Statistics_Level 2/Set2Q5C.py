# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:00:57 2023

@author: Vinu
"""
import numpy as np
from scipy import stats
from scipy.stats import norm

D1 = (stats.norm.cdf(0,5,3))
D2 = (stats.norm.cdf(0,7,4))
print("The Probability of Division-1 making Loss is: ",D1.round(4)*100)
print("The Probability of Division-2 making Loss is: ",D2.round(4)*100)