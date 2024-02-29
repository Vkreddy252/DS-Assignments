# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:00:57 2023

@author: Vinu
"""
import numpy as np
from scipy import stats
from scipy.stats import norm

Mean = 540
Std = 225
Z = stats.norm.ppf(0.05)
P = (Z * Std) + Mean
print("The 5th percentile profit of the company is {} Millions".format(P.round(2)))