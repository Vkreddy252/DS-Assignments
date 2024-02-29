# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 08:37:51 2023

@author: Vinu
"""
import scipy.stats as stats
# Z Score for 0.005
z1=stats.norm.ppf(0.005)
# Z score for 0.995
z2=stats.norm.ppf(0.995)
print('The Z Scores Z1 and Z2 are: ',z1.round(3),z2.round(3))