# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:16:28 2023

@author: Vinu
"""
# Calculating t score values for 95%, 96% and 99%
from scipy import stats
from scipy.stats import norm
# df=n-1 =>24
 
t1=stats.t.ppf(0.975,24) 
t2=stats.t.ppf(0.98,24)
t3=stats.t.ppf(0.995,24)

print("t scores of 95% confidence interval for sample size of 25 : ",t1.round(3))
print("t scores of 96% confidence interval for sample size of 25 : ",t2.round(3))
print("t scores of 99% confidence interval for sample size of 25 : ",t3.round(3))