# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:03:09 2023

@author: Vinu
"""
import pandas as pd
df=pd.read_csv("Cars.csv")
df
m=df["MPG"].mean()
s=df["MPG"].std()
from scipy.stats import norm
nd  = norm(m,s)  # mean, sd
# p(X > 38)
p1=1 - nd.cdf(38)

# p(X < 40)
p2=nd.cdf(40)

# p(2 < X < 50)
p3=nd.cdf(50) - nd.cdf(2)

#Required Probabilities

print("Probability for MPG>38 is: ",p1.round(2)*100)
print("Probability for MPG<40 is: ",p2.round(2)*100)
print("Probability for 2<MPG<50 is: ",p3.round(2)*100)