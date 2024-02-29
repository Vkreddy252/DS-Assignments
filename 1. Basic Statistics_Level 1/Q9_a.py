# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:04:51 2023

@author: Vinu
"""

import pandas as pd
df=pd.read_csv("Q9_a.csv")
df.head()

#### Skewness and Kurtosis of Speed Column ####

x=df["speed"].skew()
y=df["speed"].kurt()

print("The Skewness of Speed is: ",x.round(3))
print("The Kurtosis of Speed is: ",y.round(3))

# ==================================================================

#### Skewness and Kurtosis of Distance Column ####

a=df["dist"].skew()
b=df["dist"].kurt()

print("The Skewness of distance is: ",a.round(3))
print("The Kurtosis of distance is: ",b.round(3))



