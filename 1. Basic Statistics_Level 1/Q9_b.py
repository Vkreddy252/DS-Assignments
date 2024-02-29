# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:04:51 2023

@author: Vinu
"""

import pandas as pd
df=pd.read_csv("Q9_b.csv")
df.head()

#### Skewness and Kurtosis of SP Column ####

x=df["SP"].skew()
y=df["SP"].kurt()

print("The Skewness of SP is: ",x.round(3))
print("The Kurtosis of SP is: ",y.round(3))

# ==================================================================

#### Skewness and Kurtosis of WT Column ####

a=df["WT"].skew()
b=df["WT"].kurt()

print("The Skewness of WT is: ",a.round(3))
print("The Kurtosis of WT is: ",b.round(3))



