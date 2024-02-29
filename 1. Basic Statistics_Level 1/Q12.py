# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:09:18 2023

@author: Vinu
"""

import numpy as np
x=np.array([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])
import pandas as pd
y = pd.DataFrame(x)
y
# Calculatig the parameters
me=y.mean()
md=y.median()
va=y.var()
st=y.std()

print("The Required Parameters are:")
print("Mean = ",me)
print("Median = : ",md)
print("Variance = ",va.round(2))
print("Standard Deviation = ",st.round(2))

import matplotlib.pyplot as plt
plt.boxplot(y)