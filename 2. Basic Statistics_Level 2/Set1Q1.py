# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:18:44 2023

@author: Vinu
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
Data=pd.Series([24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00])
Names=["Allied Signal","Bankers Trust","General Mills","ITT Industries","J.P.Morgan & Co.","Lehman Brothers","Marriott","MCI","Merrill Lynch","Microsoft","Morgan Stanley","Sun Microsystems","Travelers","US Airways","Warner-Lambert"]
fig=plt.figure(figsize=(8,8))
plt.pie(Data,labels=Names,autopct='%1.2f%%')
plt.show()

sns.boxplot(Data,orient='h',color='yellow')
Mean = round(Data.mean(),3)
Std = round(Data.std(),3)
Var = round(Data.var(),3)

print("The Mean is: ",Mean)
print("The Standard Deviation is: ",Std)
print("The Variance is: ",Var)