# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:48:20 2023

@author: Vinu
"""

import pandas as pd
df=pd.read_csv("Q7.csv")
df
# Calculating the parameters Mean, Median, Mode, Variance, Standard Deviation, Range  

# ********* for Points column **********

PMean=df["Points"].mean()
PMedian=df["Points"].median()
PMode=df["Points"].mode()
PVariance=df["Points"].var()
PStd=df["Points"].std()
PRange=df["Points"].max()-df["Points"].min()

print("The Required Parameters of Points Column are:")
print("Mean of the points is: ",PMean)
print("Median of the points is: ",PMedian)
print("Mode of the points is: ",PMode)
print("Variance of the points is: ",PVariance)
print("Standard Deviation of the points is: ",PStd)
print("Range of the points is: ",PRange)

# ********* for Score column **********

SMean=df["Score"].mean()
SMedian=df["Score"].median()
SMode=df["Score"].mode()
SVariance=df["Score"].var()
SStd=df["Score"].std()
SRange=df["Score"].max()-df["Score"].min()

print("The Required Parameters of Score Column are:")
print("Mean of the Score is: ",SMean)
print("Median of the Score is: ",SMedian)
print("Mode of the Score is: ",SMode)
print("Variance of the Score is: ",SVariance)
print("Standard Deviation of the Score is: ",SStd)
print("Range of the Score is: ",SRange)

# ********* for Weigh column **********

WMean=df["Weigh"].mean()
WMedian=df["Weigh"].median()
WMode=df["Weigh"].mode()
WVariance=df["Weigh"].var()
WStd=df["Weigh"].std()
WRange=df["Weigh"].max()-df["Weigh"].min()

print("The Required Parameters of Weigh Column are: ")
print("Mean of the Weigh is: ",WMean)
print("Median of the Weigh is: ",WMedian)
print("Mode of the Weigh is: ",WMode)
print("Variance of the Weigh is: ",WVariance)
print("Standard Deviation of the Weigh is: ",WStd)
print("Range of the Weigh is: ",WRange)
