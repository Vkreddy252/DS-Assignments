# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:15:41 2023

@author: Vinu
"""

# Step-1: Importing Required Libraries and Files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')
df.shape

#==============================================================================
# Step-2: EDA
df['Quarters'] = 0
df['Year'] = 0

for i in range(42):
    p = df["Quarter"][i]
    df['Quarters'][i]= p[0:2]
    df['Year'][i]= p[3:5]

df.head()

Quarters_Dummies = pd.DataFrame(pd.get_dummies(df['Quarters']))
df1 = pd.concat([df,Quarters_Dummies],axis = 1)
df1.head()

plt.figure(figsize=(8,5))
plt.plot(df1['Sales'], color = 'blue', linewidth=3)

# Histogram
df1['Sales'].hist(figsize=(8,5),color='green')

# Density Plot
df1['Sales'].plot(kind = 'kde', figsize=(8,5),color='red')

# Box Plots for Quarters
sns.set(rc={'figure.figsize':(8,5)})
sns.boxplot(x="Quarters",y="Sales",data=df1)

# Box Plot for years
sns.boxplot(x="Year",y="Sales",data=df1)

# Lag Plot
from pandas.plotting import lag_plot
lag_plot(df1['Sales'])
plt.show()

# Line Plot
plt.figure(figsize=(8,5))
sns.lineplot(x="Year",y="Sales",data=df1)

# Autocorrelation and Partial Auto-Correlation Plot
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(df1.Sales,lags=12)
tsa_plots.plot_pacf(df1.Sales,lags=12)
plt.show()

#==============================================================================
# Step-3: Evaluation of Methods

from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

Train = df1.head(32)
Test = df1.tail(10)

plt.figure(figsize=(10,6))
df1.Sales.plot(label="org")
for i in range(2,8,2):
    df["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df1.Sales,period=12)
decompose_ts_add.plot()
plt.show()

def RMSE(org, pred):
    rmse=np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse

# Simple Exponential Method

ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
rmse_ses_model = RMSE(Test.Sales, pred_ses)
rmse_ses_model

# Holt Method
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
rmse_hw_model = RMSE(Test.Sales, pred_hw)
rmse_hw_model

# Holt Exponential Smoothing Additive
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
rmse_hwe_add_add_model = RMSE(Test.Sales, pred_hwe_add_add)
rmse_hwe_add_add_model

# Holt Exponential Smoothing Multiplicative
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
rmse_hwe_model_mul_add_model = RMSE(Test.Sales, pred_hwe_mul_add)
rmse_hwe_model_mul_add_model

df2 = df.copy()
df2.head()

df2["t"] = np.arange(1,43)
df2["t_squared"] = df2["t"]*df2["t"]
df2["log_sales"] = np.log(df2["Sales"])
df2.head()

Train = df2.head(32)
Test = df2.tail(10)

# Linear Model
import statsmodels.formula.api as smf 
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear_model = RMSE(Test['Sales'], pred_linear)
rmse_linear_model

# Exponential Model
Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp_model = RMSE(Test['Sales'], np.exp(pred_Exp))
rmse_Exp_model

# Quadratic Model
Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad_model = RMSE(Test['Sales'], pred_Quad)
rmse_Quad_model

list = [['Simple Exponential Method',rmse_ses_model], 
        ['Holt method',rmse_hw_model],
        ['HW exp smoothing add',rmse_hwe_add_add_model],
        ['HW exp smoothing mult',rmse_hwe_model_mul_add_model],
        ['Linear Mode',rmse_linear_model],['Exp model',rmse_Exp_model],
        ['Quad model',rmse_Quad_model]]

data = pd.DataFrame(list, columns =['Model', 'RMSE_Value']) 
data
#==============================================================================

#                *********** Results ********************

''' 

================================================
         Model                       RMSE_Value
------------------------------------------------          
Simple Exponential Method           1034.935927
Holt method                          786.766483
HW exp smoothing add                 610.227144
HW exp smoothing mult                569.054041
Linear Model                         752.923393
Exp model                            590.331643
Quad model                           457.735736
=================================================


'''