# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:12:44 2023

@author: Vinu
"""
# Step-1: Importing Required Libraries and Files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('Airlines+Data.xlsx')
df.shape

#==============================================================================
# Step-2: EDA
df.info()
df.describe()
# Making Month Column as Index
df.set_index('Month',inplace=True)
df.head()
df.isnull().sum()

df.plot()
plt.show()

# Histogram
df.hist()
plt.show()

# Density Plot
df.plot(kind='kde')
plt.show()

# Lag Plot
from pandas.plotting import lag_plot
lag_plot(df)
plt.show()

# Autocorrelation Plot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df,lags=30)
plt.show()

#==============================================================================
# Step-3: Evaluation of Methods

from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

Train = df.head(84)
Test = df.tail(12)

plt.figure(figsize=(10,6))
df.Passengers.plot(label="org")
for i in range(2,8,2):
    df["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df.Passengers,period=12)
decompose_ts_add.plot()
plt.show()

def RMSE(org, pred):
    rmse=np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse

# Simple Exponential Method

ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
rmse_ses_model = RMSE(Test.Passengers, pred_ses)
rmse_ses_model

# Holt Method
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
rmse_hw_model = RMSE(Test.Passengers, pred_hw)
rmse_hw_model

# Holt Exponential Smoothing Additive
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
rmse_hwe_add_add_model = RMSE(Test.Passengers, pred_hwe_add_add)
rmse_hwe_add_add_model

# Holt Exponential Smoothing Multiplicative
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=4).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
rmse_hwe_model_mul_add_model = RMSE(Test.Passengers, pred_hwe_mul_add)
rmse_hwe_model_mul_add_model

df1 = df.copy()
df1.head()

df1["t"] = np.arange(1,97)
df1["t_squared"] = df1["t"]*df1["t"]
df1["log_psngr"] = np.log(df1["Passengers"])
df1.head()

Train = df1.head(84)
Test = df1.tail(12)

# Linear Model
import statsmodels.formula.api as smf 
linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear_model = RMSE(Test['Passengers'], pred_linear)
rmse_linear_model

# Exponential Model
Exp = smf.ols('log_psngr~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp_model = RMSE(Test['Passengers'], np.exp(pred_Exp))
rmse_Exp_model

# Quadratic Model
Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad_model = RMSE(Test['Passengers'], pred_Quad)
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

#               *********** Results ********************

''' 

===============================================
         Model                      RMSE_Value
-----------------------------------------------         
Simple Exponential Method            68.006740
Holt method                          58.573847
HW exp smoothing add                 62.712082
HW exp smoothing mult                64.663738
Linear Model                         53.199237
Exp model                            46.057361
Quad model                           48.051889
================================================

'''