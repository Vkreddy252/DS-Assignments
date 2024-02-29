# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:25:41 2023

@author: Vinu
"""
#==============================================================================
# Step-1:  Importing the Required Libraries and file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("EastWestAirlines.xlsx",sheet_name="data")
df.head()
df.shape 
#==============================================================================
# Step-2:  EDA  

# Making ID as Index Column

df.set_index('ID#',inplace=True)
df
df.head()
df.duplicated().sum()
df1=df.drop_duplicates().reset_index(drop=True)
df1.duplicated().sum()
df1.isnull().sum()
 
# Making Award and cc columns as categorical
df1['Award?'] = df1['Award?'].astype('category')
df1['cc1_miles'] = df1['cc1_miles'].astype('category')
df1['cc2_miles'] = df1['cc2_miles'].astype('category')
df1['cc3_miles'] = df1['cc3_miles'].astype('category')
df1.dtypes
df1.describe().round(2) 
df1.info()

# Count Plots

# cc1_miles
df1['cc1_miles'].value_counts().plot.bar()
plt.xlabel('cc1_miles')
print(df1['cc1_miles'].value_counts())

#cc2_miles
df1['cc2_miles'].value_counts().plot.bar()
plt.xlabel('cc2_miles')
print(df1['cc2_miles'].value_counts())

# cc2_miles
df1['cc3_miles'].value_counts().plot.bar()
plt.xlabel('cc3_miles')
print(df1['cc3_miles'].value_counts())

# Award
df1['Award?'].value_counts().plot.bar()
plt.xlabel('Award')
print(df1['Award?'].value_counts())

cols = ["Balance","Qual_miles","Bonus_miles","Bonus_trans",
        "Flight_miles_12mo","Flight_trans_12","Days_since_enroll"]

# Box Plots

fig, axes=plt.subplots(7,1,figsize=(12,8),sharex=False,sharey=False)
sns.boxplot(x='Balance',data=df1,ax=axes[0])
sns.boxplot(x='Qual_miles',data=df1,ax=axes[1])
sns.boxplot(x='Bonus_miles',data=df1,ax=axes[2])
sns.boxplot(x='Bonus_trans',data=df1,ax=axes[3])
sns.boxplot(x='Flight_miles_12mo',data=df1,ax=axes[4])
sns.boxplot(x='Flight_trans_12',data=df1,ax=axes[5])
sns.boxplot(x='Days_since_enroll',data=df1,ax=axes[6])
plt.tight_layout(pad=2.0)


#pair plot
sns.pairplot(df1)

# Correlation   
cor = df1.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()

#==============================================================================
# Step-3: Data Transformation

# Standardization

from sklearn.preprocessing import StandardScaler, MinMaxScaler
standard_scaler = StandardScaler()
std_df = standard_scaler.fit_transform(df1)
std_df.shape

minmax = MinMaxScaler()
minmax_df = minmax.fit_transform(df1)
minmax_df.shape

#==============================================================================
# Step-4: Clustering Algorithms

# 1. Hiearchial Clustering

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# **** Method - single ****

# Dendogram
plt.figure(figsize=(20,8))
dendrogram = sch.dendrogram(sch.linkage(minmax_df, method='single'))
plt.title("Method - Single")
plt.show()

# Creating Clusters
hc_s = AgglomerativeClustering(n_clusters=6, affinity = 'euclidean', linkage= 'single')     
y_hc = hc_s.fit_predict(minmax_df)
Clusters_s = pd.DataFrame(y_hc, columns=['Cluster'])    
Clusters_s     
Clusters_s.value_counts()
df1['h_clusterid'] = Clusters_s     
df1.groupby('h_clusterid').agg(['mean']).reset_index()

# **** Method - Average ****

# Dendogram
plt.figure(figsize=(20,8))
dendrogram = sch.dendrogram(sch.linkage(minmax_df, method='average'))
plt.title("Method - Average")
plt.show()

# Creating Clusters
hc_a = AgglomerativeClustering(n_clusters=7, affinity = 'euclidean', linkage= 'average')   
y_hc = hc_a.fit_predict(minmax_df)
Clusters_a = pd.DataFrame(y_hc, columns=['Clusters'])     
Clusters_a   
df2=df1.copy() 
df2['h_clusterid'] = Clusters_a
Clusters_a.value_counts()
df2.groupby('h_clusterid').agg(['mean']).reset_index()

# **** Method - Complete *****

# Dendogram
plt.figure(figsize=(20,8))
dendrogram = sch.dendrogram(sch.linkage(minmax_df, method='complete'))
plt.title("Method - Complete")
plt.show()

# Creating Clusters
hc_c = AgglomerativeClustering(n_clusters=8, affinity = 'euclidean', linkage= 'complete')
y_hc = hc_c.fit_predict(minmax_df)
Clusters_c = pd.DataFrame(y_hc, columns=['Clusters'])
Clusters_c    
df3=df1.copy()
df3['h_clusterid'] = Clusters_c
Clusters_c.value_counts()  
df3.groupby('h_clusterid').agg(['mean']).reset_index()

#------------------------------------------------------------------------------

# 2.K-Means Clustering

# Finding optimum number of clusters using Elbow Method

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, random_state = 0)
  kmeans.fit(std_df)
  wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

clusters_new = KMeans(8, random_state=32)
clusters_new.fit(std_df)
clusters_new.labels_
df4=df1.copy()
df4['h_clusterid'] = clusters_new.labels_
df4
df4.groupby('h_clusterid').agg(['mean']).reset_index()

#------------------------------------------------------------------------------
# 3.DBSCAN Clustering

from sklearn.cluster import DBSCAN
df_d=pd.read_excel("EastWestAirlines.xlsx",sheet_name="data")
dbscan = DBSCAN(eps = 1, min_samples=8)
dbscan.fit(std_df)

# Labelling Noise Points with -1
dbscan.labels_
cl = pd.DataFrame(dbscan.labels_, columns=['clusters'])
cl
df_d=pd.concat([df1,cl], axis=1)
df_d.head()

df_d['clusters'].value_counts()
df_d.groupby('clusters').agg(['mean']).reset_index()

# ============================================================================= 



