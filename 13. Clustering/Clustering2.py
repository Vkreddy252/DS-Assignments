# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:25:41 2023

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

df = pd.read_csv("crime_data.csv")
df.head()
df.shape 
#==============================================================================
# Step-2:  EDA  
df.rename({'Unnamed: 0':'States'}, axis=1, inplace=True)
df.head()
df.duplicated().sum()
df.isnull().sum()
df.describe().round(2) 
df.info()

# Correlation   
cor = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()
    
# Box Plots and Histograms for Independent Variables

cols = ["Murder","Assault","UrbanPop","Rape"]

# Box Plots

plt.figure(figsize=(6,8))
i=1
for col in cols:
    plt.subplot(2,2,i)
    sns.boxplot(y=col, data=df,width=0.2)
    i+=1
plt.show()

# Histograms
plt.figure(figsize=(6,8))
i=1
for col in cols:
    plt.subplot(2,2,i)
    sns.histplot(x=col, data=df,kde=True,color='blue')
    i+=1
plt.show()

# Bar Plots 
# State Wise Murder Rate

plt.figure(figsize=(15,7))
plt.yticks(rotation = 0, fontsize = 12)
plt.xticks(rotation = 90, fontsize = 12)
sns.barplot(x=df.States, y = df.Murder, order=df.sort_values('Murder').States)
plt.xlabel('State', size=12)
plt.ylabel('Murder Rate', size=12)
plt.title('Murder Rate State wise', size=14)
plt.show()

# State Wise Assault Rate

plt.figure(figsize=(15,7))
plt.yticks(rotation = 0, fontsize = 12)
plt.xticks(rotation = 90, fontsize = 12)
sns.barplot(x=df.States, y = df.Assault, order=df.sort_values('Assault').States)
plt.xlabel('State', size=12)
plt.ylabel('Assault Rate', size=12)
plt.title('Assault Rate State wise', size=14)
plt.show()

# State Wise Urban Population

plt.figure(figsize=(15,7))
plt.yticks(rotation = 0, fontsize = 12)
plt.xticks(rotation = 90, fontsize = 12)
sns.barplot(x=df.States, y = df.UrbanPop, order=df.sort_values('UrbanPop').States)
plt.xlabel('State', size=12)
plt.ylabel('Urban Population', size=12)
plt.title('Urban Population State wise', size=14)
plt.show()

# State Wise Rape Rate

plt.figure(figsize=(15,7))
plt.yticks(rotation = 0, fontsize = 12)
plt.xticks(rotation = 90, fontsize = 12)
sns.barplot(x=df.States, y = df.Rape, order=df.sort_values('Rape').States)
plt.xlabel('State', size=12)
plt.ylabel('Rape Rate', size=12)
plt.title('State wise Rape Rate', size=14)
plt.show()

#pair plot
sns.pairplot(df)

#==============================================================================
# Step-3: Data Transformation

# Standardization

from sklearn.preprocessing import StandardScaler, MinMaxScaler
df1= df.drop(['States'], axis=1)
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
hc_s = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage= 'single')     
y_hc = hc_s.fit_predict(minmax_df)
Clusters_s = pd.DataFrame(y_hc, columns=['Cluster'])    
Clusters_s     
Clusters_s.value_counts()

df['h_clusterid'] = Clusters_s     
df.groupby('h_clusterid').agg(['mean']).reset_index()

# **** Method - Average ****

# Dendogram
plt.figure(figsize=(20,8))
dendrogram = sch.dendrogram(sch.linkage(minmax_df, method='average'))
plt.title("Method - Average")
plt.show()

# Creating Clusters
hc_a = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage= 'average')   
y_hc = hc_a.fit_predict(minmax_df)
Clusters_a = pd.DataFrame(y_hc, columns=['Clusters'])     
Clusters_a    
df['h_clusterid'] = Clusters_a
Clusters_a.value_counts()

df
df.groupby('h_clusterid').agg(['mean']).reset_index()

# **** Method - Complete *****

# Dendogram
plt.figure(figsize=(20,8))
dendrogram = sch.dendrogram(sch.linkage(minmax_df, method='complete'))
plt.title("Method - Complete")
plt.show()

# Creating Clusters
hc_c = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage= 'complete')
y_hc = hc_c.fit_predict(minmax_df)
Clusters_c = pd.DataFrame(y_hc, columns=['Clusters'])
Clusters_c    
df['h_clusterid'] = Clusters_c
Clusters_c.value_counts()

df    
df.groupby('h_clusterid').agg(['mean']).reset_index()

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

clusters_new = KMeans(4, random_state=32)
clusters_new.fit(std_df)
clusters_new.labels_
df1['h_clusterid'] = clusters_new.labels_
df1
df1.groupby('h_clusterid').agg(['mean']).reset_index()

#------------------------------------------------------------------------------
# 3.DBSCAN Clustering

from sklearn.cluster import DBSCAN
df_d=pd.read_csv("crime_data.csv")
dbscan = DBSCAN(eps = 1.25, min_samples=3)
dbscan.fit(std_df)

# Labelling Noise Points with -1
dbscan.labels_
cl = pd.DataFrame(dbscan.labels_, columns=['clusters'])
cl
df_d=pd.concat([df_d,cl], axis=1)
df_d.head()

df_d['clusters'].value_counts()
df_d.groupby('clusters').agg(['mean']).reset_index()

# ============================================================================= 
# Step-5: Inferences:

'''

1. Hiearchial Clustering
    Results of Both Average Method and Complete Method are almost Similar.
    
2. K- Means Clustering
    We can go for Cluster-1 Which is safer than other clusters.
    
2. DBSCAN Clustering
    We can go for Cluster-1 Which is safer than other clusters.
'''
#==============================================================================

1. Hiearchial Clustering
    Results of Both Average method and Complete Method are almost Similar
    
2. K- Means Clustering
    We can go for Cluster-1 Which is safer than other clusters.
    
3. DBSCC Clustering
        We can go for Cluster-1 Which is safer than other clusters.


