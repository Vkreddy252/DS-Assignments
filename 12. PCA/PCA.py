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

df = pd.read_csv("wine.csv")
df.head()
df.shape
#==============================================================================
# Step-2:  EDA  

df.duplicated().sum()
df.isnull().sum()
df.describe().round(2) 
df.info()

# Correlation   
cor = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='Greens')
plt.show()
    
# Box Plots and Histograms for Independent Variables

cols = ['Alcohol','Malic', 'Ash','Alcalinity', 
        'Magnesium','Phenols','Flavanoids','Nonflavanoids','Proanthocyanins',
        'Color','Hue','Dilution','Proline']

# Box Plots
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(5,3,i)
    sns.boxplot(y=col, data=df,width=0.2)
    i+=1
plt.show()

# Histograms
plt.figure(figsize=(10,15))
i=1
for col in cols:
    plt.subplot(5,3,i)
    sns.histplot(x=col, data=df,kde=True,color='blue')
    i+=1
plt.show()
# Pair plot
sns.pairplot(data=df, hue = 'Type')

#==============================================================================
# Step-3: Data Transformation

X = df.iloc[:,1:]

# Standardization for X 

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
X1 = pd.DataFrame(SS_X)
X1.columns =list(X)
X1.head()

Y = df["Type"]

#==============================================================================
# Step-4: PCA
from sklearn.decomposition import PCA
pca = PCA()
pca_values = pca.fit_transform(X1)
pca_values

# Variance of Each PCA 
var = pca.explained_variance_ratio_
var
# Cumulative Varaince of Each PCA
var1 = np.cumsum(np.round(var,decimals= 4)*100)
var1
plt.plot(var1,color="blue")
# Final Data Frame
final_df=pd.concat([df['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df

#==============================================================================
# Step-5: Visualization of PCA'S
# Pair Plot

fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df)

# Scatter Plot
sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Type')

x= pca_values[:,0:1]
y= pca_values[:,1:2]
plt.scatter(x,y)

#==============================================================================
# Step-6: Checking with other Clustering Algorithms

# 1. Hiearchial Clustering

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Dendogram
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(X1,'complete'))

# Creating Clusters
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters
Y1=pd.DataFrame(hclusters.fit_predict(X1),columns=['clustersid'])
Y1['clustersid'].value_counts()

# Adding clusters to dataset
df1=df.copy()
df1['Cluster Id']=hclusters.labels_
df1

# 2.K-Means Clustering
from sklearn.cluster import KMeans
X2=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(X1)
    X2.append(kmeans.inertia_)
# Elbow Graph
plt.plot(range(1,6),X2)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('X2');

# Cluster algorithm using K=3
clusters3=KMeans(3,random_state=30).fit(X1)
clusters3
clusters3.labels_
df2=df.copy()
df2['clusters3id']=clusters3.labels_
df2
df2['clusters3id'].value_counts()

# ========================= ***** ============================================= 



