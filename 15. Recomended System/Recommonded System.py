# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:55:34 2023

@author: Vinu
"""
# Step-1: Importing Required Libraries and Files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

book=pd.read_csv("book.csv",encoding = "ISO-8859-1")
book
book.head()
#==============================================================================

# Step-2: EDA

book.info()
book.isnull().sum()
book.drop(book.columns[[0]],axis=1,inplace =True)
book
book.nunique()

# Renaming the columns
book.columns = ["UserID","BookTitle","BookRating"]
book

book =book.sort_values(by=['UserID'])

# number of unique users in the dataset
len(book.UserID.unique())

len(book.BookTitle.unique())

book.loc[book["BookRating"] == 'small', 'BookRating'] = 0
book.loc[book["BookRating"] == 'large', 'BookRating'] = 1
book.BookRating.value_counts()

plt.figure(figsize=(10,6))
sns.distplot(book.BookRating)


book_df = book.pivot_table(index='UserID',
                   columns='BookTitle',
                   values='BookRating').reset_index(drop=True)
book_df.fillna(0,inplace=True)
book_df

# Average Rating of Books

Avg = book['BookRating'].mean()
print(Avg)

# Calculating the minimum number of votes required to be in the chart, 
minimum = book['BookRating'].quantile(0.90)
print(minimum)

# Filtering out all qualified Books into a new DataFrame
q_Books = book.copy().loc[book['BookRating'] >= minimum]
q_Books.shape

#==============================================================================
# Step-3: Calculating Cosine Similarities

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
user_sim=1-pairwise_distances(book_df.values,metric='cosine')
user_sim

user_sim_df=pd.DataFrame(user_sim)
user_sim_df

# Setting the index and column names to user ids 
user_sim_df.index = book.UserID.unique()
user_sim_df.columns = book.UserID.unique()
user_sim_df

np.fill_diagonal(user_sim,0)
user_sim_df

# Most Similar Users
print(user_sim_df.idxmax(axis=1)[1348])
print(user_sim_df.max(axis=1).sort_values(ascending=False).head(50))

reader = book[(book['UserID']==1348) | (book['UserID']==2576)]
reader

reader1=book[(book['UserID']==1348)] 
reader1

reader2=book[(book['UserID']==2576)] 
reader2

# Result: BookTitle with Stardust whose UserID 2576 has BookRating 10