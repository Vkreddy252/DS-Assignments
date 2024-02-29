# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:27:34 2023

@author: Vinu
"""

# Step-1: Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
book=pd.read_csv('book.csv')
book

#==============================================================================
# Step-2 Apriori Algorithm

# 1.Association Rules with 10% Support and 70% confidence

# With 10% Support

frequent_itemsets1=apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets1

# with 70% confidence
rules1=association_rules(frequent_itemsets1,metric='lift',min_threshold=0.7)
rules1

rules1.sort_values('lift',ascending=False)
rules1[rules1.lift>1]

# visualization of obtained rule
plt.scatter(rules1['support'],rules1['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

#------------------------------------------------------------------------------

# 2.Association Rules with 20% Support and 60% confidence

# With 20% Support

frequent_itemsets2=apriori(book,min_support=0.2,use_colnames=True)
frequent_itemsets2

# with 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2

rules2.sort_values('lift',ascending=False)
rules2[rules2.lift>1]

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

#------------------------------------------------------------------------------

# 3.Association Rules with 5% Support and 80% confidence

# With 5% Support

frequent_itemsets3=apriori(book,min_support=0.05,use_colnames=True)
frequent_itemsets3

# with 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3

rules3.sort_values('lift',ascending=False)
rules3[rules3.lift>1]

# visualization of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

#==============================================================================