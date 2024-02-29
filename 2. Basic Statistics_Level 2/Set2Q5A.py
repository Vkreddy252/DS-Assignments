# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:00:57 2023

@author: Vinu
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
Mean1 = 5
Mean2 = 7
Mean = Mean1+Mean2
print('The Mean Profit is Rs', Mean*45,'Million')
STD1 = 3**2
STD2 = 4**2 
STD = np.sqrt((9)+(16))
print('The Standard Deviation is Rs', STD*45, 'Million')
print('Range is Rs',(stats.norm.interval(0.95,540,225)),'in Millions')