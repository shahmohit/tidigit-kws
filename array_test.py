# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:49:26 2014

@author: mohit
"""
import numpy as np

samples = 25194
x = np.zeros((samples,10))
bsize = 100

for i in range(0,samples,bsize):
    y = x[i:(i+bsize),:]