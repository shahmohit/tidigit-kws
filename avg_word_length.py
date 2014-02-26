# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:37:28 2014

@author: mohit
"""
import os
import pickle
import numpy as np

word_lengths = []

exps = ['train','test']
for exp in exps:
    adir = '../alignments'
    
    path = os.path.join(adir,exp)
    afi = open(path,'r')
    alignments = pickle.load(afi)
    afi.close()
    
    keys = alignments.keys()
    keys.sort()
    
    for key in keys:
        tmp = []
        duration = alignments[key]['word_durations']
        prev_D = 0
        for i in range(len(duration)):
            D = duration[i] - prev_D
            word_lengths.append(D)
            prev_D  = duration[i]

print np.mean(word_lengths)
print np.std(word_lengths)