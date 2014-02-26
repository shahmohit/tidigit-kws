# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:37:28 2014

@author: mohit
"""
import os
import pickle
import csv

exp = 'train'
ldir = '../lists'

path = os.path.join(ldir,exp+'.utt2spk')
fi = csv.reader(open(path,'r'),delimiter=' ')

keys = []
spkr = []
for row in fi:
    keys.append(row[0])
    spkr.append(row[1])

spkrs = list(set(spkr))
spkrs.sort()

sdict = {}
i = 0
for key in keys:
    sdict[key] = spkrs.index(spkr[i])
    i += 1

rdir = '../data'
if not os.path.exists(rdir):
    os.mkdir(rdir)

path = os.path.join(rdir,exp+'.spkrs')
fi = open(path,'w')
pickle.dump(sdict,fi)
fi.close()