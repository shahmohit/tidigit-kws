# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:37:28 2014

@author: mohit
"""
import os
import pickle
import h5py

words = ['zero','o','one','two','three','four','five',
         'six','seven','eight','nine','sil']
         
exp = 'test'
adir = '../alignments'

path = os.path.join(adir,exp)
afi = open(path,'r')
alignments = pickle.load(afi)
afi.close()

sdir = '../data'
path = os.path.join(sdir,exp+'.spkrs')
sfi = open(path,'r')
spkrs = pickle.load(sfi)
sfi.close()

labels = []
utt_length = []
utt_boundary = []
utt_spkr = []
utt_id = []

keys = alignments.keys()
keys.sort()

prev_boundary = 0

k = 0
for key in keys:
    tmp = []
    duration = alignments[key]['word_durations']
    transcript = alignments[key]['transcript']
    spkrid = spkrs[key]
    prev_D = 0
    for i in range(len(transcript)):
        D = duration[i] - prev_D
        W = transcript[i]
        tmp += [words.index(W) for j in range(D)]
        prev_D  = duration[i]
        utt_spkr += [spkrid for j in range(D)]
        utt_id += [k for j in range(D)]
    utt_length.append(len(tmp))
    utt_boundary.append(prev_boundary)
    prev_boundary += len(tmp)
    labels += tmp
    k += 1

tmp = {}
tmp['labels'] = labels
tmp['duration'] = utt_length
tmp['boundary'] = utt_boundary
tmp['id'] = keys
tmp['utt_spkr'] = utt_spkr
tmp['utt_id'] = utt_id

dpath = os.path.join(sdir,exp+'.hdf5')
if os.path.exists(dpath):
    os.remove(dpath)
    
f = h5py.File(dpath,'w')
data = f.create_dataset('durations', (len(utt_length),))
data[:] = utt_length
data = f.create_dataset('segments', (len(utt_boundary),))
data[:] = utt_boundary
data = f.create_dataset('spkrid', (len(labels),))
data[:] = utt_spkr
data = f.create_dataset('uttid', (len(labels),))
data[:] = utt_id
data = f.create_dataset('labels', (len(labels),))
data[:] = labels

f.close()