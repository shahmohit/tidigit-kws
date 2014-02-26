# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 02:11:46 2014

@author: mohit
"""
import numpy as np
import itertools
import utils
import os
import pickle

phones = ['unk','sil','ao','ey','ih','v','uw','w','k','iy','z',
          'n','ah','ow','eh','ay','r','s','f','th','t']

words = ['zero','o','one','two','three','four','five',
         'six','seven','eight','nine','sil']
words_short = ['z','o','1','2','3','4','5','6','7','8','9','sil']

words_phones = [['z','iy','r','ow'],['ow'],['w','ah','n'],
                ['t','uw'],['th','r','iy'],['f','ao','r'],
                ['f','ay','v'],['s','ih','k','s'],
                ['s','eh','v','ah','n'],['ey','t'],['n','ay','n'],['sil']]            

false = []               
exp = 'train'

utt_ph = '../lists/'+exp+'.phones'
pfi = open(utt_ph,'r')

utt_w = '../lists/'+exp+'.words'
wfi = open(utt_w,'r')

align = '../kaldi_align/'+exp+'.align'
afi = open(align,'r')

true_trans = []
true_words = []

word_align = []
length_align = []
phone_align = []
durations = []
uttids = []

# Get word boundaries
for wline,pline,aline in zip(wfi.readlines(),pfi.readlines(),afi.readlines()):
    pline = pline.rstrip('\n')
    contents = pline.split(' ')
    utt = contents[0]
    contents = contents[1:]
    true_trans.append(contents)

    wline = wline.rstrip('\n')
    wcontents = wline.split(' ')
    true_words.append(wcontents)
    
    aline = aline.rstrip('\n')
    aline = aline.split(';')
    tmp = aline.pop(0).rstrip(' ').split(' ')
    utt_a = tmp.pop(0)
    lengths = [int(tmp[1])]
    ph_id = [int(tmp[0])]
    
    aline = [x.rstrip(' ') for x in aline]
    aline = [x.lstrip(' ') for x in aline]
    aline = [x.split(' ') for x in aline]    
    p = [int(row[0]) for row in aline]
    l = [int(row[1]) for row in aline]
    
    align_trans = list(itertools.chain(ph_id,p))
    align_lengths = list(itertools.chain(lengths,l))
    w_idx = []

    start = 0 
    for i in range(len(wcontents)):
        tmp = words_phones[words.index(wcontents[i])]
        p = [phones.index(x) for x in tmp]        
        idx = utils.KnuthMorrisPratt(align_trans,p)
        w_idx = list(itertools.chain(w_idx,idx))
    
    w_idx = list(set(w_idx))
    w_idx.sort()
    phone_align.append(align_trans)
    length_align.append(align_lengths)
    word_align.append(w_idx)
    uttids.append(utt_a)
pfi.close()
afi.close()
wfi.close()

# Add silences
for i in range(len(true_words)):
    ph = phone_align[i]
    w = word_align[i]
    tr = true_words[i]
    sil_pos = [j for j in range(len(ph)) if ph[j]==1]
    w += sil_pos
    for j in range(len(sil_pos)):
        tr.append('sil')
    if len(w) != len(tr):
        print 'Error: ' + str(i)
    idx = range(len(w))
    idx.sort(key = w.__getitem__)
    w[:] = [w[k] for k in idx]
    tr[:] = [tr[k] for k in idx]
    word_align[i] = w
    true_words[i] = tr

for i in range(len(true_words)):
    dur = []
    w = word_align[i]
    l = length_align[i]
    for j in range(len(w)-1):
        tmp = np.sum(l[w[j]:w[j+1]])        
        dur.append(tmp)
    tmp = np.sum(l[w[len(w)-1]:])
    dur.append(tmp)
    dur = np.cumsum(dur)
    durations.append(dur)


# Verify if all is OK

# Durations first
dfalse = []
for i in range(len(true_words)):
    tmp1 = durations[i]
    tmp2 = length_align[i]
    tmp1 = tmp1[-1]
    tmp2 = sum(tmp2)
    if tmp1 != tmp2:
        dfalse.append(i)

if not dfalse:
    print "Duration match...OK"
else:
    print "Durations do not match. Check."
    
# Reconstruct force-alignments from our word results and check
pfalse = []
for i in range(len(true_words)):
    w = true_words[i]
    ph = phone_align[i]
    ph_recon = []
    for x in w:
        tmp = words.index(x)
        p = words_phones[tmp]
        for y in p:
            dx = phones.index(y)
            ph_recon.append(dx)
    if ph != ph_recon:
        pfalse.append(i)

if not pfalse:
    print "Phone/Word Alignments...OK"
else:
    print "Alignments not correct. Check."

# Write to file
alignments = {}
for i in range(len(true_words)):
    tmp = {}
    u = uttids[i]
    tmp['phone_align'] = phone_align[i]
    tmp['word_align'] = word_align[i]
    tmp['phone_durations'] = length_align[i]
    tmp['word_durations'] = durations[i]
    tmp['transcript'] = true_words[i]
    alignments[u] = tmp

adir = '../alignments'
if not os.path.exists(adir):
    os.mkdir(adir)

path = os.path.join(adir,exp)
afi = open(path,'w')
pickle.dump(alignments,afi)
afi.close()
