# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 02:11:46 2014

@author: mohit
"""
import numpy as np
import itertools

phones = ['unk','sil','ao','ey','ih','v','uw','w','k','iy','z',
          'n','ah','ow','eh','ay','r','s','f','th','t']

words = ['zero','o','one','two','three','four','five',
         'six','seven','eight','nine','sil']
words_short = ['z','o','1','2','3','4','5','6','7','8','9']

words_phones = [['z','iy','r','ow'],['ow'],['w','ah','n'],
                ['t','uw'],['th','r','iy'],['f','ao','r'],
                ['f','ay','v'],['s','ih','k','s'],
                ['s','eh','v','ah','n'],['ey','t'],['n','ay','n']]            
                
exp = 'train'
trans = '../trans/'+exp+'_text'
utt_ph = '../lists/'+exp+'.phones'
utt_w = '../lists/'+exp+'.words'

tfi = open(trans,'r')
pfi = open(utt_ph,'w')
wfi = open(utt_w,'w')

ttmp = []
for line in tfi.readlines():
    line = line.rstrip('\n')
    contents = line.split(' ')
    utt = contents[0]
    contents = contents[1:]    
    ws = [words_short.index(x) for x in contents]
    w = [words[x] for x in ws]
    w_str = ' '.join(w) + '\n'
    ph = [words_phones[x] for x in ws]
    ph_unfold = list(itertools.chain(*ph))
    ph_id = [str(phones.index(x)) for x in ph_unfold]    
    ph_id = ' '.join(ph_id)
    final = utt + ' ' + ph_id + '\n'
    pfi.write(final)
    wfi.write(w_str)
                
tfi.close()
pfi.close()
wfi.close()