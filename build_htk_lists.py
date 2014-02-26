# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:21:16 2014

@author: mohit
"""
import os
import csv

exps = ["train","test"]
ldir = '../lists'
fedir = '../fe'
if not os.path.exists(ldir):
    os.mkdir(ldir)
if not os.path.exists(fedir):
    os.mkdir(fedir)

for exp in exps:
    path = os.path.join('..',exp+'_wav.scp')
    fi = csv.reader(open(path,'rb'),delimiter=' ')
    path = os.path.join(ldir,exp+'.htk')
    f1 = open(path,'wb')
    path = os.path.join(ldir,exp+'.list')
    f2 = open(path,'wb')
    for rows in fi:
        src = rows[1]
        tmp = rows[0]
        dst = os.path.join(fedir,tmp+'.htk')
        line = src + ' ' + dst + '\n'
        f1.write(line)
        f2.write(tmp+'\n')
    f1.close()
    f2.close()