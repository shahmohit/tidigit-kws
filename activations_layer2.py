# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:54:37 2014

@author: mohit
"""

import os
import h5py
import numpy as np
from sklearn import linear_model
import pickle

def chunks(csize,prev):
    tmp = np.zeros((csize,dims))
    data.read_direct(tmp,np.s_[prev:(prev+csize),:])
    tmplabs = newlabels[prev:prev+csize]
    return tmp,tmplabs

w = np.load('../params/weights.npy')
b = np.load('../params/out.npy')
w_2 = np.load('../params/weights_layer2.npy')
b_2 = np.load('../params/out_layer2.npy')

dims,hiddens = np.shape(w)
epochs = 10
chunksize = 10000
batchsize = 500
keywords = 11

for keyword in range(3,keywords):
    print 'Keyword: ' + str(keyword)
    path = os.path.join('../data','train.hdf5')
    fi = h5py.File(path,'r')
    data = fi['labels']
    labels = np.zeros(np.shape(data)).astype('int')
    data.read_direct(labels)
    
    newlabels = np.zeros(np.shape(data)).astype('int')
    idx = np.where(labels==keyword)[0]
    newlabels[idx] = 1
    idx = np.where(labels!=keyword)[0]
    newlabels[idx] = 0
    
    fi.close()
    
    # Test labels
    path = os.path.join('../data','test.hdf5')
    fi = h5py.File(path,'r')
    data = fi['labels']
    labels = np.zeros(np.shape(data)).astype('int')
    data.read_direct(labels)
    
    testlabels = np.zeros(np.shape(data)).astype('int')
    idx = np.where(labels==keyword)[0]
    testlabels[idx] = 1
    idx = np.where(labels!=keyword)[0]
    testlabels[idx] = 0
    
    fi.close()
    
    clf = linear_model.SGDClassifier(loss='log',class_weight='auto')
    
    path = os.path.join('../data','train.context.hdf5')
    fi = h5py.File(path,'r')
    data = fi['mfcc']                
    trainsize = np.shape(data)[0]
    for epoch in np.arange(epochs):
        prev = 0
        chunk_idx = 0
        for chunk in np.arange(0,trainsize,chunksize):
            csize,_ = np.shape(data[chunk:(chunk+chunksize),:])
            chunkdata,labs = chunks(csize,prev)
            prev += np.shape(chunkdata)[0]
            for batch in np.arange(0,csize,batchsize):
                batchdata = chunkdata[batch:(batch+batchsize)]
                batchlabs = labs[batch:(batch+batchsize)]
                bsize = np.shape(batchdata)[0]
                hid = 1/(1+(np.exp(-np.dot(batchdata,w) - b.T)))
                hid2 = 1/(1+(np.exp(-np.dot(hid,w_2) - b_2.T)))
                clf.partial_fit(hid2,batchlabs,classes=[0,1])
            chunk_idx += 1
    
    prev = 0
    chunk_idx = 0
    pred = []
    for chunk in np.arange(0,trainsize,chunksize):
        csize,_ = np.shape(data[chunk:(chunk+chunksize),:])
        chunkdata,labs = chunks(csize,prev)
        prev += np.shape(chunkdata)[0]
        hid = 1/(1+(np.exp(-np.dot(chunkdata,w) - b.T)))
        hid2 = 1/(1+(np.exp(-np.dot(hid,w_2) - b_2.T)))
        predtmp = clf.predict(hid2)
        pred += list(predtmp)
        chunk_idx += 1
    
    fi.close()
    
    cm = np.zeros((2,2))
    for p,t in zip(pred,newlabels):
        cm[t,p] += 1
    
    wr = np.trace(cm)/np.sum(cm)
    print wr
    print np.diag(cm)/np.sum(cm,axis=1)
    
    # Test Set Evaluation
    
    path = os.path.join('../data','test.context.hdf5')
    fi = h5py.File(path,'r')
    data = fi['mfcc']                
    testsize = np.shape(data)[0]
    
    prev = 0
    chunk_idx = 0
    pred = []
    for chunk in np.arange(0,testsize,chunksize):
        csize,_ = np.shape(data[chunk:(chunk+chunksize),:])
        chunkdata,labs = chunks(csize,prev)
        prev += np.shape(chunkdata)[0]
        hid = 1/(1+(np.exp(-np.dot(chunkdata,w) - b.T)))        
        hid2 = 1/(1+(np.exp(-np.dot(hid,w_2) - b_2.T)))
        predtmp = clf.predict(hid2)
        pred += list(predtmp)
        chunk_idx += 1
    
    fi.close()
    
    tstcm = np.zeros((2,2))
    for p,t in zip(pred,testlabels):
        tstcm[t,p] += 1
    
    wr = np.trace(tstcm)/np.sum(tstcm)
    print wr
    print np.diag(tstcm)/np.sum(tstcm,axis=1)
    
    path = '../classifiers/'+str(keyword)+'.pkl'
    fi = open(path,'wb')
    pickle.dump(clf,fi)
    fi.close()
