# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:05:30 2013

@author: mohit
"""

#Gaussian-Bernoulli RBM

import numpy as np
import h5py

class rbm():
    
    def __init__(self,indims,outdims,lr,momentum,epochs,chunksize,batchsize):
        self.dims = indims
        self.hiddens = outdims
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.chunksize = chunksize
        self.batchsize = batchsize
        self.wts = (np.asarray(np.random.uniform(
                      low=-4 * np.sqrt(6. / (outdims + indims)),
                      high=4 * np.sqrt(6. / (outdims + indims)),
                      size=(indims, outdims)))).astype('double')
        self.inb = np.zeros((indims,1)).astype('double')
        self.outb = np.zeros((outdims,1)).astype('double')
        self.w_prev = np.load('../params/weights.npy')
        self.b_prev = np.load('../params/out.npy')
        
    def cd(self,data,bsize,w,c,b):
        numloops = 1
        poshid = 1/(1+(np.exp(-np.dot(data,w) - b.T)))
        poshidstates = (poshid > np.random.rand(bsize,self.hiddens)).astype('double')
        poshidact = np.sum(poshid,axis=0)
        posvisact = np.sum(data,axis=0)
        posvh = np.dot(data.T,poshid)        
        for i in range(numloops):
            negvis = 1/(1+(np.exp(-np.dot(poshidstates,w.T) - c.T)))
            negvisstates = (negvis > np.random.rand(bsize,self.dims)).astype('double')
            neghid = 1/(1+(np.exp(-np.dot(negvisstates,w) - b.T)))
            neghidstates = (neghid > np.random.rand(bsize,self.hiddens)).astype('double')            
            data = negvisstates
            poshid = neghid
            poshidstates = neghidstates
        negvh = np.dot(negvisstates.T,neghid)
        neghidact = np.sum(neghid,axis=0)
        negvisact = np.sum(negvisstates,axis=0)
        Evh = (posvh - negvh)/bsize
        Ev = ((posvisact - negvisact)/bsize).reshape(-1,1)
        Eh = ((poshidact - neghidact)/bsize).reshape(-1,1)
        return Evh, Ev, Eh

    def update(self,w,c,b,dw,dc,db):        
        w = w + self.lr*(self.momentum*dw+dw)
        c = c + self.lr*(self.momentum*dc+dc)
        b = b + self.lr*(self.momentum*db+db)
        return w,c,b    


    def shuffled_chunks(self,csize,prev):
        tmp = np.zeros((csize,self.prev_dims))
        self.data.read_direct(tmp,np.s_[prev:(prev+csize),:])
        idx = np.arange(csize)
        np.random.shuffle(idx)
        tmp = np.take(tmp,idx,axis=0)
        tmp = 1/(1+(np.exp(-np.dot(tmp,self.w_prev) - self.b_prev.T)))
        return tmp

    def process_batch(self,batchdata,bsize):
        dw,dc,db = self.cd(batchdata,bsize,self.wts,self.inb,self.outb)
        self.wts,self.inb,self.outb = self.update(self.wts,self.inb,
                                                  self.outb,dw,dc,db)
        
    def train(self):
        for epoch in np.arange(self.epochs):
            print 'Epoch: ' + str(epoch+1)
            prev = 0
            chunk_idx = 0
            for chunk in np.arange(0,self.trainsize,self.chunksize):
                if (((chunk_idx+1) % 5) == 0):                    
                    print 'Chunk: ' + str(chunk_idx+1)
                csize,_ = np.shape(self.data[chunk:(chunk+self.chunksize),:])
                chunkdata = self.shuffled_chunks(csize,prev)
                prev += np.shape(chunkdata)[0]
                for batch in np.arange(0,csize,self.batchsize):
                    batchdata = chunkdata[batch:(batch+self.batchsize)]
                    bsize = np.shape(batchdata)[0]
                    self.process_batch(batchdata,bsize)
                chunk_idx += 1
            #print self.reconstruction_error()
        
        self.fi.close()
    
    def reconstruction_error(self):
        data = np.zeros((self.trainsize,self.dims))
        self.data.read_direct(data,np.s_[0:self.trainsize,:])
        hid = 1/(1+(np.exp(-np.dot(data,self.wts) - self.outb.T)))
        hidstates = (hid > np.random.rand(self.trainsize,self.hiddens)).astype('double')
        vis = np.dot(hidstates,self.wts.T) + self.inb.T
        err = np.sum((data-vis)**2)/self.trainsize
        return err

    def set_data(self,path):
        self.fi = h5py.File(path,'r')
        self.data = self.fi['mfcc']                
        self.trainsize = np.shape(self.data)[0]
        self.prev_dims = np.shape(self.data)[1]
        #self.trainsize = 200000
    
    def get_params(self):
        return self.wts, self.inb, self.outb
