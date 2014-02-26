# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:54:37 2014

@author: mohit
"""

import os
import GRBM

frames = 31
dims = 13
indims = frames*dims
hiddens = 100

datapath = os.path.join('../data','train.context.hdf5')
rbm = GRBM.rbm(indims=indims,outdims=hiddens,lr=0.002,momentum=0.5,
               epochs=10,chunksize=10000,batchsize=100)
rbm.set_data(datapath)
rbm.train()