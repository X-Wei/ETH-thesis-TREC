### utility functions for data preparing 
import numpy as np 
def to_khot(sidhid2icds, icd_ctr, K):
    '''generate khot encoding dict 
    * sidhid2icds is a dict[(int,int), set(str)], maps (sid,hid) pair to all icd codes for this patient/stay
    * icd_ctr is a Counter obj of icd codes 
    * keep top K-1 most freq icd codes (plus one 'other' label) 
    returns `sidhid2khot:dict[(int,int), np.array]`, that maps (sid,hid) to a khot encoding vector 
    returns the topicds as well 
    '''
    topicds = zip( *icd_ctr.most_common(K-1) )[0] + ('other',)
    # now turn each subject into a k-hot vector
    sidhid2khot = {} # map subject_id to k-hot vector
    for sid,hid in sidhid2icds.keys():
        _khot = np.zeros(K)
        for _icd in sidhid2icds[(sid,hid)]:
            if _icd in topicds: 
                _khot[topicds.index(_icd)] = 1
            else: # label 'other icds'
                _khot[-1] = 1
        if sum(_khot) == 0: print 'strange case: ', (sid,hid)
        sidhid2khot[(sid,hid)] = _khot
    return sidhid2khot, topicds 

def getXY(sidhid_lst, sidhid2seq, sidhid2khot): 
    '''given a list of (sid, hid) pairs, generate the X and Y matrices 
    sidhid2seq, sidhid2khot are dicts mapping (sid,hid) pairs to (padded)sequences/khot encodings
    '''
    data, labels = [], []
    for sidhid in sidhid_lst:
        data.append(sidhid2seq[sidhid])
        labels.append(sidhid2khot[sidhid])
    X = np.array(data)
    Y = np.array(labels)
    return X,Y


### utility functions for model training
import keras.backend as K
import os, sys, time 
from settings import * 
''' ***NOTE***
To load models from file, we have to modify `metrics.py` at: 
`/local/XW/SOFT/anaconda2/envs/thesis_nb/lib/python2.7/site-packages/keras` 
to add the `multlabel_XXX` function, otherwise throws exception ! 

cf issue: https://github.com/fchollet/keras/issues/3911
m = load_model(os.path.sep.join([MODEL_PATH, 'model_1conv1d.h5']))
'''

def multlabel_prec(y_true, y_pred):
    y_pred, y_true = y_pred[:,:-1], y_true[:,:-1] # test without last column considered
    y_pred = K.round(K.clip(y_pred, 0, 1)) # turn to 0/1 
    tp = K.sum(y_true * y_pred, axis =-1)
    sum_true = K.sum(y_true, axis=-1)
    sum_pred = K.sum(y_pred, axis=-1)
    return K.mean(tp/(sum_pred+1e-10)) # to avoid NaN precision

def multlabel_recall(y_true, y_pred):
    y_pred, y_true = y_pred[:,:-1], y_true[:,:-1] # test without last column considered
    y_pred = K.round(K.clip(y_pred, 0, 1)) # turn to 0/1 
    tp = K.sum(y_true * y_pred, axis =-1)
    sum_true = K.sum(y_true, axis=-1)
    sum_pred = K.sum(y_pred, axis=-1)
    return K.mean(tp/(sum_true+1e-10)) 

def multlabel_F1(y_true, y_pred):
    y_pred, y_true = y_pred[:,:-1], y_true[:,:-1] # test without last column considered
    y_pred = K.round(K.clip(y_pred, 0, 1)) # turn to 0/1 
    tp = K.sum(y_true * y_pred, axis =-1)
    sum_true = K.sum(y_true, axis=-1)
    sum_pred = K.sum(y_pred, axis=-1)
    return 2*K.mean(tp/(sum_true+sum_pred+1e-10))

def multlabel_acc(y_true, y_pred):
    y_pred, y_true = y_pred[:,:-1], y_true[:,:-1] # test without last column considered
    y_pred = K.round(K.clip(y_pred, 0, 1)) # turn to 0/1 
    intersect = y_true * y_pred
    intersect = K.sum(intersect, axis=-1)
    union = K.clip(y_true+y_pred, 0, 1)
    union = K.sum(union, axis=-1)
    return K.mean(intersect/(union+1e-10))