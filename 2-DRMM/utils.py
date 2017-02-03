# coding: utf-8
import numpy as np 
import sys
from keras.callbacks import EarlyStopping, TensorBoard

from settings import * 


def get_idx_pairs(qids, instances):
    idx_pairs = []
    for qid in qids:
        for posid, negid in instances[qid]:
            idx_pairs.append( (qid,posid, negid) )
    return idx_pairs


def batch_generator(idx_pairs, qid_docid2histvec, IDFs, batch_size=BATCH_SZ):
    '''returns training data generator for ranking model, 
    the returned generator yields `[idfs_batch, pos_batch, neg_batch], y_true_batch_dummy`
    parameters:
    *  `idx_pairs`: is list of tuple [ (qid, pos_docid, neg_docid) ]
    * `qid_docid2histvec`: is a dict mapping (qid, docid) to the histogram vector
    * `IDFs`: is a dict mapping qid to its corresponding query idf vector
    ''' 
    np.random.shuffle(idx_pairs)
    batches_pre_epoch = len(idx_pairs) // batch_size
    samples_per_epoch = batches_pre_epoch * batch_size # make samples_per_epoch a multiple of batch size
    counter = 0
    y_true_batch_dummy = np.ones((batch_size))
    while 1:
        idx_batch = idx_pairs[batch_size*counter: min(samples_per_epoch, batch_size*(counter+1))]
        idfs_batch, pos_batch, neg_batch = [], [], []
        for qid, pos_docid, neg_docid in idx_batch:
            idfs_batch.append(IDFs[qid])
            pos_batch.append(qid_docid2histvec[(qid,pos_docid)].reshape(-1,N_HISTBINS))
            neg_batch.append(qid_docid2histvec[(qid,neg_docid)].reshape(-1,N_HISTBINS))
        idfs_batch, pos_batch, neg_batch = map(np.array, [idfs_batch, pos_batch, neg_batch])
        counter += 1
        if (counter >= batches_pre_epoch):
            np.random.shuffle(idx_pairs)
            counter=0
        yield [idfs_batch, pos_batch, neg_batch], y_true_batch_dummy


def TREC_output(qid, scoring_model, candidates, qid_docid2histvec, IDFs, \
                run_name = 'my_run', fpath = None):
    res = [] # list of (score, pmcid) tuples
    for docid in candidates[qid]:
        input_idf = IDFs[qid].reshape((1,-1)) # (-1, QLEN)
        input_hist = qid_docid2histvec[(qid,docid)]
        score = scoring_model.predict([input_idf, input_hist])[0]
        res.append( (score, docid) )
    res = sorted(res, reverse=True)
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, docid) in enumerate(res[:2000]):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)

_callbacks = [ EarlyStopping(monitor='val_loss', patience=2),
               TensorBoard(log_dir=LOGDIR, histogram_freq=0, write_graph=False) ]


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def KFold(qids, ranking_model, scoring_model, data_pickle, K = 5, \
            fpath=None, run_name = 'my_run', batch_size=BATCH_SZ, nb_epoch = N_EPOCH):
    instances         = data_pickle['instances'] 
    IDFs              = data_pickle['IDFs']
    qid_docid2histvec = data_pickle['qid_docid2histvec']
    candidates        = data_pickle['candidates']
    # wrapper functions
    def wrapper_batch_generator(idx_pairs): 
        return batch_generator(idx_pairs, qid_docid2histvec, IDFs, batch_size)
    def wrapper_TREC_output(scoring_model, qid):
        return TREC_output(qid, scoring_model, candidates, qid_docid2histvec, IDFs, run_name, fpath)
    
    initial_weights = ranking_model.get_weights()
    np.random.seed(0)
    np.random.shuffle(qids)
    fold_sz = len(qids) / K

    open(fpath,'w').close() # clear previous content in file 
    for fold in xrange(K):
        print 'fold %d' % fold, 
        val_start, val_end = fold*fold_sz, (fold+1)*fold_sz
        qids_val = qids[val_start:val_end] # train/val queries for each fold 
        qids_train = qids[:val_start] + qids[val_end:]
        print qids_val
        idx_pairs_train = get_idx_pairs(qids_train,instances)
        idx_pairs_val = get_idx_pairs(qids_val,instances)
        
        shuffle_weights(ranking_model, initial_weights) # reset model parameters
        ranking_model.fit_generator( 
                    generator         = wrapper_batch_generator(idx_pairs_train), # train model 
                    samples_per_epoch = len(idx_pairs_train)//batch_size*batch_size,
                    nb_epoch          = nb_epoch,
                    validation_data   = wrapper_batch_generator(idx_pairs_val),
                    nb_val_samples    = len(idx_pairs_val)//batch_size*batch_size, 
                    callbacks         = _callbacks )
        print 'fold %d complete, outputting to %s...' % (fold, fpath)
        for qid in qids_val:
            wrapper_TREC_output(scoring_model, qid)