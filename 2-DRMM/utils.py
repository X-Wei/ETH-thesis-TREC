# coding: utf-8
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
        input_idf = np.array(IDFs[qid]).reshape((1,-1)) # (-1, QLEN)
        input_hist = qid_docid2histvec[(qid,docid)]
        score = scoring_model.predict([input_idf, input_hist])[0]
        res.append( (score, docid) )
    res = sorted(res, reverse=True)
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, docid) in enumerate(res[:2000]):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)

_callbacks = [ # EarlyStopping(monitor='val_loss', patience=2),
               TensorBoard(log_dir=LOGDIR, histogram_freq=0, write_graph=False) ]


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def KFold(ranking_model, scoring_model, data_pickle,
            K = 5, initial_weights=None, instances=None, qids = None,\
            batch_size=BATCH_SZ, nb_epoch = N_EPOCH, verbose=1, \
            fpath=None, run_name = 'my_run'):
    if instances is None: 
        instances     = data_pickle['instances'] 
    if qids is None: 
        qids = data_pickle['QUERIES'].keys()
    IDFs              = data_pickle['IDFs']
    qid_docid2histvec = data_pickle['qid_docid2histvec']
    candidates        = data_pickle['candidates']

    # wrapper functions
    def wrapper_batch_generator(idx_pairs): 
        return batch_generator(idx_pairs, qid_docid2histvec, IDFs, batch_size)
    def wrapper_TREC_output(scoring_model, qid):
        return TREC_output(qid, scoring_model, candidates, qid_docid2histvec, IDFs, run_name, fpath)
    
    if initial_weights is None: 
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
                    callbacks         = _callbacks,
                    verbose           = verbose )
        print 'fold %d complete, outputting to %s...' % (fold, fpath)
        for qid in qids_val:
            wrapper_TREC_output(scoring_model, qid)


def gen_instances(QUERIES, relevance, candidates, n_pos, mode = 'quantiles', verbose=1):
    '''generate an `instance: dict[int, list<str, str>]` mapping qid to the list of (pos_docid, neg_docid) pairs
    meaning of the parameters can be found in `data_prep.py`, these are pickled into a local file. 
    there are 2 modes: 
    * mode `quantiles`: sample more pairs for queries with less positive docids -- origin DRMM source code use this, cf NN4IR.cpp line 218
    * mode `uniform`: each qid generates the same number (8000) of pairs, sample uniformly
    note: the parameters in those 2 modes are hard-coded in this function
    '''
    assert mode in ('quantiles', 'uniform')
    instances = {}
    from numpy.random import choice 
    np.random.seed(1)

    all_pos = sorted( n_pos.values() ) 
    if verbose: 
        print all_pos

    if mode == 'quantiles': 
        quantile_1 = all_pos[len(all_pos) * 5 / 30] # x10
        quantile_2 = all_pos[len(all_pos) * 5 / 10 - 2] # x3
        quantile_3 = all_pos[len(all_pos) * 9 / 10 - 1] # x1.5
        for qid in QUERIES.keys():
            pernegative = 20 # number of limited pairs per positive sample
            num_of_instances = 8000 # number limit of pairs per query
            
            num_pos_currquery = n_pos[qid]
            curr_pernegative = pernegative
            curr_num_of_instance = num_of_instances # -- their trick: gen less pairs for queries with more pos docs
            if(num_pos_currquery <= quantile_1): 
                curr_pernegative *= 10; curr_num_of_instance *= 10
            elif(num_pos_currquery <= quantile_2): 
                curr_pernegative *= 3; curr_num_of_instance *= 3; 
            elif(num_pos_currquery <= quantile_3): 
                curr_pernegative *= 1.5; curr_num_of_instance *= 1.5; 
            
            rel_scores = defaultdict(list) # mapping a rel score to list of docids
            for docid in candidates[qid]:
                rel = relevance[(qid,docid)]
                rel_scores[rel].append(docid)
            scores = sorted( rel_scores.keys(), reverse=True ) # scores are sorted in desc order
            if verbose: 
                print 'scores =',scores, 
            total_instance = 0
            for i in xrange(len(scores)): # scores[i] = pos score
                for j in xrange(i+1, len(scores)): # scores[j] = neg score
                    total_instance += len(rel_scores[scores[i]]) * len(rel_scores[scores[j]])
            if verbose: 
                print 'total=', total_instance, 
            total_instance = min(total_instance, curr_num_of_instance)
            instances_for_q = []
            for i in xrange(len(scores)):# scores are sorted in desc order
                pos_score = scores[i]
                cur_pos_ids = rel_scores[pos_score] # mapping a rel score to list of docids
                cur_neg_ids = []
                for j in xrange(i+1, len(scores)):
                    neg_score = scores[j]
                    cur_neg_ids += rel_scores[neg_score]# FOUND A BUG HERE
                if len(cur_neg_ids)==0: break
                for posid in cur_pos_ids:
                    for negid in choice(cur_neg_ids, min(len(cur_neg_ids),int(curr_pernegative)), replace=False):
                        instances_for_q.append( (posid,negid) )
                    if len(instances_for_q)>=total_instance: break
                if len(instances_for_q)>=total_instance: break
            if verbose: 
                print 'got %d instances for query %d' % (len(instances_for_q), qid)
            instances[qid] = instances_for_q

    elif mode == 'uniform':
        n_pairs = [n_pos[qid] * (len(candidates[qid])-n_pos[qid]) for qid in candidates.keys()]
        if verbose: print n_pairs
        N_PAIRS_PER_QUERY = min(n_pairs) // 1000 * 1000 
        for qid in QUERIES.keys():
            rel_scores = defaultdict(list) # mapping a rel score to list of docids
            for docid in candidates[qid]:
                rel = relevance[(qid,docid)]
                rel_scores[rel].append(docid)
            scores = sorted( rel_scores.keys(), reverse=True ) # scores are sorted in desc order
            print 'scores =',scores, 
            all_instances = []
            for i in xrange(len(scores)): 
                pos_score = scores[i]
                for j in xrange(i+1, len(scores)): 
                    neg_score = scores[j]
                    for posid in rel_scores[pos_score]:
                        for negid in rel_scores[neg_score]: 
                            all_instances.append( (posid, negid) )
            instances_for_q = []
            for i in choice(len(all_instances), N_PAIRS_PER_QUERY, replace=False):
                instances_for_q.append(all_instances[i])
            print 'got %d instances out of %d for query %d' % (len(instances_for_q), len(all_instances), qid)
            instances[qid] = instances_for_q
    return instances 
