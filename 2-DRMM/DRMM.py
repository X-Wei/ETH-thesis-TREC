from tqdm import tqdm 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, InputLayer, Flatten, Input, Merge, merge, Reshape
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
import numpy as np 
import pydot
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

np.random.seed(1)
N_HISTBINS = 30 
BATCH_SZ = 64
N_EPOCH = 10
LOGDIR = './logs/DRMM_0125'


# helper function: model visualization 
def viz_model(model):
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))

#### component 1: feed-forward network 
feed_forward = Sequential(
    [Dense(input_dim= N_HISTBINS, output_dim=5, activation='tanh'),
     Dense(output_dim=1, activation='tanh'),
     ], # 30-5-1 as described in the paper
    name='feed_forward_nw')


#### component 2: gating network
from keras.engine.topology import Layer
class ScaledLayer(Layer): # a scaled layer
    def __init__(self, **kwargs):
        super(ScaledLayer
    , self).__init__(**kwargs)
    def build(self, input_shape):
        self.output_dim = input_shape[1] 
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer='one', trainable=True)
        super(ScaledLayer
    , self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        return tf.mul(x, self.W)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

input_idf = Input(shape=(None,), name='input_idf')
scaled = ScaledLayer()(input_idf)
gs = Activation('softmax', name='softmax')(scaled)
gating = Model(input=input_idf, output=gs, name='gating')

#### some helper functions to build the DRMM 
from keras.layers.core import Lambda
# take the ith slice of the input x
def slicei(x, i): return x[:,i,:]
def slicei_output_shape(input_shape): return (input_shape[0], input_shape[2])

# concatenate inputs into one tensor 
def concat(x): return K.concatenate(x) 
def concat_output_shape(input_shape): return (input_shape[0][0], input_shape[0][1])

# innerproduct of zs and gs
def innerprod(x): return K.sum( tf.mul(x[0],x[1]), axis=1)
def innerprod_output_shape(input_shape): return (input_shape[0][0],1)

# get the difference of input 
def diff(x): return tf.sub(x[0], x[1]) 
def diff_output_shape(input_shape): return input_shape[0]

# custom loss function: hinge of (score_pos - score_neg)
def pairwise_hinge(y_true, y_pred): # y_pred = score_pos - score_neg, **y_true doesn't matter here**
    return K.mean( K.maximum(0.1 - y_pred, y_true*0.0) )  

# self-defined metrics
def ranking_acc(y_true, y_pred):
    y_pred = y_pred > 0 
    return K.mean(y_pred)

##### the function to give a DRMM model with custom query length 
def gen_DRMM_model(QLEN, feed_forward = feed_forward): 
    '''
    generates a DRMM model for query length = `QLEN`
    returns 2 items: `scoring_model, ranking_model`
    * `scoring_model` is for predicting;
    * `ranking_model` is for training, and ranking_model has another field: `initial_weights`, which is used in `KFold`
    '''
    # input 1: idf 
    input_idf = Input(shape=(QLEN,), name='input_idf')

    # `gs` = gating output 
    # scaled = ScaledLayer(input_idf)
    # gs = Activation('softmax', name='softmax')(scaled)
    # gating = Model(input=input_idf, output=gs, name='gating')
    gs = gating(input_idf)

    # input 2: hist vectors, shape = QLEN x  N_HISTBINS
    input_hists = Input(shape=(QLEN, N_HISTBINS), name='input_hists')
    
    # `zs` = feed_forward network output  
    zs = [ feed_forward( Lambda(lambda x:slicei(x,i), slicei_output_shape, name='slice%d'%i)(input_hists) )\
            for i in xrange(QLEN) ] 
    zs = Lambda(concat, concat_output_shape, name='concat_zs')(zs)

    # score = inner product of zs and gs 
    scores = Lambda(innerprod, innerprod_output_shape, name='innerprod_zs_gs')([zs, gs])

    # scoring model 
    scoring_model = Model(input=[input_idf, input_hists], output=[scores], name='scoring_model')

    # input3 -- the negative hists vector 
    input_hists_neg = Input(shape=(QLEN, N_HISTBINS), name='input_hists_neg')
    zs_neg = [ feed_forward( Lambda(lambda x:slicei(x,i), slicei_output_shape, name='slice%d_neg'%i)(input_hists_neg) )\
          for i in xrange(QLEN) ]
    zs_neg = Lambda(concat, concat_output_shape, name='concat_zs_neg')(zs_neg)
    scores_neg = Lambda(innerprod, innerprod_output_shape, name='innerprod_zs_gs_neg')([zs_neg, gs])

    ''' **NOTE** 
    for negative document's score, can't just do (don't know why...): 
        `scores_neg = scoring_model([input_idf, input_hists_neg])`
    if we do like above, both outputs from `two_score_model` will be 0.0 
    '''

    two_score_model = Model(input=[input_idf, input_hists, input_hists_neg], 
                            output=[scores, scores_neg], name='two_score_model')
    # positive score - negative score 
    posneg_score_diff = Lambda(diff, diff_output_shape, name='posneg_score_diff')([scores, scores_neg])
    ranking_model = Model(input=[input_idf, input_hists, input_hists_neg], 
                            output=[posneg_score_diff], name='ranking_model')
    
    ranking_model.compile(optimizer='adagrad', loss=pairwise_hinge, metrics=[ranking_acc])
    ranking_model.initial_weights = ranking_model.get_weights()
    return scoring_model, ranking_model

# generator for model training
def batch_generator(idx_pairs, batch_size=BATCH_SZ): 
    # ** parameter `idx_pairs` is list of tuple (qid, pos_docid, neg_docid)**
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
            pos_batch.append(qid_docid2histvec[(qid,pos_docid)].reshape(QLEN,30))
            neg_batch.append(qid_docid2histvec[(qid,neg_docid)].reshape(QLEN,30))
        idfs_batch, pos_batch, neg_batch = map(np.array, [idfs_batch, pos_batch, neg_batch])
#         print idfs_batch.shape, pos_batch.shape, neg_batch.shape
        counter += 1
        if (counter >= batches_pre_epoch):
            np.random.shuffle(idx_pairs)
            counter=0
        yield [idfs_batch, pos_batch, neg_batch], y_true_batch_dummy

def get_idx_pairs(qids, instances):
    idx_pairs = []
    for qid in qids:
        for posid, negid in instances[qid]:
            idx_pairs.append( (qid,posid, negid) )
    return idx_pairs

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def TREC_output(scoring_model, qid, run_name = 'my_run', fpath = None):
    res = [] # list of (score, pmcid) tuples
    for docid in candidates[qid]:
        input_idf = IDFs[qid].reshape((-1,QLEN))
        input_hist = qid_docid2histvec[(qid,docid)]
        score = scoring_model.predict([input_idf, input_hist])[0]
        res.append( (score, docid) )
    res = sorted(res, reverse=True)
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, docid) in enumerate(res[:2000]):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)

_callbacks = [ EarlyStopping(monitor='val_loss', patience=2),
               TensorBoard(log_dir=LOGDIR, histogram_freq=0, write_graph=False) ]

def KFold(qids, fpath, ranking_model, scoring_model, instances, K = 5, run_name = 'my_run',  batch_size=BATCH_SZ):
    open(fpath,'w').close() # clear previous content in file 
    np.random.seed(0)
    np.random.shuffle(qids)
    fold_sz = len(qids) / K
    for fold in xrange(K):
        print 'fold %d' % fold, 
        val_start, val_end = fold*fold_sz, (fold+1)*fold_sz
        qids_val = qids[val_start:val_end] # train/val queries for each fold 
        qids_train = qids[:val_start] + qids[val_end:]
        print qids_val
        idx_pairs_train = get_idx_pairs(qids_train,instances)
        idx_pairs_val = get_idx_pairs(qids_val,instances)
        
        shuffle_weights(ranking_model, ranking_model.initial_weights) # reset model parameters
        ranking_model.fit_generator( batch_generator(idx_pairs_train, batch_size=batch_size), # train model 
                    samples_per_epoch = len(idx_pairs_train)//batch_size*batch_size,
                    nb_epoch = N_EPOCH,
                    validation_data=batch_generator(idx_pairs_val, batch_size=batch_size),
                    nb_val_samples=len(idx_pairs_val)//batch_size*batch_size, 
                    callbacks = _callbacks)
        print 'fold %d complete, outputting to %s...' % (fold, fpath)
        for qid in qids_val:
            TREC_output(scoring_model, qid, run_name = run_name, fpath = fpath)