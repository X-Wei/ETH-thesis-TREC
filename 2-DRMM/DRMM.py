from settings import * 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, InputLayer, Flatten, Input, Merge, merge, Reshape
import keras.backend as K
import tensorflow as tf
import pydot
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

np.random.seed(1)


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