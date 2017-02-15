# coding: utf-8

from settings import * 
from DRMM import *
from utils import KFold, gen_instances

with open(DRMM_PK_FPATH) as f:
    data_pickle = pk.load(f)

data_pickle.keys()


MAX_QLEN = 12

QUERIES           = data_pickle['QUERIES']
instances         = data_pickle['instances']
candidates        = data_pickle['candidates']
qid_docid2histvec = data_pickle['qid_docid2histvec']
relevance         = data_pickle['relevance']
n_pos             = data_pickle['n_pos']
IDFs              = data_pickle['IDFs']

instances2 = gen_instances(QUERIES, relevance, candidates, n_pos, mode = 'uniform')



ffwd_3layer = Sequential(
    [Dense(input_dim= N_HISTBINS, output_dim=10, activation='relu'),
     Dense(input_dim= N_HISTBINS, output_dim=5, activation='relu'),
     Dense(output_dim=1, activation='tanh'),
     ], 
    name='ffwd_3layer')


ffwd_4layer = Sequential(
    [Dense(input_dim= N_HISTBINS, output_dim=10, activation='relu'),
     Dense(input_dim= N_HISTBINS, output_dim=6, activation='relu'),
     Dense(input_dim= N_HISTBINS, output_dim=3, activation='relu'),
     Dense(output_dim=1, activation='tanh'),
     ], 
    name='ffwd_4layer')



def tune_ffwd(feed_forward, suffix = '', instances = instances):
    scoring_model, ranking_model = gen_DRMM_model(MAX_QLEN, feed_forward)
    initial_weights = ranking_model.get_weights()
    KFold(ranking_model=ranking_model, scoring_model=scoring_model, data_pickle=data_pickle,
          initial_weights = initial_weights, verbose=0,
          fpath = '../data/trec-output/0211_50bins_5fold_%s.rankedlist' % suffix, 
          instances = instances)
    KFold(ranking_model=ranking_model, scoring_model=scoring_model, data_pickle=data_pickle,
          initial_weights = initial_weights, verbose=0, K = 30,
          fpath = '../data/trec-output/0211_50bins_LOO_%s.rankedlist' % suffix, 
          instances = instances)


tune_ffwd(feed_forward, 'orignial')
tune_ffwd(ffwd_3layer, 'ffwd3layer')
tune_ffwd(ffwd_4layer, 'ffwd4layer')


tune_ffwd(feed_forward, 'orignial_unif', instances=instances2)
tune_ffwd(ffwd_3layer, 'ffwd3layer_unif', instances=instances2)
tune_ffwd(ffwd_4layer, 'ffwd4layer_unif', instances=instances2)
