from settings import * 
from DRMM import *
from utils import KFold, gen_instances

OUT_PATH = '../data/trec-output/tipster-DRMM'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# TP_DRMM_PK_FPATH = '../data/DRMM_tipster_0215.pk'
# DATE_TODAY = '0217'
print TP_DRMM_PK_FPATH

with open(TP_DRMM_PK_FPATH) as f:
    data_pickle = pk.load(f)

MAX_QLEN = data_pickle['MAX_QLEN']

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
          initial_weights = initial_weights, verbose=0, K = 5, 
          fpath = '%s/%s_5fold%s.rankedlist' % (OUT_PATH, DATE_TODAY, suffix), 
          instances = instances)
    # KFold(ranking_model=ranking_model, scoring_model=scoring_model, data_pickle=data_pickle,
    #       initial_weights = initial_weights, verbose=0, K = len(QUERIES),
    #       fpath = '%s/%s_LOO%s.rankedlist' % (OUT_PATH, DATE_TODAY, suffix), 
    #       instances = instances)


tune_ffwd(feed_forward, '_original_trecModel')
tune_ffwd(ffwd_3layer, '_ffwd3layer_trecModel')
tune_ffwd(ffwd_4layer, '_ffwd4layer_trecModel')


tune_ffwd(feed_forward, '_original_unif_trecModel', instances=instances2)
tune_ffwd(ffwd_3layer, '_ffwd3layer_unif_trecModel', instances=instances2)
tune_ffwd(ffwd_4layer, '_ffwd4layer_unif_trecModel', instances=instances2)
