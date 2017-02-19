# encoding: utf-8
# common imports
import pandas as pd
import cPickle as pk
from tqdm import tqdm
import numpy as np
import os, sys, time
from collections import defaultdict, Counter

# constants
DATE_TODAY       = time.strftime('%m%d')
MAX_SEQ_LEN      = 500
MAX_NB_WORDS     = 100000 # top 20k most freq words
WD_PLACEHOLDER   = '</s>'
PARA_PLACEHOLDER = '</s>'
N_HISTBINS       = 30 # number of histograms in DRMM model
BATCH_SZ         = 64
N_EPOCH          = 10

# paths
MODEL_FPATH      = '../models/0206_model_2conv1d_2FC.h5' # path of best trained model 
MIMIC_PK_FPATH   = '../data/CNN_embedding_preprocessed.pk'
QRELS_FPATH      = '../data/qrels.txt' # training qrels file 
TOPICS_FPATH     = '../data/topics2016.xml' # query file path
PMC_PATH         = '/local/XW/DATA/TREC/PMCs/' # path of docset
DRMM_PK_FPATH    = '../data/DRMM+embedding_processed_0215.pk' # output file
LOGDIR           = '../logs/DRMM_%s' % DATE_TODAY
NOTES_DIR        = '/local/XW/DATA/MIMIC/noteevents_by_sid_hid/'
W2V_FPATH        = '/local/XW/DATA/WORD_EMBEDDINGS/biomed-w2v-200.txt'
GLOVE_FPATH      = '/local/XW/DATA/WORD_EMBEDDINGS/glove.6B.200d.txt'

# tipster dataset paths
TP_MODEL_FPATH   = '../models/0218_model_2conv1d_2FC_glove.h5'
TP_R52_PK_FPATH  = '../data/R52_CNN_embedding_preprocessed.pk'
TP_QRELS_FPATH   = '../data/tipster-qrels'
TP_TOPICS_FPATH  = '../data/tipster-topics'
TP_DOC_FPATH     = '/local/XW/DATA/tipster/zips'
TP_DRMM_PK_FPATH = '../data/DRMM_tipster_0219_NH.pk' 
