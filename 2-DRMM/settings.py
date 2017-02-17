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
MAX_SEQ_LEN      = 1000
MAX_NB_WORDS     = 100000 # top 20k most freq words
WD_PLACEHOLDER   = '</s>'
PARA_PLACEHOLDER = '</s>'
N_HISTBINS       = 30 # number of histograms in DRMM model
BATCH_SZ         = 64
N_EPOCH          = 10

# paths
MODEL_FPATH      = '../models/0206_model_2conv1d_2FC.h5' # path of best trained model 
MIMIC_PK_FPATH   = '../data/CNN_embedding_preprocessed.pk'
TOPICS_FPATH     = '../data/topics2016.xml' # query file path
QRELS_FPATH      = '../data/qrels.txt' # training qrels file 
DRMM_PK_FPATH    = '../data/DRMM+embedding_processed_0215.pk' #% DATE_TODAY
LOGDIR           = '../logs/DRMM_%s' % DATE_TODAY
NOTES_DIR        = '/local/XW/DATA/MIMIC/noteevents_by_sid_hid/'
W2V_FPATH        = '/local/XW/DATA/WORD_EMBEDDINGS/biomed-w2v-200.txt'
GLOVE_FPATH      = '/local/XW/DATA/WORD_EMBEDDINGS/glove.6B.200d.txt'
PMC_PATH         = '/local/XW/DATA/TREC/PMCs/' # path that stores all xml files of the docset

# tipster dataset paths
TP_QRELS_FPATH   = '/local/XW/DATA/tipster/qrels'
TP_DOC_FPATH     = '/local/XW/DATA/tipster/zips'
TP_TOPICS_FPATH  = '/local/XW/DATA/tipster/topics'
TP_DRMM_PK_FPATH = '../data/DRMM_tipster_0215.pk'