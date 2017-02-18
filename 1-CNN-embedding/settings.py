# coding: utf-8
# common imports 
import os, sys, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import cPickle as pk

# constants
N_LABELS      = 50 # predict only on top K frequent icd codes
N_SIDHID      = 58328  # number of unique (sid,hid) pairs
MAX_SEQ_LEN   = 500 # max length of input sequence (pad/truncate to fixed length)
MAX_NB_WORDS  = 100000 # top 100k most freq words
EMBEDDING_DIM = 200
DATE_TODAY    = time.strftime('%m%d')

# for data preparation
NOTES_DIR   = '/local/XW/DATA/MIMIC/noteevents_by_sid_hid/'
ICD_FPATH   = '../data/sid_hid_diagicds.txt'
PK_FPATH    = '../data/CNN_embedding_preprocessed.pk'
W2V_FPATH   = '/local/XW/DATA/WORD_EMBEDDINGS/biomed-w2v-200.txt'
GLOVE_FPATH = '/local/XW/DATA/WORD_EMBEDDINGS/glove.6B.200d.txt'
# for tipster dataset (train using R52 dataset)
R52_FPATH_TRAIN = '/local/XW/DATA/phd-datasets/r52-train-all-terms.txt'
R52_FPATH_TEST  = '/local/XW/DATA/phd-datasets/r52-test-all-terms.txt'
R52_PK_FPATH    = '../data/R52_CNN_embedding_preprocessed.pk'

# for model training
MODEL_PATH       = '../models/'
LOG_PATH         = '../logs/'
VALIDATION_SPLIT = 0.2 # learning configurations
N_EPOCHS         = 20
BATCH_SZ         = 512
