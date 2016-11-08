
# coding: utf-8

import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import cPickle as pk
np.random.seed(1) # to be reproductive

#~ beg, end = map(int, sys.argv[-2:])
#~ print beg, end

NOTE_DATA_DIR = '/local/XW/DATA/MIMIC/noteevents_by_sid/'
QUICKUMLS_DATA_DIR = '/local/XW/SOFT/UMLS/quickUMLS_data/'
UMLS_DATA_DIR = '/local/XW/DATA/MIMIC/UMLS_by_sid/'


from QuickUMLS.quickumls import QuickUMLS
print 'initializing matcher...'
matcher = QuickUMLS(QUICKUMLS_DATA_DIR)


'''
dump quickUMLS results into files in folder UMLS_DATA_DIR, each subject id has a file `sid.pk`

each .pk file contains a list, each element in list corresponds to a noteevent

each noteevent is represented as a list of lists of dicts (one list per sentence?), 
the concepts are stored in the dicts
'''
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (2048,4096))# IOError: [Errno 24] Too many open files

print 'start extracting concepts...'
for fname in tqdm(os.listdir(NOTE_DATA_DIR)[4931+645+6831+4811+790+90:]):
    fpath = os.path.join(NOTE_DATA_DIR, fname)
    df = pd.read_csv(fpath)
    umls = []
    for text in df['text']:
        res = matcher.match(text, best_match=True, ignore_syntax=False)
        umls.append(res)
    fout_name = fname[:-4]+'.pk' # output name: sid.pk
    fout_path = os.path.join(UMLS_DATA_DIR, fout_name)
    with open(fout_path, 'wb') as f:
        pk.dump(umls, f, pk.HIGHEST_PROTOCOL)
        
print('done!')

