# coding: utf-8
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from settings import * 
np.random.seed(1) 

# ## 1. prepare Y: khot encoding (`sidhid2khot`)

# load the icd code into a dict
from collections import Counter
sidhid2icds = {} # dict[(int,int), set(str)] map (subject_id, hadm_id) ---> icd codes of this patient (set)
icd_ctr = Counter()

with open(ICD_FPATH) as f: # read icd info from the ICD file
    for line in f: 
        sid, hid, _icds = line.split(',')
        sid, hid = map( int, (sid,hid) )
        _icds = _icds.split()
        icd_ctr.update(_icds)
        sidhid2icds[(sid,hid)] = set(_icds)

from utils import to_khot
sidhid2khot, topicds = to_khot(sidhid2icds, icd_ctr, K=N_LABELS)



# ## 2. prepare X: turn notes into fix-length of word ids (`sidhid2seq`)

sidhids = []
texts = [] # text bodies
for fname in tqdm(os.listdir(NOTES_DIR)): # the data is 3.7G in size, can hold in memory...
    sid,hid = map( int, fname[:-4].split('_') )
    sidhids.append( (sid,hid) )
    fpath = os.path.join(NOTES_DIR, fname)
    df = pd.read_csv(fpath)
    texts.append( '\n=======\n\n\n'.join(df['text']) )
print('found %d texts' % len(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, # filter out numbers, otherwise lots of numbers
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'+'0123456789') 
print 'fitting on whole text corpus...',
tokenizer.fit_on_texts(texts) # this might take some time
print 'done.'

seqs = tokenizer.texts_to_sequences(texts) # turn article into seq of ids
word2idx = tokenizer.word_index # dictionary mapping words (str) ---> their index (int)

print 'found %s unique tokens, use most frequent %d of them'%(len(word2idx), MAX_NB_WORDS)

print 'padding sequences...',
seqs_padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN) # turn into fix-length sequences
print 'done.'

sidhid2seq = {}
for (sid,hid), seq in zip(sidhids,seqs_padded):
    sidhid2seq[(sid,hid)] = seq

del texts


# ## 3. Prepare embedding matrix

# build index mapping: map word to its vector
word2vec = {} # maps word ---> embedding vector
with open(W2V_FPATH) as f:
    for line in tqdm(f, total=5443657):
        vals = line.split()
        word = vals[0]
        if word in word2idx or word=='</s>':
            word2vec[word] = np.asarray(vals[1:], dtype='float')
print 'found %d word vectors.' % len(word2vec)

nb_words = min(MAX_NB_WORDS, len(word2idx))
embedding_w2v = np.zeros( (nb_words+1, EMBEDDING_DIM) ) # +1 because ids in sequences starts from 1 ?
for word,wd_id in word2idx.items(): 
    if wd_id > MAX_NB_WORDS or word not in word2vec: # there might be 0 rows in embedding matrix
        continue # word_id>MAX_NB_WORDS, this id is not in the generated sequences, discard
    embedding_w2v[wd_id,:] = word2vec[word]


glove = {} # maps word ---> embedding vector
with open(GLOVE_FPATH) as f:
    for line in tqdm(f, total=400000):
        vals = line.split()
        word = vals[0]
        if word in word2idx or word=='</s>':
            glove[word] = np.asarray(vals[1:], dtype='float')
print 'found %d word vectors.' % len(glove)

embedding_glove = np.zeros( (nb_words+1, EMBEDDING_DIM) ) # +1 because ids in sequences starts from 1 ?
for word,wd_id in word2idx.items(): 
    if wd_id > MAX_NB_WORDS or word not in glove: # there might be 0 rows in embedding matrix
        continue # word_id>MAX_NB_WORDS, this id is not in the generated sequences, discard
    embedding_glove[wd_id,:] = glove[word]


# ## 4. Split data

# Now we have all (sid,hid) pairs in the list `sidhids`, for each (sid,hid) pair, can get the sequence vector by dict `sidhid2seq`, the khot encoding by dict `sidhid2khot`, and if we want all icds (instead of the khot representation), just use the dict `sidhid2icds`. 


# split data
indices = np.arange(len(sidhids))
np.random.shuffle(indices)
validset_sz = int(VALIDATION_SPLIT*len(sidhids))
train_sidhids, val_sidhids = sidhids[:-validset_sz], sidhids[-validset_sz:]

from utils import getXY

X_train, Y_train = getXY(train_sidhids, sidhid2seq, sidhid2khot)
# print X_train.shape, Y_train.shape
X_val, Y_val = getXY(val_sidhids, sidhid2seq, sidhid2khot)
# print X_val.shape, Y_val.shape


# ## 5. Dump to pk file


description = '''This file contains the preprocessed data for note2vec training:  
* sidhids:     
    list of the 58361 unique (sid,hid) pairs
* icd_ctr: 
    Counter obj of all icds appeared in dataset 
* sidhid2icds: 
    mapping from (sid,hid) pair --> set of icd codes
* sidhid2khot: 
    mapping from (sid,hid) pair --> khot-encoding correponding to this sidhid pair
* sidhid2seq:  
    mapping from (sid,hid) pair --> fix-length sequences (len=MAX_SEQ_LEN) of word ids, this is input to embedding models
* tokenizer: 
    the tokenizer fit on corpus, it's useful when trying to feed new text paragraphs into our model
* embedding_w2v／embedding_glove: 
    matrices for the embedding layer (used as the weights parameter)
* train_sidhids/val_sidhids: 
    list of (sid,hid) pairs used as training/validation set
* X_train/Y_train/X_val/Y_val: 
    ndarray generated for training/validation

And the `to_khot` and `getXY` in utils.py might be helpful when generating new datasets.  
'''

data_to_pickle = {
    'description'     : description,

    'sidhids'         : sidhids,
    'icd_ctr'         : icd_ctr,
    'sidhid2icds'     : sidhid2icds,
    'sidhid2khot'     : sidhid2khot,
    'sidhid2seq'      : sidhid2seq,
    'tokenizer'       : tokenizer,

    'embedding_w2v'   : embedding_w2v,
    'embedding_glove' : embedding_glove,

    'train_sidhids'   : train_sidhids,
    'val_sidhids'     : val_sidhids,
    'X_train'         : X_train,
    'Y_train'         : Y_train,
    'X_val'           : X_val,
    'Y_val'           : Y_val,
}
with open(PK_FPATH, 'wb') as fout:
    pk.dump(data_to_pickle, fout, pk.HIGHEST_PROTOCOL)
print 'processed data is written into %s' % PK_FPATH

