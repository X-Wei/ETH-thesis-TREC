from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from lxml import etree

from settings import * 
np.random.seed(1) 

corpus = {} # dict[str, str] mapping a name (`folder-filename`) to the content
name2khot = {} # dict[str, np.array] mapping a name (`folder-filename`) to a khot encoding np array

all_labels = [] # list of all unique topic labels (there are 126 lables)
with open(os.path.join(RCV1_DIR, 'codes', 'topic_codes.txt')) as f:
    for line in f.readlines()[2:]:
        if line.split():
            all_labels.append(line.split()[0])
N_LABELS = len(all_labels) # =126
label2id = {} # mapping label(string) to an idx
for i, label in enumerate( all_labels ):
    label2id[label] = i 

# read from data dir 
for subdir in tqdm(os.listdir(RCV1_DIR)): 
    if not subdir.isdigit(): continue
    for fn in os.listdir(os.path.join(RCV1_DIR, subdir)):
        name = '%s-%s' % (subdir, fn)
        sel = etree.parse(os.path.join(RCV1_DIR, subdir, fn))
        topics = sel.xpath('//codes[@class="bip:topics:1.0"]')
        if topics: 
            topics = topics[0]
            topics = topics.xpath('.//code/@code')
            _khot = np.zeros(N_LABELS)
            for t in topics: 
                _khot[label2id[t]] = 1
            name2khot[name] = _khot 
            corpus[name] = sel.xpath('//text')[0].xpath('string(.)').encode('utf-8')


# tokenize texts
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
texts = corpus.itervalues() # generator of strings (text contents) 
tokenizer.fit_on_texts(texts)
word2idx = tokenizer.word_index # dictionary mapping words (str) ---> their index (int)


# split validation set 
names = corpus.keys()
np.random.shuffle(names)
validset_sz = int(VALIDATION_SPLIT*len(names))
train_names, val_names = names[:-validset_sz], names[-validset_sz:] 

def getXY(names):
    texts = [corpus[name] for name in names]
    seqs = tokenizer.texts_to_sequences(texts) # turn article into seq of ids
    seqs_padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN) # turn into fix-length sequences
    X = np.array(seqs_padded)
    Y = np.array([name2khot[name] for name in names])
    return X, Y

X_train, Y_train = getXY(train_names)
X_val, Y_val     = getXY(val_names)


# prepare embeddings 
nb_words = min(MAX_NB_WORDS, len(word2idx))
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

# output
data_to_pickle = {
    'tokenizer'       : tokenizer,
    'embedding_glove' : embedding_glove,
    'X_train'         : X_train,
    'Y_train'         : Y_train,
    'X_val'           : X_val,
    'Y_val'           : Y_val,
}

with open(RCV1_PK_FPATH, 'wb') as fout:
    pk.dump(data_to_pickle, fout, pk.HIGHEST_PROTOCOL)
print 'processed data is written into %s' % RCV1_PK_FPATH