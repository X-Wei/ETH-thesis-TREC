from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from settings import * 
np.random.seed(1) 

training_set, testing_set = [], [] # list of (text, label) pairs

with open(R52_FPATH_TRAIN) as f:
    for line in f: 
        label, text = line.split('\t', 1)
        training_set.append( (label, text) )

with open(R52_FPATH_TEST) as f:
    for line in f: 
        label, text = line.split('\t', 1)
        testing_set.append( (label, text) )

all_labels = set(zip(*training_set)[0]) # 52 unique labels
N_LABELS = len(all_labels)
label2id = {} # mapping label(string) to an idx
for i, label in enumerate( sorted(all_labels) ):
    label2id[label] = i 


# tokenize texts, pad to sequences of ids
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
texts = zip(*training_set)[1]
tokenizer.fit_on_texts(texts)
word2idx = tokenizer.word_index # dictionary mapping words (str) ---> their index (int)

def getXY(dataset):
    labels, texts = zip(*dataset)
    seqs = tokenizer.texts_to_sequences(texts) # turn article into seq of ids
    seqs_padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN) # turn into fix-length sequences
    X = np.array(seqs_padded)
    Y = []
    for label in labels: 
        _onehot = np.zeros(N_LABELS)
        k = label2id[label]
        _onehot[k] = 1
        Y.append(_onehot)
    Y = np.array(Y)
    return X, Y
X_train, Y_train = getXY(training_set)
X_val, Y_val     = getXY(testing_set)


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

with open(R52_PK_FPATH, 'wb') as fout:
    pk.dump(data_to_pickle, fout, pk.HIGHEST_PROTOCOL)
print 'processed data is written into %s' % R52_PK_FPATH