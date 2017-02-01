# constants
N_LABELS = 50 # predict only on top K frequent icd codes
N_SIDHID = 58328  # number of unique (sid,hid) pairs
MAX_SEQ_LEN = 1000 # max length of input sequence (pad/truncate to fixed length)
MAX_NB_WORDS = 100000 # top 100k most freq words
EMBEDDING_DIM = 200

# for data preparation
NOTES_DIR = '/local/XW/DATA/MIMIC/noteevents_by_sid_hid/'
ICD_FPATH = '../data/sid_hid_diagicds.txt'
PK_FPATH = '../data/CNN_embedding_preprocessed.pk'

# word2vec configurations
W2V_FPATH = '/local/XW/DATA/WORD_EMBEDDINGS/biomed-w2v-200.txt'
GLOVE_FPATH = '/local/XW/DATA/WORD_EMBEDDINGS/glove.6B.200d.txt'


# for model training
MODEL_PATH = '../models/'
LOG_PATH = '../logs/'
# learning configurations
VALIDATION_SPLIT = 0.2
N_EPOCHS = 20
BATCH_SZ = 512
