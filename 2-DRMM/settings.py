# paths
MODEL_FPATH     = '../models/1124_model_2embed_2conv1d_2FC.h5' # path of best trained model 
TOKENIZER_FPATH = '../data/tokenizer.pk'
TOPICS_FPATH    = '../data/topics2016.xml'
PK_FPATH        = '../data/processed_data_sidhid.pk'
LOGDIR          = '../logs/DRMM_0125'
NOTES_DIR       = '/local/XW/DATA/MIMIC/noteevents_by_sid_hid/'
W2V_FPATH       = '/local/XW/DATA/WORD_EMBEDDINGS/biomed-w2v-200.txt'
GLOVE_FPATH     = '/local/XW/DATA/WORD_EMBEDDINGS/glove.6B.200d.txt'
PMC_PATH        = '/local/XW/DATA/TREC/PMCs/' # path that stores all xml files of the docset

# constants
MAX_SEQ_LEN      = 1000
MAX_NB_WORDS     = 20000 # top 20k most freq words
N_LABELS         = 50
N_SIDHID         = 58328
WD_PLACEHOLDER   = '</s>'
PARA_PLACEHOLDER = '</s>'
N_HISTBINS       = 30 # number of histograms in model
BATCH_SZ         = 64
N_EPOCH          = 10
