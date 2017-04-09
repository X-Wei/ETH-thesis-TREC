# encoding: utf-8
from lxml import etree
import nltk, time, re 
from numpy.linalg import norm
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from settings import * 
np.random.seed(1) 

MODEL_FPATH = '../models/0206_model_2conv1d_2FC.h5'
out_fpath = '../data/trec-output/0310_cosine_sim_all.rankedlist'
MAX_SEQ_LEN = 1000

print '# loading objects from file'
topic_tree = etree.parse(TOPICS_FPATH)
pmcid2fpath = {} # mapping docid to its local file path 
for subdir1 in os.listdir(PMC_PATH):
    for subdir2 in os.listdir(os.path.join(PMC_PATH, subdir1)):
        diry = os.path.join(PMC_PATH, subdir1, subdir2)
        for fn in os.listdir(diry):
            pmcid = fn[:-5]
            fpath = os.path.join(diry, fn)
            pmcid2fpath[pmcid] = fpath

# load tokenizer 
with open(MIMIC_PK_FPATH, 'rb') as f:
    data_pickle = pk.load(f)
    tokenizer = data_pickle['tokenizer']
    del data_pickle 

# load model 
model = load_model(MODEL_FPATH)
get_embedvec = K.function( [model.layers[0].input, K.learning_phase()],
                           [model.layers[-4].output] )
embedvec = lambda X: get_embedvec([X,0])[0]

def text2vec(paragraph): # turn a piece of text into embedding vector
    seqs = tokenizer.texts_to_sequences([paragraph.encode('utf-8')])
    seqs_padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN)
    return embedvec(seqs_padded)

def get_query_text(i): # returns the paragraphs in topic i 
    text = '\n\n\n'.join( topic_tree.xpath('//topic[@number="%d"]/*/text()'%i) )
    return text.lower()

QUERIES = {qid:get_query_text(qid) for qid in xrange(1, 31)}

def get_article_abstract(pmcid):
    'returns article abstract'
    fpath = pmcid2fpath[pmcid]
    tree = etree.parse(fpath)
    ret = u'' + tree.xpath('string(//article-title)') + '\n'
    abstracts = tree.xpath('//abstract')
#     abstracts = tree.xpath('//p')
    ret += u' '.join( [abstract.xpath('string(.)') for abstract in abstracts] )
    if len(ret.split())<20: 
        raise Exception(u'abstraction too short: '+pmcid + ret)
    return ret.lower()

def get_cossim(qid, pmcid): 
    qvec = text2vec(QUERIES[qid])
    dvec = text2vec(get_article_abstract(pmcid))
    cossim = np.dot(dvec, qvec.T) / norm(qvec) / norm(dvec)
    return cossim[0][0]

candidates = defaultdict(list) # list of (score, pmcid) pairs
with open(QRELS_FPATH) as f:
    for line in tqdm(f, total=37707): 
        qid, _, pmcid, rel = line.split()
        qid = int(qid); rel = int(rel)
        try: 
            cossim = get_cossim(qid, pmcid)
            candidates[qid].append( (cossim, pmcid) )
        except: pass
        
open(out_fpath, 'w').close()
run_name = 'inner_prod'
for qid in xrange(1,31): 
    res = candidates[qid]
    res = sorted(res, reverse=True)
    fout = sys.stdout if out_fpath==None else open(out_fpath, 'a')
    for rank, (score, docid) in enumerate(res[:2000]):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)
print 'done, rankedlist is in %s' % out_fpath
