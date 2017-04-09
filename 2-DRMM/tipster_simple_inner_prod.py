# encoding: utf-8
from lxml import etree
import nltk, time, re 
from numpy.linalg import norm
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from settings import * 
np.random.seed(1) 

MAX_SEQ_LEN = 500


# load tokenizer 
with open(TP_RCV1_PK_FPATH, 'rb') as f:
    data_pickle = pk.load(f)
    tokenizer = data_pickle['tokenizer']
    del data_pickle 

# load model 
model = load_model(TP_MODEL_FPATH)
get_embedvec = K.function( [model.layers[0].input, K.learning_phase()],
                           [model.layers[-4].output] )
embedvec = lambda X: get_embedvec([X,0])[0]

def text2vec(paragraph): # turn a piece of text into embedding vector
    seqs = tokenizer.texts_to_sequences([paragraph.encode('utf-8')])
    seqs_padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN)
    return embedvec(seqs_padded)

QUERIES = {} # dict[int, list<str>] mapping query id to query paras
with open(TP_TOPICS_FPATH) as f: 
    ctt = '\n'.join(f.readlines())
tops = ['<top>'+t for t in ctt.split('<top>') if len(t)>0] 
parser = etree.XMLParser(recover=True)
for t in tops: 
    sel = etree.XML(t, parser = parser)
    qid = int( sel.xpath('//num/text()')[0].strip().split(':')[-1] )
    desc = sel.xpath('//desc/text()')[0].replace('Description:', '').strip()
    smry = sel.xpath('//smry/text()')[0].replace('Summary:', '').strip()
    narr = sel.xpath('//narr/text()')[0].replace('Narrative:', '').strip()
    ccpt = sel.xpath('//con/text()')[0].replace('Concept(s):', '').strip()
    QUERIES[qid] = '\n\n\n'.join( [desc, smry, narr, ccpt] )

print '# populate docid2fpath'
docid2fpath = {}
for subdir in os.listdir(TP_DOC_FPATH): 
    for fn in os.listdir(os.path.join(TP_DOC_FPATH, subdir)): 
        docid = fn
        docid2fpath[docid] = os.path.join(os.path.join(TP_DOC_FPATH, subdir, fn))
    
def get_article_text(docid):
    fpath = docid2fpath[docid]
    sel = etree.parse(fpath)
    txt = sel.xpath('//TEXT')[0]
    ret = txt.xpath('string(.)').lower().strip()
    return ret

def get_cossim(qid, docid): 
    qvec = text2vec(QUERIES[qid])
    dvec = text2vec(get_article_text(docid))
    cossim = np.dot(dvec, qvec.T) / norm(qvec) / norm(dvec)
    return cossim[0][0]

candidates = defaultdict(list) # list of (score, docid) pairs
with open(TP_QRELS_FPATH) as f:
    for line in tqdm(f, total=70397): 
        qid, _, docid, rel = line.split()
        qid = int(qid); rel = int(rel)
        try: 
            cossim = get_cossim(qid, docid)
            candidates[qid].append( (cossim, docid) )
        except: pass
        
fpath = '../data/trec-output/0227_simple_cossim_tp.rankedlist'
open(fpath, 'w').close()
run_name = 'inner_prod'
for qid in QUERIES.keys(): 
    res = candidates[qid]
    res = sorted(res, reverse=True)
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, docid) in enumerate(res[:2000]):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)
