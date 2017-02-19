# encoding: utf-8
from lxml import etree
import nltk, time, re 
from numpy.linalg import norm
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from settings import * 
np.random.seed(1) 


print '# populate QUERIES'
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
    QUERIES[qid] = [desc, smry, narr, ccpt]
# no need to pad: all QUERIES have the same length
MAX_QLEN = max( map(len, QUERIES.values()) ) 


print '# populate `IDFs`'
IDFs = {}
def idf(para): return -10 if para==PARA_PLACEHOLDER else 1.0 
for qid in QUERIES.keys(): 
    IDFs[qid] = np.array([idf(para) for para in QUERIES[qid]])

print '# populate docid2fpath'
docid2fpath = {}
for subdir in os.listdir(TP_DOC_FPATH): 
    for fn in os.listdir(os.path.join(TP_DOC_FPATH, subdir)): 
        docid = fn
        docid2fpath[docid] = os.path.join(os.path.join(TP_DOC_FPATH, subdir, fn))

def get_article_paras(docid): # return list<str> -- paragraphs of a document
    fpath = docid2fpath[docid]
    sel = etree.parse(fpath)
    # if document has para elements, just return this paras
    paras = sel.xpath('//PAR/text() | //PA1/text() | //PAL/text()')
    if paras:
        ret = [p.strip().lower() for p in paras]
    else: 
        txt = sel.xpath('//TEXT')[0]
        txt = txt.xpath('string(.)').lower().strip()
        ret = txt.split('   ')
    if len(ret)>200: # some outliers have MANY paragraphs...
        ret2 = [] # in this case: merge small paragraphs into larger paragraphs!!
        mean_len = sum(len(p) for p in ret) // 200
        _buf = []; _buflen = 0
        for p in ret: 
            if _buflen >= mean_len: 
                ret2.append('\n'.join(_buf)); _buf = [p]; _buflen = len(p)
            else: _buf.append(p); _buflen += len(p)
        if _buf: ret2.append('\n'.join(_buf))
        ret = ret2
    return ret 


print '# load model and preprocessed data'
# get the same tokenizer during model training
with open(TP_R52_PK_FPATH, 'rb') as f:
    data_pickle = pk.load(f)
    tokenizer = data_pickle['tokenizer']
    del data_pickle 
# get model, turn it into a paragraph embedder
model = load_model(TP_MODEL_FPATH)
get_embedvec = K.function( [model.layers[0].input, K.learning_phase()],
                           [model.layers[-4].output] )
embedvec = lambda X: get_embedvec([X,0])[0]

def paragraph2vec(paragraph): # turn a piece of text into embedding vector
    seqs = tokenizer.texts_to_sequences([paragraph.encode('utf-8')])
    seqs_padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN)
    return embedvec(seqs_padded)

def get_article_embeddingvecs(docid):
    # return np.vstack( [ paragraph2vec(p.encode('ascii','ignore')) for p in  )
    seqs_padded = [] # put all paragphs together,  feed them into embedding model at once
    for para in get_article_paras(docid): 
        para = para.encode('ascii','ignore')
        seqs = tokenizer.texts_to_sequences([para.encode('utf-8')])
        seqs_padded.append( pad_sequences(seqs, maxlen=MAX_SEQ_LEN) )
    seqs_padded = np.vstack( seqs_padded )
    return embedvec(seqs_padded)

def get_histvec(query_para, docid):
    '''given a query paragraph and a docid, returns the histogram vector 
    return shape = (1 * N_HISTBINS) 
    this vector is to be fed to ffwd network of DRMM'''
    if query_para == PARA_PLACEHOLDER: 
        return np.zeros(N_HISTBINS)
    qvec = paragraph2vec(query_para)
    dvecs = get_article_embeddingvecs(docid)
    cossims = np.dot(dvecs, qvec.T) / norm(qvec) / norm(dvecs, axis=1)
    hist, _ = np.histogram( cossims, bins=N_HISTBINS, range=(-1,1) )
    hist = hist * 1.0 / len(cossims) # NORMALIZE the histogram as there are a lot documents with few paras
    # ret = np.log(hist+1.0) 
    ret = hist 
    return ret 

def get_query_doc_feature(qid, docid): 
    'given a query id and a doc id, get the input vectors for DRMM, shape = (qlen * N_HISTBINS)'
    query_paras = QUERIES[qid]
    return np.array([ get_histvec(para, docid) for para in query_paras])


print '# populate `candidates`, `relevance`, `n_pos`'
relevance  = {}
candidates = defaultdict(list)
n_pos      = defaultdict(int)
with open(TP_QRELS_FPATH) as f:
    for line in tqdm(f): 
        qid, _, docid, rel = line.split()
        qid = int(qid); rel = int(rel)
        # # Pb: a lot (1/3) of short length (<3) articles ? 
        # if len( get_article_paras(docid) ) <= 3: # discard too short articles
        #     # print docid
        #     continue
        relevance[(qid,docid)] =rel
        candidates[qid].append(docid)
        if rel>0: n_pos[qid] += 1


print '# populate `qid_docid2histvec`'
qid_docid2histvec = {}
for qid in QUERIES.keys():
    for docid in tqdm(candidates[qid]):
        _hist = get_query_doc_feature(qid, docid).reshape(1, MAX_QLEN, N_HISTBINS)
        qid_docid2histvec[(qid, docid)] = _hist


print '# populate `instances`'
from utils import gen_instances
instances = gen_instances(QUERIES, relevance, candidates, n_pos, mode = 'quantiles')


data_to_pickle = {
    'embedding_model'  : TP_MODEL_FPATH,
    'QUERIES'          : QUERIES,
    'MAX_QLEN'         : MAX_QLEN,
    'candidates'       : candidates,
    'relevance'        : relevance,
    'n_pos'            : n_pos,
    'IDFs'             : IDFs,
    'qid_docid2histvec': qid_docid2histvec,
    'instances'        : instances,
}

with open(TP_DRMM_PK_FPATH, 'wb') as f:
    pk.dump(data_to_pickle, f, pk.HIGHEST_PROTOCOL)
