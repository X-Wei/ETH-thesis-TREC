# encoding: utf-8
from lxml import etree
import nltk, time, re 
from numpy.linalg import norm
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from settings import * 
np.random.seed(1) 
global QUERIES, MAX_QLEN, candidates, relevances, n_pos, IDFs, qid_docid2histvec, instances

def memoize_1arg(f):
    """ Memoization decorator for a function taking a SINGLE argument """
    class memoize_1arg(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memoize_1arg().__getitem__

### objects used in data preparation
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
with open(MIMIC_PK_FPATH, 'rb') as f:
    data_pickle = pk.load(f)
    tokenizer = data_pickle['tokenizer']
    del data_pickle 
model = load_model(MODEL_FPATH)
get_embedvec = K.function( [model.layers[0].input, K.learning_phase()],
                           [model.layers[-4].output] )
embedvec = lambda X: get_embedvec([X,0])[0]


### helper functions
def paragraph2vec(paragraph): # turn a piece of text into embedding vector
    seqs = tokenizer.texts_to_sequences([paragraph.encode('utf-8')])
    seqs_padded = pad_sequences(seqs, maxlen=MAX_SEQ_LEN)
    return embedvec(seqs_padded)

@memoize_1arg
def get_topic(i):# returns the summary string of the ith topic
    summary = topic_tree.xpath('//topic[@number="%d"]/summary/text()'%i)[0]
    return str(summary).lower()

@memoize_1arg
def get_query_paragraphs(i): # returns the paragraphs in topic i 
    text = '\n=====\n'.join( topic_tree.xpath('//topic[@number="%d"]/*/text()'%i) )
    pat = re.compile('\W*\n\W*\n')
    paras = pat.split(text.lower())
    return [p.strip() for p in paras]

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

@memoize_1arg
def get_article_paragraphs(pmcid):
    'returns a list of texts, each as a paragraph'
    fpath = pmcid2fpath[pmcid]
    tree = etree.parse(fpath)
    ret = []
    body = tree.xpath('//body')[0]
    for p in body.xpath('.//p'):
        ret.append( p.xpath('string(.)').strip() )
    if len(ret)>300: # some outliers have MANY <p> tags in the xml file
        ret2 = [] # in this case: merge small paragraphs into larger paragraphs!!
        mean_len = sum(len(p) for p in ret) // 300
        _buf = []; _buflen = 0
        for p in ret: 
            if _buflen >= mean_len: 
                ret2.append('\n'.join(_buf)); _buf = [p]; _buflen = len(p)
            else: _buf.append(p); _buflen += len(p)
        ret = ret2
    return ret

# @memoize_1arg # pb with memoize: the embedding vectors take too much memory...
def get_article_embeddingvecs(pmcid):
    # return np.vstack( [ paragraph2vec(p.encode('ascii','ignore')) for p in  )
    seqs_padded = [] # put all paragphs together,  feed them into embedding model at once
    for para in get_article_paragraphs(pmcid): 
        para = para.encode('ascii','ignore')
        seqs = tokenizer.texts_to_sequences([para.encode('utf-8')])
        seqs_padded.append( pad_sequences(seqs, maxlen=MAX_SEQ_LEN) )
    seqs_padded = np.vstack( seqs_padded )
    return embedvec(seqs_padded)

def get_histvec(query_para, pmcid):
    '''given a query paragraph and a docid, returns the histogram vector 
    return shape = (1 * N_HISTBINS) 
    this vector is to be fed to ffwd network of DRMM'''
    if query_para == PARA_PLACEHOLDER: 
        return np.zeros(30)
    qvec = paragraph2vec(query_para)
    dvecs = get_article_embeddingvecs(pmcid)
    cossims = np.dot(dvecs, qvec.T) / norm(qvec) / norm(dvecs, axis=1)
    hist, _ = np.histogram( cossims, bins=30, range=(0,1) )
    ret = np.log(hist+1)
    return ret 

def get_query_doc_feature(qid, pmcid): 
    'given a query id and a doc id, get the input vectors for DRMM, shape = (qlen * N_HISTBINS)'
    query = QUERIES[qid]
    return np.array([ get_histvec(p, pmcid) for p in query])


### data to be generated
QUERIES           = {} # dict[int, list<str>] mapping query id to query paragraphs, padded to same length
MAX_QLEN          = -1 # length of padded queries
candidates        = defaultdict(list) # dict[int, list<str>] mapping qid to list of its candidate docids (in the qrel)
relevance         = {} # dict[(int,str), int] mapping (qid,docid) pairs to relevance (0,1,2)
n_pos             = {} # dict[int, int] mapping qid to number of its pos-docids, useful for generating new training instances
IDFs              = {} # dict[int, array] mapping qid to its idf input vector
qid_docid2histvec = {} # dict[(int,str), array] mapping  (qid, docid) to the corresponding histvec (input to DRMM)
instances         = {} # dict[int, list<(str,str)>] mapping qid to its list of (pos_docid, neg_docid) pairs


if __name__ == '__main__':
    # global QUERIES, MAX_QLEN, candidates, relevances, n_pos, IDFs, qid_docid2histvec, instances
    ### populate the above-mentioned data
    print '# populate `QUERIES`'
    QUERIES = {qid:get_query_paragraphs(qid) for qid in xrange(1,31)}
    MAX_QLEN = max(map(len, QUERIES.values()))
    print 'max query length = %d' % MAX_QLEN

    print '# padding queries to the same length MAX_QLEN'
    def pad_query(q, SZ=MAX_QLEN): return q + [PARA_PLACEHOLDER]*(SZ-len(q))
    for i,q in QUERIES.items():
        QUERIES[i] = pad_query(q)

    print '# populate `IDFs`'
    def idf(para): return -10 if para==PARA_PLACEHOLDER else 1.0 
    for qid in QUERIES.keys(): 
        IDFs[qid] = np.array([idf(para) for para in QUERIES[qid]])
    
    print '# populate `candidates`, `relevance`, `n_pos`'
    with open(QRELS_FPATH) as f:
        for line in tqdm(f, total=37707): 
            qid, _, pmcid, rel = line.split()
            qid = int(qid); rel = int(rel)
            try: 
                if len( get_article_paragraphs(pmcid) ) <= 3: # discard too short articles
                    continue
                relevance[(qid,pmcid)] =rel
                candidates[qid].append(pmcid)
                if rel>0: n_pos[qid] += 1
            except: pass

    print '# populate `qid_docid2histvec`'
    for qid in QUERIES.keys():
        for docid in tqdm(candidates[qid]):
            _hist = get_query_doc_feature(qid, docid).reshape(1, MAX_QLEN, N_HISTBINS)
            qid_docid2histvec[(qid, docid)] = _hist
    
    print '# populate `instances`'
    from utils import gen_instances
    instances = gen_instances(QUERIES, relevance, candidates, n_pos, mode = 'quantiles')
    
    
    ### pickle data into local files
    description = '''
    The file contains pre-processed data as a dictionary, here are the keys of this dictionary: 
    * QUERIES          : dict[int, list<str>] mapping query id to query paragraphs, padded to same length
    * MAX_QLEN         : length of padded queries
    * candidates       : dict[int, list<str>] mapping qid to list of its candidate docids (that appeared in the qrel)
    * relevance        : dict[(int,str), int] mapping (qid,docid) pairs to relevance (0,1,2)
    * n_pos            : dict[int, int] mapping qid to number of its positive docids, useful for generating new training instances
    * IDFs             : dict[int, array] mapping qid to its idf input vector
    * qid_docid2histvec: dict[(int,str), array] mapping  (qid, docid) to the corresponding histvec (input to DRMM)
    * instances        : dict[int, list<(str,str)>] mapping qid to its list of (pos_docid, neg_docid) pairs
    And new training instances can be generated using `gen_instances` in `utils.py`. 
    '''
    data_to_pickle = {
        'description'      : description,
        'QUERIES'          : QUERIES,
        'MAX_QLEN'         : MAX_QLEN,
        'candidates'       : candidates,
        'relevance'        : relevance,
        'n_pos'            : n_pos,
        'IDFs'             : IDFs,
        'qid_docid2histvec': qid_docid2histvec,
        'instances'        : instances,
    }
    with open(DRMM_PK_FPATH, 'wb') as f:
        pk.dump(data_to_pickle, f, pk.HIGHEST_PROTOCOL)
    
    print 'all done, data is pickled into %s' % DRMM_PK_FPATH