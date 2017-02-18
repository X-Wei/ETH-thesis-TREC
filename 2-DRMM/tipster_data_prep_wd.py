# data preparation for original DRMM (as in the paper) ran on tipster
from lxml import etree
import re
import nltk
from numpy.linalg import norm 

from settings import * 
np.random.seed(1) 

MAX_QLEN = -1
QUERIES = {}
relevance = {} 
candidates = defaultdict(list)
n_pos = defaultdict(int)
IDFs = {}
qid_docid2histvec = {}
corpus = {}

# # seems that the following can't parse all topics... 
# topic_tree = etree.parse(TP_TOPICS_FPATH, 
#             parser = etree.XMLParser(recover=True)) 
# sel = topic_tree.xpath('//top')


print '# populate QUERIES'
with open(TP_TOPICS_FPATH) as f: 
    ctt = '\n'.join(f.readlines())
tops = ['<top>'+t for t in ctt.split('<top>') if len(t)>0] 
parser = etree.XMLParser(recover=True)
for t in tops: 
    sel = etree.XML(t, parser = parser)
    qid = int( sel.xpath('//num/text()')[0].strip().split(':')[-1] )
    desc = sel.xpath('//desc/text()')[0].replace('Description:', '').strip()
    QUERIES[qid] = re.split('\W+', desc.lower())
MAX_QLEN = max( map(len, QUERIES.values()) ) 
WD_PLACEHOLDER = '</s>'                      
def pad_query(q, SZ=MAX_QLEN):
    return q + [WD_PLACEHOLDER]*(SZ-len(q))
for i,q in QUERIES.items():
    QUERIES[i] = pad_query(q)


print '# populate docid2fpath'
docid2fpath = {}
for subdir in os.listdir(TP_DOC_FPATH): 
    for fn in os.listdir(os.path.join(TP_DOC_FPATH, subdir)): 
        docid = fn
        docid2fpath[docid] = os.path.join(os.path.join(TP_DOC_FPATH, subdir, fn))

def get_article_paras(docid): 
    fpath = docid2fpath[docid]
    sel = etree.parse(fpath)
    # paras = sel.xpath('//PAR/text() | //PA1/text()')
    # return '\n'.join( [p.strip().lower() for p in paras] )
    txt = sel.xpath('//TEXT')[0]
    return txt.xpath('string(.)').lower().strip()


print '# populate relevance/candidates/n_pos/corpus '
with open(TP_QRELS_FPATH) as f:
    for line in tqdm(f): 
         qid, _, docid, rel = line.split() 
         if docid not in corpus: 
             corpus[docid] = get_article_paras(docid)
             rel = int(rel); qid = int(qid)
             if rel>0: n_pos[qid]+=1
             candidates[qid].append(docid)
             relevance[(qid,docid)] = rel


print '# calculate idf'
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus.itervalues())
vocab = vectorizer.vocabulary_ # mapping word to its internal index

def get_idf(wd):
    if wd ==WD_PLACEHOLDER: return -10.0
    return vectorizer.idf_[vocab[wd]] if wd in vocab else -1.0

for qid, q in QUERIES.iteritems(): 
    IDFs[qid] = [get_idf(wd) for wd in q ]

print '# load w2v '
from gensim.models import Word2Vec
word2vec = Word2Vec.load_word2vec_format(W2V_FPATH, binary=True)


# helper functions
def similarity(wd1, wd2):
    if wd1==wd2: return 1.0
    if wd1 in word2vec and wd2 in word2vec: 
        return word2vec.similarity(wd1,wd2)
    else: return None

def get_histvec(q_wd, doc): # get LCH feature for qwd, doc
    if q_wd == WD_PLACEHOLDER: 
        return np.zeros(30)
    doc_words = doc.split()
    hist = np.zeros(30)
    for d_wd in doc_words:
        sim = similarity(q_wd, d_wd)
        if sim is None: continue
        idx = (sim+1.0)/2.0 * (30-1) # position in the histogram
        hist[int(idx)] += 1.0 
    ret = np.log10(hist+1.0) 
    return ret 

def get_query_doc_feature(query, docid): # query: list of words
    doc = get_article_paras(docid)
    return np.array([ get_histvec(wd, doc) for wd in query])

print '# populate qid_docid2histvec'
for qid in tqdm(QUERIES.keys()):
    for docid in candidates[qid]:
        _hist = get_query_doc_feature(QUERIES[qid], docid).reshape(1,MAX_QLEN,30)
        qid_docid2histvec[(qid, docid)] = _hist

from utils import gen_instances
instances = gen_instances(QUERIES, relevance, candidates, n_pos, mode = 'quantiles')

data_to_pickle = {
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
