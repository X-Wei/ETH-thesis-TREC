# encoding: utf-8
from lxml import etree
import nltk, time, re 
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from settings import * 
np.random.seed(1) 

vectorizer = TfidfVectorizer(max_features=MAX_NB_WORDS)

out_fpath = '../data/trec-output/0306_simple_cossim_tfidf_tp.rankedlist'


QUERIES = {} # dict[int, list<str>] mapping query id to query paras
with open(TP_TOPICS_FPATH) as f: 
    ctt = '\n'.join(f.readlines())
tops = ['<top>'+t for t in ctt.split('<top>') if len(t)>0] 
parser = etree.XMLParser(recover=True)
for t in tops: 
    sel = etree.XML(t, parser = parser)
    qid = int( sel.xpath('//num/text()')[0].strip().split(':')[-1] )
    desc = sel.xpath('//desc/text()')[0].replace('Description:', '').strip()
    smry = ''# sel.xpath('//smry/text()')[0].replace('Summary:', '').strip()
    narr = ''# sel.xpath('//narr/text()')[0].replace('Narrative:', '').strip()
    ccpt = ''# sel.xpath('//con/text()')[0].replace('Concept(s):', '').strip()
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


# populate corpus
corpus = [] # list of strings 
corpus.extend(QUERIES.values())
all_candidate_docids = set()
with open(TP_QRELS_FPATH) as f:
    for line in f: 
        qid, _, docid, rel = line.split()
        all_candidate_docids.add(docid) 

for docid in tqdm(all_candidate_docids): 
    abstract = get_article_text(docid)
    if abstract: 
        corpus.append(abstract)

print '# fitting on corpus', 
vectorizer.fit(corpus)
del corpus 
print 'done'



def text2vec(paragraph): # turn a piece of text into embedding vector
    return vectorizer.transform([paragraph]).toarray()[0]

def get_cossim(qid, docid): 
    qvec = text2vec(QUERIES[qid])
    dvec = text2vec(get_article_text(docid))
    cossim = np.dot(dvec, qvec.T) / norm(qvec) / norm(dvec)
    return cossim

candidates = defaultdict(list) # list of (score, docid) pairs
with open(TP_QRELS_FPATH) as f:
    for line in tqdm(f, total=70397): 
        qid, _, docid, rel = line.split()
        qid = int(qid); rel = int(rel)
        try: 
            cossim = get_cossim(qid, docid)
            candidates[qid].append( (cossim, docid) )
        except: pass
        
open(out_fpath, 'w').close()
run_name = 'inner_prod'
for qid in QUERIES.keys(): 
    res = candidates[qid]
    res = sorted(res, reverse=True)
    fout = sys.stdout if out_fpath==None else open(out_fpath, 'a')
    for rank, (score, docid) in enumerate(res[:2000]):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)
print 'done, rankedlist is in %s' % out_fpath
