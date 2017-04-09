# encoding: utf-8
from lxml import etree
import nltk, time, re 
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from settings import * 
np.random.seed(1) 

vectorizer = TfidfVectorizer(max_features=MAX_NB_WORDS)

out_fpath = '../data/trec-output/0306_simple_cossim_tfidf.rankedlist'

print '# loading objects from file',
topic_tree = etree.parse(TOPICS_FPATH)
pmcid2fpath = {} # mapping docid to its local file path 
for subdir1 in os.listdir(PMC_PATH):
    for subdir2 in os.listdir(os.path.join(PMC_PATH, subdir1)):
        diry = os.path.join(PMC_PATH, subdir1, subdir2)
        for fn in os.listdir(diry):
            pmcid = fn[:-5]
            fpath = os.path.join(diry, fn)
            pmcid2fpath[pmcid] = fpath
print 'done'


def get_article_abstract(pmcid):
    'returns article abstract'
    fpath = pmcid2fpath[pmcid]
    tree = etree.parse(fpath)
    ret = u'' + tree.xpath('string(//article-title)') + '\n'
    abstracts = tree.xpath('//abstract')
#     abstracts = tree.xpath('//p')
    ret += u' '.join( [abstract.xpath('string(.)') for abstract in abstracts] )
    if len(ret.split())<20: 
        # raise Exception(u'abstraction too short: '+pmcid + ret)
        # print 'abstraction too short: '+pmcid + ret
        return None 
    return ret.lower()


def get_query_text(i): # returns the paragraphs in topic i 
    text = '\n\n\n'.join( topic_tree.xpath('//topic[@number="%d"]/*/text()'%i) )
    return text.lower()

QUERIES = {qid:get_query_text(qid) for qid in xrange(1, 31)}


# populate corpus
corpus = [] # list of strings 
corpus.extend(QUERIES.values())
all_candidate_pmcids = set()

with open(QRELS_FPATH) as f:
    for line in f: 
        qid, _, pmcid, rel = line.split()
        all_candidate_pmcids.add(pmcid) 

for pmcid in tqdm(all_candidate_pmcids): 
    abstract = get_article_abstract(pmcid)
    if abstract: 
        corpus.append(abstract)

print '# fitting on corpus', 
vectorizer.fit(corpus)
del corpus 
print 'done'


def text2vec(paragraph): # turn a piece of text into embedding vector
    return vectorizer.transform([paragraph]).toarray()[0]

def get_cossim(qid, pmcid): 
    qvec = text2vec(QUERIES[qid])
    dvec = text2vec(get_article_abstract(pmcid))
    cossim = np.dot(dvec, qvec.T) / norm(qvec) / norm(dvec)
    return cossim

### calculate all scores 

candidates = defaultdict(list) # list of (score, pmcid) pairs
with open(QRELS_FPATH) as f:
    for line in tqdm(f, total=37707): 
        qid, _, pmcid, rel = line.split()
        qid = int(qid); rel = int(rel)
        try: 
            cossim = get_cossim(qid, pmcid)
            candidates[qid].append( (cossim, pmcid) )
        except: pass


## output to rankedlist 
open(out_fpath, 'w').close()
run_name = 'inner_prod_tfidf'
for qid in xrange(1,31): 
    res = candidates[qid]
    res = sorted(res, reverse=True)
    fout = sys.stdout if out_fpath==None else open(out_fpath, 'a')
    for rank, (score, docid) in enumerate(res[:2000]):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)
print 'done, rankedlist is in %s' % out_fpath