# coding: utf-8

import gensim
from gensim import corpora
import math, os, sys
from lxml import etree
from tqdm import tqdm


topic_tree = etree.parse('data/topics2016.xml')

def get_topic(i):# returns the summary string of the ith topic
    summary = topic_tree.xpath('//topic[@number="%d"]/summary/text()'%i)[0]
    return str(summary).lower()


# In[15]:

# get_topic(1)


# In[6]:

PMC_PATH = '/local/XW/DATA/TREC/PMCs/'
pmcid2fpath = {}

for subdir1 in os.listdir(PMC_PATH):
    for subdir2 in os.listdir(os.path.join(PMC_PATH, subdir1)):
        diry = os.path.join(PMC_PATH, subdir1, subdir2)
        for fn in os.listdir(diry):
            pmcid = fn[:-5]
            fpath = os.path.join(diry, fn)
            pmcid2fpath[pmcid] = fpath


# In[7]:

def get_article_abstract(pmcid):
    fpath = pmcid2fpath[pmcid]
    tree = etree.parse(fpath)
    ret = u'' + tree.xpath('string(//article-title)') + '\n'
    abstracts = tree.xpath('//abstract')
#     abstracts = tree.xpath('//p')
    ret += u' '.join( [abstract.xpath('string(.)') for abstract in abstracts] )
    if len(ret.split())<20: 
        raise Exception(u'abstraction too short: '+ pmcid + ret)
    return ret.lower()    


# In[22]:

documents = [[] for _ in xrange(31)] # documents[i] are pmcids for topic i
with open('data/qrels.txt') as f:
    for line in f: 
        topicid, _, pmcid, relevance = line.split()
        topicid = int(topicid)
        documents[topicid].append(pmcid)


# In[16]:

# print get_article_abstract('107838')


# In[17]:

def get_corpus(t): # get raw data for topic t
    corpus = []
    pmcids = []
    for pmcid in tqdm(documents[t]):
        try:
            abstract = get_article_abstract(pmcid)
            corpus.append( abstract.split() )
            pmcids.append(pmcid)
        except:
            pass
    return pmcids, corpus


# In[33]:

class BM25 :
    def __init__(self, corpus) :
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.DocLen = []
        self.buildDictionary(corpus)
        self.TFIDF_Generator(corpus)

    def buildDictionary(self, corpus) :
        self.dictionary.add_documents(corpus)

    def TFIDF_Generator(self, corpus, base=math.e) :
        docTotalLen = 0
        for doc in corpus:
            docTotalLen += len(doc)
            self.DocLen.append(len(doc))
            #print self.dictionary.doc2bow(doc)
            bow = dict([(term, freq*1.0/len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
            for term, tf in bow.items() :
                if term not in self.DF :
                    self.DF[term] = 0
                self.DF[term] += 1
            self.DocTF.append(bow)
            self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] +0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = docTotalLen / self.N

    def BM25Score(self, Query=[], k1=1.5, b=0.75) :
        query_bow = self.dictionary.doc2bow(Query)
        scores = []
        for idx, doc in enumerate(self.DocTF) :
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms :
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return scores

    def TFIDF(self) :
        tfidf = []
        for doc in self.DocTF :
            doc_tfidf  = [(term, tf*self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def Items(self) :
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        items.sort()
        return items


# In[78]:

def TREC_output(topic_id, run_name = 'bm25', fpath = None):
    ids, corp = get_corpus(topic_id)
    bm25 = BM25(corp)
    query = get_topic(topic_id).split()
    scores = bm25.BM25Score(query)
    score_id_pairs = zip(scores, ids) # list of (score, pmcid) tuples
    ranked_pairs = sorted(score_id_pairs, reverse=False)
#     print ranked_pairs[:10]
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, pmcid) in enumerate(ranked_pairs[:1000], 1):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (topic_id, pmcid, rank, score, run_name)


# In[79]:

fpath = 'data/Trec_eval/bm25.txt'
open(fpath, 'w') # clear previous results
for t in tqdm(xrange(1,31)):
    TREC_output(t, fpath=fpath)


# In[82]:

def get_topic_desc(i):# returns the summary string of the ith topic
    summary = topic_tree.xpath('//topic[@number="%d"]/description/text()'%i)[0]
    return str(summary).lower()


# In[84]:

def TREC_output_desc(topic_id, run_name = 'bm25', fpath = None):
    ids, corp = get_corpus(topic_id)
    bm25 = BM25(corp)
    query = get_topic_desc(topic_id).split()
    scores = bm25.BM25Score(query)
    score_id_pairs = zip(scores, ids) # list of (score, pmcid) tuples
    ranked_pairs = sorted(score_id_pairs, reverse=False)
#     print ranked_pairs[:10]
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, pmcid) in enumerate(ranked_pairs[:1000], 1):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (topic_id, pmcid, rank, score, run_name)


# In[85]:

fpath = 'data/Trec_eval/bm25_desc.txt'
open(fpath, 'w') # clear previous results
for t in tqdm(xrange(1,31)):
    TREC_output(t, fpath=fpath)


# In[ ]:

#  trec_eval/trec_eval -q -c qrels.txt bm25_desc.txt > bm25_desc.eval

