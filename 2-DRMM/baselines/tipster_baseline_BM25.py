# coding: utf-8

import gensim
from gensim import corpora
import math, os, sys, re 
from lxml import etree
from tqdm import tqdm

# get queries
TP_TOPICS_FPATH  = '../../data/tipster-topics'
QUERIES = {}
with open(TP_TOPICS_FPATH) as f: 
    ctt = '\n'.join(f.readlines())
tops = ['<top>'+t for t in ctt.split('<top>') if len(t)>0] 
parser = etree.XMLParser(recover=True)
for t in tops: 
    sel = etree.XML(t, parser = parser)
    qid = int( sel.xpath('//num/text()')[0].strip().split(':')[-1] )
    desc = sel.xpath('//desc/text()')[0].replace('Description:', '').strip()
    QUERIES[qid] = re.split('\W+', desc.lower())


# get corpus 
TP_DOC_FPATH     = '/local/XW/DATA/tipster/zips'
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


TP_QRELS_FPATH   = '../../data/tipster-qrels'
from collections import defaultdict
candidates = defaultdict(list)
with open(TP_QRELS_FPATH) as f:
    for line in f: 
        topicid, _, docid, relevance = line.split()
        topicid = int(topicid)
        candidates[topicid].append(docid)


def get_corpus(qid): # get raw data for topic qid, returns list[list[str]]
    corpus = []
    docids = []
    for docid in tqdm(candidates[qid]):
        txt = get_article_paras(docid)
        corpus.append( re.split('\W+', txt)  )
        docids.append(docid)
    return docids, corpus


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


def TREC_output(qid, run_name = 'bm25', fpath = None):
    ids, corp = get_corpus(qid)
    bm25 = BM25(corp)
    query = QUERIES[qid]
    scores = bm25.BM25Score(query)
    score_id_pairs = zip(scores, ids) # list of (score, docid) tuples
    ranked_pairs = sorted(score_id_pairs, reverse=False)
#     print ranked_pairs[:10]
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, docid) in enumerate(ranked_pairs[:2000], 1):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)



fpath = 'bm25.txt'
open(fpath, 'w') # clear previous results
for qid in tqdm(QUERIES.keys()):
    TREC_output(qid, fpath=fpath)


#  trec_eval/trec_eval -q -c qrels.txt bm25_desc.txt > bm25_desc.eval

