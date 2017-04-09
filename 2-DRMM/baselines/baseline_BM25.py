# coding: utf-8

from BM25 import BM25 
import math, os, sys, re
from lxml import etree
from tqdm import tqdm


topic_tree = etree.parse('../../data/topics2016.xml')
outfpath = '../../data/trec-output/bm25_all.rankedlist'

def get_topic(i):# returns the summary string of the ith topic
    summary = '\n\n'.join( topic_tree.xpath('//topic[@number="1"]/*/text()') )
    return str(summary).lower()


PMC_PATH = '/local/XW/DATA/TREC/PMCs/'
pmcid2fpath = {}
for subdir1 in os.listdir(PMC_PATH):
    for subdir2 in os.listdir(os.path.join(PMC_PATH, subdir1)):
        diry = os.path.join(PMC_PATH, subdir1, subdir2)
        for fn in os.listdir(diry):
            pmcid = fn[:-5]
            fpath = os.path.join(diry, fn)
            pmcid2fpath[pmcid] = fpath


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


documents = [[] for _ in xrange(31)] # documents[i] are pmcids for topic i
with open('../../data/qrels.txt') as f:
    for line in f: 
        topicid, _, pmcid, relevance = line.split()
        topicid = int(topicid)
        documents[topicid].append(pmcid)


def get_corpus(t): # get raw data for topic t
    corpus = []
    pmcids = []
    for pmcid in documents[t]:
        try:
            abstract = get_article_abstract(pmcid)
            corpus.append( re.split('\W+', abstract) )
            pmcids.append(pmcid)
        except:
            pass
    return pmcids, corpus


def TREC_output(topic_id, run_name = 'bm25', fpath = None):
    ids, corp = get_corpus(topic_id)
    bm25 = BM25(corp)
    query = get_topic(topic_id).split()
    scores = bm25.BM25Score(query)
    score_id_pairs = zip(scores, ids) # list of (score, pmcid) tuples
    ranked_pairs = sorted(score_id_pairs, reverse=1)
#     print ranked_pairs[:10]
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, pmcid) in enumerate(ranked_pairs[:1000], 1):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (topic_id, pmcid, rank, score, run_name)


open(outfpath, 'w') # clear previous results
for t in tqdm(xrange(1,31)):
    TREC_output(t, fpath=outfpath)

#  trec_eval/trec_eval -q -c qrels.txt bm25_desc.txt > bm25_desc.eval

