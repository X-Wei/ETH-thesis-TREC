# coding: utf-8

from BM25 import BM25 
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
    smry = sel.xpath('//smry/text()')[0].replace('Summary:', '').strip()
    narr = sel.xpath('//narr/text()')[0].replace('Narrative:', '').strip()
    ccpt = sel.xpath('//con/text()')[0].replace('Concept(s):', '').strip()
    query = '\n\n\n'.join( [desc, smry, narr, ccpt] )
    QUERIES[qid] = re.split('\W+', query.lower())


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
    for docid in candidates[qid]:
        txt = get_article_paras(docid)
        corpus.append( re.split('\W+', txt)  )
        docids.append(docid)
    return docids, corpus


def TREC_output(qid, run_name = 'bm25', fpath = None):
    ids, corp = get_corpus(qid)
    bm25 = BM25(corp)
    query = QUERIES[qid]
    scores = bm25.BM25Score(query)
    score_id_pairs = zip(scores, ids) # list of (score, docid) tuples
    ranked_pairs = sorted(score_id_pairs, reverse=1)
#     print ranked_pairs[:10]
    fout = sys.stdout if fpath==None else open(fpath, 'a')
    for rank, (score, docid) in enumerate(ranked_pairs[:2000], 1):
        print >>fout, '%d  Q0  %s  %d  %f  %s' % (qid, docid, rank, score, run_name)



fpath = '../../data/trec-output/bm25_tp.txt'
open(fpath, 'w') # clear previous results
for qid in tqdm(QUERIES.keys()):
    TREC_output(qid, fpath=fpath)


#  trec_eval/trec_eval -q -c qrels.txt bm25_desc.txt > bm25_desc.eval

