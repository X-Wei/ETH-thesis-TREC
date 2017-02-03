import gensim
from gensim import corpora
import math, os

class BM25 :
    def __init__(self, fn_docs, delimiter='|') :
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.delimiter = delimiter
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.fn_docs = fn_docs
        self.DocLen = []
        self.buildDictionary()
        self.TFIDF_Generator()

    def buildDictionary(self) :
        raw_data = []
        for line in file(self.fn_docs) : # one line per document
            raw_data.append(line.strip().split(self.delimiter))
        self.dictionary.add_documents(raw_data)

    def TFIDF_Generator(self, base=math.e) :
        docTotalLen = 0
        for line in file(self.fn_docs) :
            doc = line.strip().split(self.delimiter)
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

from lxml import etree
topic_tree = etree.parse('data/topics2016.xml')

def get_topic(i):# returns the summary string of the ith topic
    summary = topic_tree.xpath('//topic[@number="%d"]/summary/text()'%i)[0]
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
        raise Exception(u'abstraction too short: '+pmcid + ret)
    return ret.lower()    
    

if __name__ == '__main__' :
    #mycorpus.txt is as following:
    '''
    Human machine interface for lab abc computer applications
    A survey of user opinion of computer system response time
    The EPS user interface management system
    System and human system engineering testing of EPS
    Relation of user perceived response time to error measurement
    The generation of random binary unordered trees
    The intersection graph of paths in trees
    Graph IV Widths of trees and well quasi ordering
    Graph minors A survey
    '''
    fn_docs = 'mycorpus.txt'
    #~ bm25 = BM25(fn_docs, delimiter=' ')
    #~ Query = 'The intersection graph of paths in trees survey Graph'
    #~ Query = Query.split()
    #~ scores = bm25.BM25Score(Query)
    #~ scores_tfidf = bm25.TFIDF()
    #~ print bm25.Items()
    #~ for i, tfidfscore in enumerate(scores_tfidf):
        #~ print i, tfidfscore
