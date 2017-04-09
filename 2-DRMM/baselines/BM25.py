import gensim
from gensim import corpora
import math, os, sys

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
        for doc in corpus: # type: list[list[str]]
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