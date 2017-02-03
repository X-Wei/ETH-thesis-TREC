# encoding: utf-8
from tqdm import tqdm
from lxml import etree
import os, nltk, sys, time
import numpy as np
from numpy.linalg import norm
import pandas as pd
import cPickle as pk

from settings import * 
np.random.seed(1) 

topic_tree = etree.parse(TOPICS_FPATH)

def get_topic(i):# returns the summary string of the ith topic
    summary = topic_tree.xpath('//topic[@number="%d"]/summary/text()'%i)[0]
    return str(summary).lower()

pmcid2fpath = {} # mapping docid to its local file path 
for subdir1 in os.listdir(PMC_PATH):
    for subdir2 in os.listdir(os.path.join(PMC_PATH, subdir1)):
        diry = os.path.join(PMC_PATH, subdir1, subdir2)
        for fn in os.listdir(diry):
            pmcid = fn[:-5]
            fpath = os.path.join(diry, fn)
            pmcid2fpath[pmcid] = fpath

with open(TOKENIZER_FPATH, 'rb') as f:
    tokenizer = pk.load(f)

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