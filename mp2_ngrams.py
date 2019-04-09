# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 23:16:35 2019

@author: Nikita
"""


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)


import pandas as pd


def divide_pos_tweets(Z):
    
    Z_pos_sexism = Z[Z['Class'] == "sexism"]
    Z_pos_racism = Z[Z['Class'] == "racism"]
    
    frames = [Z_pos_sexism, Z_pos_racism]
    
    Z_pos = (pd.concat(frames)).sort_values(by=['ID'])
    
    return Z_pos
    
def divide_neg_tweets(Z):
    
    Z_neg = Z[Z['Class'] == 'none']
    
    return Z_neg

def makewords(Z):
    
    #WORD LEVEL N-GRAMS (keep numbers as they represent unicode for emojis)
    cv = CountVectorizer(stop_words='english', min_df=.002, max_df=.8, ngram_range=(2,2))
    cv.fit(Z.stemmed)
    cv_mat = cv.transform(Z.stemmed)
    
    bigrams = pd.DataFrame(cv_mat.todense(), index=Z['ID'], columns=cv.get_feature_names())
    bigrams = bigrams.add_prefix('word_bigrams:')
    oc = np.asarray(cv_mat.sum(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'Term': cv.get_feature_names(), '# occurrences': oc})
    counts_df.sort_values(by='# occurrences', ascending=False)
    
                                                    
    return counts_df

def makescoreset(Z_pos_terms, Z_neg_terms):
    
    ngrams_set = []
    words = pd.DataFrame(Z_pos_terms, columns = ["Term"]).values.ravel().tolist()
    occurrences = pd.DataFrame(Z_pos_terms, columns = ['# occurrences']).values.ravel().tolist()
    for i in range(len(words)):
        dic_pos_set = {}
        dic_pos_set[words[i]] = occurrences[i]   
        ngrams_set.append((dic_pos_set, 'pos'))
        
    words = pd.DataFrame(Z_neg_terms, columns = ["Term"]).values.ravel().tolist()
    occurrences = pd.DataFrame(Z_neg_terms, columns = ['# occurrences']).values.ravel().tolist()
    for i in range(len(words)):
        dic_neg_set = {}
        dic_neg_set[words[i]] = occurrences[i]    
        ngrams_set.append((dic_neg_set, 'neg'))
        
    return ngrams_set    



