# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:08:44 2018

@author: Nikita
"""

'''
2(ii) -- TFIDF of sexist racist and none categories of training set
'''
#Import necessary packages.
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from Data_Cleaning_1 import term_vec


#Read in the csvs we created and convert them into lists.
sexism = pd.read_csv('csvs/sexist_tokens.csv', encoding = "ISO-8859-1", index_col=0)
racism = pd.read_csv('csvs/racist_tokens.csv', encoding = "ISO-8859-1", index_col=0)
other = pd.read_csv('csvs/other_tokens.csv', encoding = "ISO-8859-1", index_col=0)

sexism = list(sexism['Sexist tokens'])
racism = list(racism['Racist tokens'])
other = list(other['Other'])

#Convert the term vectors into a gensim dictionary, then create the TFIDF vectors from the bag-of-words corpora
dict = Dictionary(term_vec)           

corp = []
for i in range(0, len(term_vec)):
    corp.append( dict.doc2bow( term_vec[ i ] ) )


#  Create TFIDF vectors based on term vectors bag-of-word corpora
tfidf_model = TfidfModel(corp)         

tfidf1 = []
for i in range( 0, len( corp ) ):
    tfidf1.append( tfidf_model[ corp[ i ] ] )

for i in range (3):
    r = tfidf1.pop(0)
    if i == 0:
        sex_index = []
        sex_tfidf = []
        for j in range (len(r)):
            k = r.pop()
            sex_index.append(k[0])
            sex_tfidf.append(k[1])
    elif i == 1:
        race_index = []
        race_tfidf = []
        for j in range (len(r)):
            k = r.pop()
            race_index.append(k[0])
            race_tfidf.append(k[1])
    else:
        none_index = []
        none_tfidf = []
        for j in range (len(r)):
            k = r.pop()
            none_index.append(k[0])
            none_tfidf.append(k[1])


dictionary = dict.token2id
sex_df = pd.DataFrame({
        'sex_index': sex_index,
        'sex_tfidf': sex_tfidf})

race_df = pd.DataFrame({
        'race_index': race_index,
        'race_tfidf': race_tfidf})

none_df = pd.DataFrame({
    'none_index': none_index,
    'none_tfidf': none_tfidf})

