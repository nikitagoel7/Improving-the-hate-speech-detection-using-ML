# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:04:23 2018

@author: Nikita
"""

'''
1 - Data Cleaning
''' 
import pandas as pd
import re
import nltk

hs = pd.read_csv(r'csvs/hatespeech.csv', encoding="ISO-8859-1",index_col=6, keep_default_na=False)

orig = pd.read_csv(r'csvs/NAACL_SRW_2016.csv', index_col=0, header=None)
orig.index.name = 'ID'
orig = orig.rename(columns={1: 'Class'})
orig.index = orig.index.astype(str)

#merging the two dataframes
hs = pd.merge(hs, orig, how='inner', left_index=True, right_index=True)

sexism =  hs.loc[hs['Class'] == 'sexism']
racism = hs.loc[hs['Class'] == 'racism']
none = hs.loc[hs['Class'] == 'none']

s_tweets = list(sexism.Tweets)
r_tweets = list(racism.Tweets)
n_tweets = list(none.Tweets)

class_list = [s_tweets, r_tweets, n_tweets]

punctuation = [':',';','!',',','.']

term_vec = []

for i in class_list:
    doc = []
    doc2 = []
    for d in i:
        d = re.sub('\@\w+', '', d)
        d = re.sub('\#\w+','', d)
        d = re.sub('\#','',d)
        d = re.sub('RT','',d)
        d = re.sub('&amp;','',d)
        d = re.sub('[0-9]+','',d)
        d = re.sub('//t.co/\w+','',d)
        d = re.sub('w//','',d)
        d = d.lower()
        doc.append( nltk.word_tokenize( d ) )
    for j in doc:
        for s in j:
            if s not in punctuation:
                doc2.append(s)
    term_vec.append(doc2)

stop_words = nltk.corpus.stopwords.words( 'english' )

for j in term_vec:
    for i in j:
        contractions = re.match('\'', i)
        if i in stop_words or contractions:
            j.remove(i)

porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )

sexist = pd.DataFrame({'Sexist tokens': term_vec[0]})
racist = pd.DataFrame({'Racist tokens': term_vec[1]})
other = pd.DataFrame({'Other':term_vec[2]})

racist.to_csv('csvs/racist_tokens.csv')
sexist.to_csv('csvs/sexist_tokens.csv')
other.to_csv('csvs/other_tokens.csv')

hs.to_csv('csvs/hs_merged.csv')            
