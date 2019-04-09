# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:55:53 2019

@author: Nikita
"""

import pandas as pd
import nltk
import re
import math
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ngrams
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn import preprocessing
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from TFIDFpercategory_2 import sex_tfidf, race_tfidf, none_tfidf


train = pd.read_csv('csvs/hate_speech_with_sentiment.csv', encoding = "ISO-8859-1")

freq_pos = train["Class"].str.count("sexism|racism").sum()
freq_neg = train["Class"].str.count("none").sum()

def clean(test):

    tweet = str(test).lower()
    clean_tweets = ' '.join(re.sub(" (@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweet).split())
    
    stop_words = nltk.corpus.stopwords.words('english')
    dh = clean_tweets.split(" ")
    for word in dh:
        contractions = re.match('\'', word)
        if word in stop_words or contractions:
            dh.remove(word)
    return dh

def count_occurrences(word, sentence):
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), sentence.lower()))
    return count
    
def check_pmi(tweet):
    tweet = clean(tweet)
    training_set = pd.DataFrame(train, columns = ['Tweets', 'Class'])
    score = []
    for j in range (len(tweet)):
        freq_t_pos = 0
        freq_t_neg = 0
        freq_t = 0
        for i in range (len(training_set)):
            if training_set["Class"][i] == "sexism" or training_set["Class"][i] == "racism":
                freq_t_pos += count_occurrences(tweet[j], training_set["Tweets"][i])
            else:
                freq_t_neg += count_occurrences(tweet[j], training_set["Tweets"][i])
            freq_t += count_occurrences(tweet[j], training_set["Tweets"][i])
        p = math.log(freq_t_pos + 0.5) - math.log(freq_t_neg + 0.5)
        if np.isinf(p):
            p = 0
        if math.isnan(p):
           p = 0
        score.append(p)
    return sum(score)

def check_tfidf(tweet):
    tweet = clean(tweet)
    test = []
    for i in tweet:
        test.append([i]) 
        
    dict = Dictionary(test)

    corp = []
    for i in range(0, len(test)):
        corp.append(dict.doc2bow( test[i] ) )
    
    tfidf_model = TfidfModel( corp )
    
    tfidf = []
    i = 0
    while i<len(corp):
        tfidf.append(  tfidf_model[ corp[i] ] )
        i = i+1
    
    
    bag = {}
    for i in range (0,len(tfidf)):
        for j in range (len(tfidf[i])):
            bag[test[i][j]] = tfidf[i][j]
    
    min_term = {}
    for key, val in bag.items():
        diff_sex = []
        diff_race = []
        diff_none = []
        for j in sex_tfidf:
            diff_sex.append(abs(val[1]-j))
        diff_sex.sort()
        min_sex = diff_sex[0]
        for j in race_tfidf:
            diff_race.append(abs(val[1]-j))
        diff_race.sort()
        min_race = diff_race[0]
        for j in none_tfidf:
            diff_none.append(abs(val[1]-j))
        diff_none.sort()
        min_none = diff_none[0]
        min_val = min(min_sex, min_race, min_none)
        if min_val == min_sex:
            min_term[key] = 'Sexist'
        elif min_val == min_race:
            min_term[key] = 'Racist'
        else:
            min_term[key] = 'None'
    
    sexist_words = []
    racist_words = []
    none_words  = []
    
    for key,val in min_term.items():
        if val == 'Sexist':
            sexist_words.append(key)
        elif val == 'Racist':
            racist_words.append(key)
        else:
            none_words.append(key)
    if len(none_words) > len(sexist_words) and len(none_words) > len(racist_words):
        return 0
    else:
        return 1
    
    
def check_ngram_pmi(tweet):
    bi = ngrams(tweet.split(), 2)
    score = []
    for i in bi:
        score.append(check_pmi(i[0])+check_pmi(i[1]))
    return sum(score)


train_tweets = pd.DataFrame(train, columns = ["Tweets"]).values.tolist()
train_class = pd.DataFrame(train, columns = ["Class"]).values.tolist()
for i in range (len(train_tweets)):
    train_tweets[i] = clean(train_tweets[i])    

def check_ngram(tweet):
    tweet = [tweet]
    cv = CountVectorizer(stop_words = 'english', min_df = 0, max_df = 1, ngram_range = (1,3))
    tweet = pd.DataFrame({'col': tweet})
    cv.fit(tweet['col'])
    cv_mat = cv.transform(tweet['col'])

    bi = pd.DataFrame(cv_mat.todense(), columns = cv.get_feature_names())
    oc = np.asarray(cv_mat.sum(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'Term': cv.get_feature_names(), '# occurrences': oc})
    counts_df.sort_values(by='# occurrences', ascending=False)
    
    cl = []
    bi = pd.DataFrame(counts_df, columns = ['Term']).values.ravel().tolist()
    
    for b in bi:
        clas = []
        for j in range (len(train_tweets)):
            if b in train_tweets[j]:
                clas.append(' '.join(train_class[j]))
        print (b)
        print (clas)    
        if len(clas)/3 > (clas.count('sexism')+clas.count('racism')):
            cl.append('neg')
        else:
            cl.append('pos')
        print (cl)
    if cl.count('neg') >= cl.count('pos'):
        return  0
    else:
        return 1
        
