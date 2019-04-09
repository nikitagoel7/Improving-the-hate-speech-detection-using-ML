# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:32:38 2019

@author: Nikita
"""

import pandas as pd
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
import nltk
import re
from TFIDFpercategory_2 import sex_tfidf, race_tfidf, none_tfidf

def clean(test):
    clean_tweets = []
    for i in test:
        tweet = str(i).lower()
        clean_tweets.append(' '.join(re.sub(" (@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweet).split()))
    
    stop_words = nltk.corpus.stopwords.words('english')
    niki =[]
    for i in range(len(clean_tweets)):
        dhruv = clean_tweets[i].split(" ")
        for word in dhruv:
            contractions = re.match('\'', word)
            if word in stop_words or contractions:
                dhruv.remove(word)
        niki.append(dhruv)
        
    clean_tweets = niki
    return clean_tweets


def tfidf(train):
    X_train = train.sort_values(by=['ID'])
    X_train = pd.DataFrame(X_train, columns = ["Tweets", "Class"])
    
    
    training_tweets = pd.DataFrame(X_train, columns = ["Tweets"]).values.tolist()
    training_tweets = pd.DataFrame(training_tweets).values.ravel().tolist()
    training_class = pd.DataFrame(X_train, columns = ["Class"]).values.tolist()
    training_class = [ item for t in training_class for item in t ]
    
    
    training_tweets = clean(training_tweets)
    
            
    dict = Dictionary(training_tweets)

    corp = []
    for i in range(0, len(training_tweets)):
        corp.append(dict.doc2bow( training_tweets[i] ) )
    
    tfidf_model = TfidfModel( corp )
    
    tfidf = []
    i = 0
    while i<len(corp):
        tfidf.append(  tfidf_model[ corp[i] ] )
        i = i+1
    
    bag = {}
    for i in range (0,len(tfidf)):
        for j in range (len(tfidf[i])):
            bag[training_tweets[i][j]] = tfidf[i][j]
    
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
    
    tfidf_trainscore = []
    
    for i in range (len(training_tweets)):
        dic = {}
        s_pos = 0
        s_neg = 0
        for word in training_tweets[i]:
            if word in bag and word in min_term:
                dic[word] = bag[key][1]
                if min_term[word] == 'Sexist' or min_term[word] == 'Racist':
                    s_pos += bag[key][1]
                else:
                    s_neg += bag[key][1]
        if s_pos >= s_neg:
            tfidf_trainscore.append((dic, 'pos'))
        else:
            tfidf_trainscore.append((dic, 'neg'))
        
    return tfidf_trainscore



    
    
    
    
