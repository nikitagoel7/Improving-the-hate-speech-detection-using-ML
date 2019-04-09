# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:17:55 2019

@author: Nikita
"""

import pandas as pd
import math
import re
import nltk
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)



def count_occurrences(word, sentence):
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), sentence.lower()))
    return count


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
        niki.append(' '.join(dhruv))
    
    clean_tweets = niki
    return clean_tweets

def pmi_testing(X_train, X_test):
    X_test = X_test.sort_values(by=['ID'])
    test = pd.DataFrame(X_test, columns = ["Tweets"]).values.tolist()
    test = pd.DataFrame(test).values.ravel().tolist()
    test = clean(test)
    X_train = X_train.sort_values(by=['ID'])
    pmi_train = pd.DataFrame(X_train, columns = ["Tweets", "Class"])
    tweets = pd.DataFrame(X_train, columns = ["Tweets"]).values.tolist() #tweets is a list of tweets in training set because testing is always done using data of training set
    tweets = pd.DataFrame(tweets).values.ravel().tolist()
    
    pmi_test_list = []
    for i in range(0,len(test)):
        input_tweet = test[i]
        score = []
        dic = {}
        n = input_tweet.split(" ")
        for inp in range (0, len(n)):
            freq_t = 0
            freq_t_pos = 0
            freq_t_neg = 0
            for tw in range(0,len(tweets)):
                if pmi_train["Class"][tw] == "sexism" or pmi_train["Class"][tw] == "racism":
                    freq_t_pos += count_occurrences(n[inp], tweets[tw])
                else:
                    freq_t_neg += count_occurrences(n[inp], tweets[tw])
                freq_t += count_occurrences(n[inp], tweets[tw])
            p = math.log(freq_t_pos + 0.5) - math.log(freq_t_neg + 0.5)
        
            if np.isinf(p):
                p = 0
            if math.isnan(p):
               p = 0
            score.append(p)
            dic[n[inp]] = p
        
        if sum(score)>0:
            pmi_test_list.append((dic, 'pos'))
        else:
            pmi_test_list.append((dic, 'neg'))
        
    
    print(pmi_test_list)
            
    return pmi_test_list

def pmi_training(training_set):

    training = training_set
    training_set = pd.DataFrame(training, columns = ["Tweets"]).iloc[:,0].values.tolist()
    training_set = clean(training_set)
    
    sent_set = pd.DataFrame(training , columns = ["Class"]).iloc[:,0].values.tolist()
     
    pmi_train_list = []
    for i in range(0,len(training_set)):
        training_set[i] = training_set[i].split(" ")
        dic = {}
        if sent_set [i] == 'sexism' or sent_set [i] == 'racism':
            for k in training_set[i]:
                dic[k] = 1
            pmi_train_list.append((dic, 'pos'))
        else:
            for k in training_set[i]:
                dic[k] = -1
            pmi_train_list.append((dic, 'neg'))
            
    return pmi_train_list
        

  
