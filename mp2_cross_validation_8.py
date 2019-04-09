# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:29:01 2019

@author: Nikita
"""
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from random import shuffle
import pandas as pd
import datetime
import time
import mp2_ngrams

train = pd.read_csv('csvs/hate_speech_with_sentiment.csv', encoding = "ISO-8859-1")

Z_train, Z_test= train_test_split(train[:], test_size = 0.2, random_state = 300)

stemmer = SnowballStemmer("english")

####################################### 1. Train Score ########################################################
Z_train['stemmed'] = Z_train.Tweets.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

Z_pos_train = mp2_ngrams.divide_pos_tweets(Z_train)
Z_neg_train = mp2_ngrams.divide_neg_tweets(Z_train)

Z_pos_train_terms = mp2_ngrams.makewords(Z_pos_train)
Z_neg_train_terms = mp2_ngrams.makewords(Z_neg_train)

ngram_trainscore = mp2_ngrams.makescoreset(Z_pos_train_terms, Z_neg_train_terms)
print ("ngram training over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

####################################### 1. Test Score ########################################################

Z_test['stemmed'] = Z_test.Tweets.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

Z_pos_test = mp2_ngrams.divide_pos_tweets(Z_test)
Z_neg_test = mp2_ngrams.divide_neg_tweets(Z_test)

Z_pos_test_terms = mp2_ngrams.makewords(Z_pos_test)
Z_neg_test_terms = mp2_ngrams.makewords(Z_neg_test)

ngram_testscore = mp2_ngrams.makescoreset(Z_pos_test_terms, Z_neg_test_terms)
print ("ngram testing over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
shuffle(ngram_trainscore)
shuffle(ngram_testscore)


SVC_ngram = SklearnClassifier(SVC(gamma="auto"))
SVC_ngram.train(ngram_trainscore)
svc_ngram = nltk.classify.accuracy(SVC_ngram, ngram_testscore)*100
print("Accuracy using pmi score is: ", svc_ngram)
f = open("Accuracy_new.txt", 'a')
f.write("\n"+str(svc_ngram))
f.close()
print ("svc ngram completed!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
