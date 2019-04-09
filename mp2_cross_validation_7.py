# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:03:40 2019

@author: Nikita
"""
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
import mp2_pmi_ngrams
import datetime
import time

train = pd.read_csv('csvs/hate_speech_with_sentiment.csv', encoding = "ISO-8859-1")

Y_train, Y_test= train_test_split(train[:4000], test_size = 0.001, shuffle = False)

stemmer = SnowballStemmer("english")
Y_train['stemmed'] = Y_train.Tweets.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
Y_test['stemmed'] = Y_test.Tweets.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

#WORD LEVEL N-GRAMS (keep numbers as they represent unicode for emojis)
cv = CountVectorizer(stop_words='english', min_df=.002, max_df=.8, ngram_range=(2,2))
cv.fit(Y_train.stemmed)
cv_mat = cv.transform(Y_train.stemmed)

bigrams = pd.DataFrame(cv_mat.todense(), index=Y_train['ID'], columns=cv.get_feature_names())
bigrams = bigrams.add_prefix('word_bigrams:')
bigrams.to_csv('word_bigram_features_training_set.csv')


oc = np.asarray(cv_mat.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cv.get_feature_names(), '# occurrences': oc})
counts_df.sort_values(by='# occurrences', ascending=False)


pmi_ngram_trainscore = mp2_pmi_ngrams.pmi_training_ngrams(train, counts_df)    #score will be provided by training set
print ("pmi-ngram training over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
cv = CountVectorizer(stop_words='english', min_df=.002, max_df=.8, ngram_range=(2,2))
cv.fit(Y_test.stemmed)
cv_mat = cv.transform(Y_test.stemmed)

bigrams = pd.DataFrame(cv_mat.todense(), index=Y_test['ID'], columns=cv.get_feature_names())
bigrams = bigrams.add_prefix('word_bigrams:')
bigrams.to_csv('word_bigram_features_test_set.csv')


oc = np.asarray(cv_mat.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cv.get_feature_names(), '# occurrences': oc})
counts_df.sort_values(by='# occurrences', ascending=False)

pmi_ngram_testscore = mp2_pmi_ngrams.pmi_training_ngrams(train, counts_df)    #score will be provided by training set
print ("pmi-ngram testing over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

SVC_ngram_pmi = SklearnClassifier(SVC(gamma="auto"))
SVC_ngram_pmi.train(pmi_ngram_trainscore)
svc_ngram_pmi = nltk.classify.accuracy(SVC_ngram_pmi, pmi_ngram_testscore)*100
print("Accuracy using pmi score is: ", svc_ngram_pmi)

f = open("Accuracy_new.txt", 'a')
f.write("\n"+str(svc_ngram_pmi))
f.close()
print ("svc ngram pmi hybrid completed!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
