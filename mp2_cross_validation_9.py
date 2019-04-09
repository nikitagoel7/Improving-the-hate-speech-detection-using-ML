# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:23:57 2019

@author: Nikita
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import mp2_tfidf
from nltk.classify.scikitlearn import SklearnClassifier  
import nltk
from sklearn.svm import SVC
import datetime
import time

train = pd.read_csv('csvs/hate_speech_with_sentiment.csv', encoding = "ISO-8859-1")

X_train, X_test= train_test_split(train[:], test_size = 0.2, shuffle = False)


tfidf_trainscore = mp2_tfidf.tfidf(X_train) 
print ("tfidf training over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
tfidf_testscore = mp2_tfidf.tfidf(X_test)
print ("tfidf testing over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))


SVC_tfidf = SklearnClassifier(SVC(gamma = "auto"))
SVC_tfidf.train(tfidf_trainscore)
svc_tfidf = nltk.classify.accuracy(SVC_tfidf, tfidf_testscore)*100
print ("SVC accuracy of tfidf with 16000 tweets: ",svc_tfidf)
f = open("Accuracy_new.txt", 'a')
f.write(str(svc_tfidf))
f.close()
print ("svc tfidf completed!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

