# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:29:01 2019

@author: Nikita
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import mp2_pmi
from nltk.classify.scikitlearn import SklearnClassifier  
import nltk
from sklearn.svm import SVC
import datetime
import time

train = pd.read_csv('csvs/hate_speech_with_sentiment.csv', encoding = "ISO-8859-1")

X_train, X_test= train_test_split(train[:], test_size = 0.001, shuffle = False)


print ("pmi training started!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
pmi_trainscore = mp2_pmi.pmi_training(X_train) 
print ("pmi training over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

print ("pmi testing started!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
pmi_testscore = mp2_pmi.pmi_testing(X_train, X_test)
print ("pmi testing over!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))



SVC_pmi = SklearnClassifier(SVC(gamma="auto"))
SVC_pmi.train(pmi_trainscore)
svc_pmi = nltk.classify.accuracy(SVC_pmi, pmi_testscore)*100
print("Accuracy using pmi score is: ", svc_pmi)
f = open("Accuracy_new.txt", 'a')
f.write(str(svc_pmi))
f.close()
print ("svc pmi completed!", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

