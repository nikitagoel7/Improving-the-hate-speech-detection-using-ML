# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:08:13 2019

@author: Nikita
"""

import re
import pandas as pd
import numpy as np
import math
import sys
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)


                      
def count_occurrences(word, sentence):
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), sentence.lower()))
    return count
                      
def pmi_training_ngrams(train, test):
    train_tweets = pd.DataFrame(train, columns = ["Tweets"]).values.ravel().tolist()
    
    test = pd.DataFrame(test, columns = ["Term"]).values.ravel().tolist()
    a=[]
    for i in range(len(test)):
        a.append(test[i].split(" "))
    test = pd.DataFrame(a).values.ravel().tolist()
    
    pmi_train = pd.DataFrame(train, columns = ["Tweets", "Class"])
    
    freq_pos = pmi_train["Class"].str.count("sexism|racism").sum()
    freq_neg = pmi_train["Class"].str.count("none").sum()
    
    pmi_test_list = []
    for i in range(0,len(test)):
        score = []
        dic = {}
        freq_t = 0
        freq_t_pos = 0
        freq_t_neg = 0
        for tw in range(0,len(train_tweets)):
            if pmi_train["Class"][tw] == "sexism" or pmi_train["Class"][tw] == "racism":
                freq_t_pos += count_occurrences(test[i], train_tweets[tw])
            else:
                freq_t_neg += count_occurrences(test[i], train_tweets[tw])
            freq_t += count_occurrences(test[i], train_tweets[tw])
            p = math.log(freq_t_pos + 0.5) - math.log(freq_t_neg + 0.5)
        
            if np.isinf(p):
                p = 0
            
            if math.isnan(p):
                p = 0
            score.append(p)
            dic[test[i]] = p
        if sum(score)>0:
            pmi_test_list.append((dic, 'pos'))
        else:
            pmi_test_list.append((dic, 'neg'))
            
    return pmi_test_list

            
