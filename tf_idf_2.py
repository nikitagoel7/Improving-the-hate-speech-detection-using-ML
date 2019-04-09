# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:23:23 2018

@author: User
"""

'''
2 - TFIDF
'''
#Import necessary packages.
import pandas as pd
from gensim.models import TfidfModel
from Data_Cleaning_1 import term_vec
from gensim.corpora import Dictionary

#Read in the csvs we created and convert them into lists.
sexism = pd.read_csv('csvs/sexist_tokens.csv', encoding = "ISO-8859-1", index_col=0)
racism = pd.read_csv('csvs/racist_tokens.csv', encoding = "ISO-8859-1", index_col=0)
other = pd.read_csv('csvs/other_tokens.csv', encoding = "ISO-8859-1", index_col=0)

sexism = list(sexism['Sexist tokens'])
racism = list(racism['Racist tokens'])
other = list(other['Other'])

#Convert the term vectors into a gensim dictionary, then create the TFIDF vectors from the bag-of-words corpora
corp = []
dict = Dictionary(term_vec)
for i in range(0, len(term_vec)):
    corp.append( dict.doc2bow( term_vec[ i ] ) )
    
#  Create TFIDF vectors based on term vectors bag-of-word corpora
tfidf_model = TfidfModel(corp)         
tfidf = [ ]
for i in range( 0, len( corp ) ):
    tfidf.append( tfidf_model[ corp[ i ] ] )
dictionary = dict.token2id

#Create a list called "terms." this is where we'll be appending the different TFIDF words for each list.
#(Terms will be a list of lists.) This for loop prints the words with the top 100 TFIDF values for each list,
#by sorting them and then taking the first 100 of the list, then matching the indices to the gensim dictionary 
#above. We append each term into an internal list, then append the whole list into the terms list of lists.

terms = []
for i in range(0,len(tfidf)):
    term = {}
    sorted_values = sorted(tfidf[i], key=lambda x: x[1])
    first = (len(sorted_values))-99
    top_100 = sorted_values[first:]
    values =  [int(i[0]) for i in top_100]
    num = i+1
    for key,val in dictionary.items():
        if val in values:
            term[key] = val
    if i!= 2:
        terms.append((term,'pos'))
    else:
        terms.append((term,'neg'))
    
#TFIDF FOR ACCURACY PREDICTION FOR SVC, SDGC, LOGISTIC ACCURACY PREDICTION 
#        FOR NAIVE BAYES ESTIMATION FOR DISTRIBUTING INPUT DATA TO
#        DIFFERENT CATEGORIES WORDS AND THEIR TFIDF ARE TAKEN IN THE DICT
terms1 = []
for i in range (0, len(tfidf)):
    term  = {}
    sorted_values = sorted(tfidf[i], key=lambda x: x[1])
    first = (len(sorted_values))-99
    top_100 = sorted_values[first:]
    values =  [int(i[0]) for i in top_100]
    num = i+1
    for j in top_100:
        for key, val in dictionary.items():
            if j[0] == val:
                term[key] = j[1]
    if i!= 2:
        terms1.append((term,'pos'))
    else:
        terms1.append((term,'neg'))


tfidf = pd.DataFrame({'Sexist_terms': terms[0], 'Racist_terms': terms[1],'Other_terms': terms[2]})
tfidf.to_csv('csvs/tfidf.csv')    
