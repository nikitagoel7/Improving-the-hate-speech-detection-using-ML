# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:36:48 2018

@author: Nikita
"""

'''
3 - 7th sem end sem
'''
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import nltk
import re
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from tf_idf_2 import terms1
import matplotlib.pyplot as plt 
from TFIDFpercategory_2 import sex_tfidf, race_tfidf, none_tfidf


train = pd.read_csv('csvs/tfidf.csv', header = None)
none = train[1:][1]
racist = train[1:][2]
sexist = train[1:][3]

#datacleaning- part 1 removing unwanted words and hashtags, (more than 2 sentences required)
punctuation = [':',';','!',',','.', '&', '?', '/','>','<',']','[','-']
raw = input("Enter some text: ").split('.')
term_vec = []

for i in raw:
    doc2 = []
    i = re.sub('\@\w+', '', i)
    i = re.sub('\#\w+','', i)
    i = re.sub('\#','',i)
    i = re.sub('RT','',i)
    i = re.sub('&amp;','',i)
    i = re.sub('[0-9]+','',i)
    i = re.sub('//t.co/\w+','',i)
    i = i.lower()
    term_vec.append( nltk.word_tokenize( i ) )


#Data Cleaning-  part 2- removing punctuations 
for sen in term_vec:
    for word in sen:
        if word in punctuation:
            sen.remove(word)

#Data cleanig- part3- removing stop words           
stop_words = nltk.corpus.stopwords.words( 'english' )

for sen in term_vec:
    for word in sen:
        contractions = re.match('\'', word)
        if word in stop_words or contractions:
            sen.remove(word)

#Data Cleaning - part 4 -  combining each sentence into one list and iterating till length of term_vec so that every tokenized word is considered 
test = []
for i in range (0, len(term_vec)):
    for word in term_vec[i]:
        test.append([word])
    

#data Cleaning- part 5 - Porter Stemmer 
#porter = nltk.stem.porter.PorterStemmer()
#
#for sen in range( 0, len( test ) ):
#    test[ sen ] = porter.stem( test[ sen ] )
#print(term_vec[0][0:30])


#TFIDF - part 1 (more than 2 sentences required) converting normal dictionery to gensim 
    #type dictionary
dict = Dictionary(test)

#TFIDF = PART 2 - converting sentence into bag-of-word format = returns list(int, int),
# representing tokens and their frequency 
corp = []
for i in range(0, len(test)):
    corp.append(dict.doc2bow( test[i] ) )

#TDIDF = Part 3 -  Create TFIDF vectors based on term vectors bag-of-word corpora
tfidf_model = TfidfModel( corp )

tfidf = []
i = 0
#TFIDF = PART 4 - Assigning TFIDF value to each TFIDF vector
#returns vector: list ( int, float) float value is the TFIDF value for the int token
while i<len(corp):
    tfidf.append(  tfidf_model[ corp[i] ] )
    i = i+1

#TFIDF = part 5 - converting to dict:->(key:value) where key is the word  
    # Higher the value more is the hateness
d1 = dict.token2id

bag = {}
for i in range (0,len(tfidf)):
    for j in range (len(tfidf[i])):
        bag[test[i][j]] = tfidf[i][j]

#PREDICTION OF WORDS BELONGING TO RACIST, SEXIST AND NONE TYPE CATEGORIES
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
all1 = ' '.join([text for text in sexist_words])
all2 = ' '.join([text for text in racist_words])
all3 = ' '.join([text for text in none_words])

from wordcloud import WordCloud

def print_wordcloud(al):
    word_cloud = WordCloud(width = 200, height = 200, random_state = 21, max_font_size = 80).generate(al)
    plt.figure(figsize = (10, 7))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
if all1 == '' and all2 == '' and all3 == '':
    print ("Nothing to show")
    
if all1 == '' and all2 != '' and all3 == '':
    print ("Racist values: \n")    
    print_wordcloud(all2)

if all1 == '' and all2 == '' and all3 != '':
    print ("None values: \n")    
    print_wordcloud(all3)
            
if all1 == '' and all2 != '' and all3 != '':
    print ("Racist values: \n")    
    print_wordcloud(all2)
    print ("None values: \n")    
    print_wordcloud(all3)
    
if all1 != '' and all2 == '' and all3 == '':
    print ("Sexist values: \n")    
    print_wordcloud(all1)
    
if all1 != '' and all2 != '' and all3 == '':
    print ("Racist values: \n")    
    print_wordcloud(all2)
    print ("Sexist values: \n")    
    print_wordcloud(all1)

if all1 != '' and all2 == '' and all3 != '':
    print ("None values: \n")    
    print_wordcloud(all3)
    print ("Sexist values: \n")    
    print_wordcloud(all1)
            
if all1 != '' and all2 != '' and all3 != '':
    print ("Racist values: \n")    
    print_wordcloud(all2)
    print ("None values: \n")    
    print_wordcloud(all3)
    print ("Sexist values: \n")    
    print_wordcloud(all1)
    
       
#Accuracy prediction -- 7th end sem
pos_list = []
for key1, val1 in bag.items():
    for i in sexist_words:
        sex_test = {}
        if i == key1:
            sex_test[i] = val1[1]
        if (sex_test != {}):
            pos_list.append((sex_test, 'pos'))
    for i in racist_words:
        race_test = {}
        if i == key1:
            race_test[i] = val1[1]
        if (race_test != {}):
            pos_list.append((race_test, 'pos'))

neg_list = []
for key1, val1 in bag.items():
    for i in none_words:
        none_test = {}
        if i == key1:
            none_test[i] = val1[1]
        if (none_test != {}):
            neg_list.append((none_test, 'neg'))


ter = pos_list + neg_list

SVC_classifier1 = SklearnClassifier(SVC())
SVC_classifier1.train(terms1)
svc = nltk.classify.accuracy(SVC_classifier1, ter)*100
print ("\n\n\n")
print ("SVC accuracy for prediction of hatespeech in seperate categories: ",svc)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(terms1)
print ("\n\n\n")
logis = nltk.classify.accuracy(SVC_classifier1, ter)*100
print ("Logistic Regression accuracy for prediction of hatespeech in seperate categories: ",logis)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(terms1)
sdgc = (nltk.classify.accuracy(SGDClassifier_classifier, ter))*100
print ("\n\n\n")
print ("SGDC classifier accuracy for prediction of hatespeech in seperate categories: ",sdgc) 

        

#PERCENTAGE OF SEXISM AND RACISM AND NEUTRAL CONTENT
perc_sexism = (len(sexist_words)/(len(sexist_words+racist_words+none_words)))*100
perc_racism = (len(racist_words)/(len(sexist_words+racist_words+none_words)))*100
perc_none = (len(none_words)/(len(sexist_words+racist_words+none_words)))*100
print ('\n\n\n')
print ("Percentage sexism", perc_sexism)
print ("Percentage racism", perc_racism)
print ("Percentage of neutral content", perc_none)
print ('\n\n\n')

plt.xlim(0,1000)
plt.bar(['Old Accuracy', 'SVC', 'Logistic Regression', 'SDGC'], [40, svc, logis, sdgc], width = 0.5, align = 'center')
labels = 'Sexism', 'Racism', 'Neutral   '

plt.pie([perc_sexism, perc_racism, perc_none], labels = labels, autopct = '%1.1f%%', radius = 0.5)
