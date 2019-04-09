# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:05:35 2019

@author: Nikita
"""

import datetime
import time
from flask import Flask, render_template, request
print ("Started", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
from mp2_app_support import check_pmi, check_tfidf, check_ngram_pmi, check_ngram
import warnings

warnings.filterwarnings("ignore", category = RuntimeWarning)
app = Flask(__name__, static_url_path = '/static')


f = open("Accuracy_new.txt", 'r')
accuracy = f.readlines()
f.close()

random_tweet = "RT @MrWolfeee: Can't stand female announcers doing play by play for football. Not sexist but every time I hear Holly Rowe doing a game"
@app.route('/home.html')
def home():
    return render_template('/home.html', pmi = accuracy[0], pmi_ngram = accuracy[1], ngram = accuracy[2])



@app.route('/pmi.html')
def pmi():
    if check_pmi(random_tweet) > 0:
        pmi.hate_p = 1
    else:
        pmi.hate_p = 0
    print("hello")
    return render_template('/pmi.html', tweet = random_tweet, hateness = pmi.hate_p)

@app.route('/pmi.html', methods = ['POST'])
def checking_pmi():
    pmi()
    if (check_pmi(request.form['tweet']) > 0):
        ans = 'pos' #EXTENSION: threshold
    else:
        ans = 'neg'
        
    return render_template('/pmi.html', answer = ans, tweet_entered = request.form['tweet'], tweet = random_tweet, hateness = pmi.hate_p)
 
@app.route('/ngrampmi.html')
def ngrampmi():
    if check_ngram_pmi(random_tweet) > 0 :
        ngrampmi.hate_np = 1
    else:
        ngrampmi.hate_np = 0

    return render_template('/ngrampmi.html', tweet = random_tweet, hateness = ngrampmi.hate_np)

@app.route('/ngrampmi.html', methods = ['POST'])
def checking_ngrampmi():
    ngrampmi()
    if (check_ngram_pmi(request.form["tweet"]) > 0):
        ans = 'pos' #EXTENSION: threshold
    else:
        ans = 'neg'
    return render_template('/ngrampmi.html', answer = ans, tweet_entered = request.form['tweet'], tweet = random_tweet, hateness = ngrampmi.hate_np)    


@app.route('/ngram.html')
def ngram():
    if check_ngram(random_tweet):
        ngram.hate_np = 1
    else:
        ngram.hate_np = 0
    return render_template('/ngram.html', tweet = random_tweet, hateness = ngram.hate_np)


@app.route('/ngram.html', methods = ['POST', 'GET'])
def checking_ngram():
    ngram()
    if (check_ngram(request.form['tweet']) > 0):
        ans = 'pos' #EXTENSION: threshold
    else:
        ans = 'neg'
    return render_template('/ngram.html', answer = ans, tweet_entered = request.form['tweet'], tweet = random_tweet, hateness = ngram.hate_np)

@app.route('/tfidf.html')
def tfidf():
    if check_tfidf(random_tweet):
        tfidf.hate_np = 1
    else:
        tfidf.hate_np = 0
    return render_template('/tfidf.html', tweet = random_tweet, hateness = tfidf.hate_np)

@app.route('/tfidf.html', methods = ['POST', 'GET'])
def checking_tfidf():
    tfidf()
    if (check_tfidf(request.form['tweet']) > 0):
        ans = 'pos' #EXTENSION: threshold
    else:
        ans = 'neg'
    return render_template('/tfidf.html', answer = ans, tweet_entered = request.form['tweet'], tweet = random_tweet, hateness = tfidf.hate_np)


if __name__ == "__main__":
    app.run(debug = True, host='127.0.0.1', port= 5000)
     
   
