from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
import numpy as np
import sys
import os,re
from nltk.corpus import stopwords
from load import *

#initialise flask app

app = Flask(__name__)
Bootstrap(app)

#load the models from load.py

global model, graph
model, graph = init()

#word preprocessing
tokenizer=Tokenizer(num_words=2000,split=' ')
def html_to_text(review):
    """Return extracted text string from provided HTML string."""
    review_text = BeautifulSoup(review, "lxml").get_text()
    if len(review_text) == 0:
        review_text = review
    review_text = re.sub(r"\<.*\>", "", review_text)
    try:
        review_text = review_text.encode('ascii', 'ignore').decode('ascii')#ignore \xc3 etc.
    except UnicodeDecodeError:
        review_text = review_text.decode("ascii", "ignore")
    return review_text

def letters_only(text):
    """Return input string with only letters (no punctuation, no numbers)."""
    # It is probably worth experimenting with milder prepreocessing (eg just removing punctuation)
    return re.sub("[^a-zA-Z]", " ", text)

def rnn_tokenizer_review_preprocess(review):
    """Preprocessing used before fitting/transforming RNN tokenizer - Html->text, remove punctuation/#s, lowercase."""
    return letters_only(html_to_text(review)).lower()

#main code

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method=="POST":
        text=request.form['rawtext']
        print(text)
        text=[text]
        print(text)
        #test=rnn_tokenizer_review_preprocess(text)
        tokens=tokenizer.texts_to_sequences(text)
        review=pad_sequences(tokens,maxlen=28,dtype='int32',value=0)
        print(model.summary())
        with graph.as_default():
            ans=model.predict(review,batch_size=1,verbose=2)[0]
    print(ans)
    predict=(np.argmax(ans))
    return render_template('index.html',output=predict)

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0',port=port)
