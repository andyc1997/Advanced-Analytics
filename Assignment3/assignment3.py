"""
======================================================
Classification tweets - Assignment 3
======================================================

"""

import numpy as np
from time import time
import matplotlib.pyplot as plt
import os
import string
import joblib
import nltk
import spacy
import pandas as pd


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

english_stemmer = SnowballStemmer('english')
nlp = spacy.load('es_core_news_sm')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
nltk.download("punkt")

# Paths ======================================================================
os.chdir('/Users/amgiraldov/KU Leuven/Notebooks')
os.getcwd()

# Functions ==================================================================

def tokenize(text):
    doc = nlp(text)
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if t.isalpha()]
    stems = [english_stemmer.stem(token) for token in lexical_tokens]
    return stems

def tokenize1(text):
    doc = nlp(text)
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if t.isalpha()]
    return lexical_tokens



# Load data from the training set
# ------------------------------------

df = pd.read_excel("data.xlsx")

dummyTag = pd.get_dummies(df["label"])
dummyTag = pd.concat([df[["tweet_id","tweet_text"]], dummyTag], axis = 1).drop_duplicates()



data_train, data_test, train_labels, test_labels = train_test_split(dummyTag["tweet_text"], 
                                                                    dummyTag["covid"],
                                                                    stratify = dummyTag["covid"],
                                                                    test_size = 0.20)

results = []

for vectorizer in ((TfidfVectorizer(tokenizer = tokenize)),
                   (TfidfVectorizer(tokenizer = tokenize1)),
                   (HashingVectorizer(tokenizer = tokenize)),
                   (HashingVectorizer(tokenizer = tokenize1))):
    print('=' * 80)
    print("Vectorizer: ")
    print(vectorizer)
    X_train = vectorizer.fit_transform(data_train)
    X_test = vectorizer.transform(data_test)
    
    for classify in ((RidgeClassifier(tol=1e-2, solver="sag")),
                 (Perceptron(max_iter=50)),
                 (PassiveAggressiveClassifier(max_iter=50)),
                 (LinearSVC(penalty="l1", dual=False, tol=1e-3)),
                 (LinearSVC(penalty="l2", dual=False, tol=1e-3)),
                 (SGDClassifier(alpha=.0001, max_iter=50, penalty='l1')),
                 (SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")),
                 (BernoulliNB(alpha=.01))):
                  print('-' * 80)
                  print("Training: ")
                  print(classify)
                  classify.fit(X_train, train_labels)
                  pred = classify.predict(X_test)
                  score = metrics.accuracy_score(test_labels, pred)
                  print("accuracy:   %0.3f" % score)
                  print("dimensionality: %d" % classify.coef_.shape[1])
                  print("density: %f" % density(classify.coef_))
                  print("classification report:")
                  print(metrics.classification_report(test_labels, pred))
                  print("confusion matrix:")
                  print(metrics.confusion_matrix(test_labels, pred))
                  print('=' * 80)
                  print()



                  

       



