""" 
    01Vectorizacion.py ===========================================================
    Description: his code navigates through the content of the sentences and 
    from the search for exact patterns it tags legal decisions with the 
    entities mentioned in it
    
    Author(s): AG
    
    input: Database of legal decisions
    
    output: Tag database
    
    coding: utf-8
    
    File history:
        202101: Creation
    """

# Libraries ==================================================================

import os
import pandas as pd
import string
import joblib
import numpy as np
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


spanish_stemmer = SnowballStemmer('spanish')
nlp = spacy.load('es_core_news_sm')
stop_words = set(stopwords.words('spanish'))
punctuation = set(string.punctuation)
nltk.download("punkt")


# Paths ======================================================================
os.chdir('/Users/amgiraldov/OneDrive - Consejo Superior de la Judicatura/Documentos/2021/EntrenamientoDerechos/src')
os.getcwd()

# Functions ==================================================================

def qualityText(sentence):
    tokens = tokenize(sentence)
    nToken = len(tokens)
    incorrectToken = [word for word in tokens if len(word) >= 20]
    propIncorrect = len(incorrectToken) / nToken
    if propIncorrect > 0.05:
        boolean = 'has_error'
    elif nToken > 21492 or nToken < 1228:
        boolean = "has_error"
    else:
        boolean = 'ok'   
    return boolean

def tokenize(text):
    doc = nlp(text)
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if len(t) > 3 and len(t) <= 20 and t.isalpha()]
    stems = [spanish_stemmer.stem(token) for token in lexical_tokens]
    return stems

def tokenize1(text):
    doc = nlp(text)
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if len(t) > 3 and len(t) <= 20 and t.isalpha()]
    return lexical_tokens


def vectorization(dataBaseTag): 
    
    global train_text, test_text, train_labels, test_labels
     # Database tags
    clasificacion = pd.read_excel(dataBaseTag)
    train_text = [x[:10000] for x in clasificacion['text']]

    
    # Learning the vocabulary
    
    count_Stemming = CountVectorizer(tokenizer = tokenize)
    train_Stemm_Count = count_Stemming.fit_transform(train_text) # Learn the vocabulary dictionary and return document-term matrix.
   # valid_Stemm_Count = count_Stemming.transform(test_text) # Extract token counts out of raw text documents using the vocabulary fitted with fit or the one provided to the constructor.
    nameJobLib = '../output/clasCountStem_First1k.joblib'
    joblib.dump(count_Stemming, nameJobLib) 
    
    
    count_NStemming = CountVectorizer(tokenizer = tokenize1)
    train_NStemm_Count = count_Stemming.fit_transform(train_text)
    nameJobLib = '../output/clasCountNStem_First1k.joblib'
    joblib.dump(count_Stemming, nameJobLib) 

    
    Tfidf_Stemmin = TfidfVectorizer(tokenizer = tokenize)
    train_Stem_Tfidf = Tfidf_NStemmin.fit_transform(train_text)
    nameJobLib = '../output/clasTfidfStem_First1k.joblib'
    joblib.dump(Tfidf_Stemmin, nameJobLib) 
    
    Tfidf_NStemmin = TfidfVectorizer(tokenizer = tokenize1)
    train_NStem_Tfidf = Tfidf_NStemmin.fit_transform(train_text)
    nameJobLib = '../output/clasTfidfNStem_First1k.joblib'
    joblib.dump(Tfidf_NStemmin, nameJobLib) 
    
    hashin_Stemmin = HashingVectorizer(tokenizer = tokenize)
    train_Stem_Hash = hashin_Stemmin.fit_transform(train_text)
    nameJobLib = '../output/clasHashStem_First1k.joblib'
    joblib.dump(hashin_Stemmin, nameJobLib) 
    
    hashin_NStemmin = HashingVectorizer(tokenizer = tokenize1)
    train_NStem_Hash = hashin_NStemmin.fit_transform(train_text)
    nameJobLib = '../output/clasHashNSte_First1km.joblib'
    joblib.dump(hashin_NStemmin, nameJobLib) 
    
  
dataBaseTag = "/Users/amgiraldov/OneDrive - Consejo Superior de la Judicatura/Documentos/2021/EntrenamientoDerechos/input/datosModelo.xlsx"




