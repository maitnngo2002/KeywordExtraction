


## TEXT PREPROCESSING: Remove unwanted noise

# Preprocessing includes tokenization, lemmatization, lowercasing, removing numbers, white spaces and
# words shorter than three letters, removing stop words, remove symbols and punctuation.

# clean text applying all the text preprocessing functions

# For cleaning the text

import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
import string
import numpy as np
import nltk.data
import re
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag

# cleaning text function

def preprocess_text(text):
    #1. Tokenize to alphabetic tokens
    text = remove_numbers(text)
    text = remove_http(text)
    text = remove_punctuation(text)
    text = convert_to_lowercase(text)
    text = remove_white_space(text)
    text = remove_short_words(text)

    tokens = tokenizing(text)

    # POS tagging
    pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}

    pos_tag_list = pos_tag(tokens)

    # print("Pos_tag: ", pos_tag)

    # lowercase and lemmatise
    lemmatiser = WordNetLemmatizer()
    tokens = [lemmatiser.lemmatize(w.lower(), pos=pos_map.get(p[0], 'v')) for w, p in pos_tag_list]

    return tokens
# preprocessing helper functions
def remove_numbers(text):
    text = re.sub(r'd+', '', text)
    return text
def remove_http(text):
    text = re.sub("https?://t.co/[A-Za-z0-9]*", ' ', text)
    return text
def remove_punctuation(text):
    punctuations = '''!()[]{};«№»:'",`./?@=#$-(%^)+&[*_]~'''

    cleaned_text = ""

    for char in text:
        if char not in punctuations:
            cleaned_text += char

    return cleaned_text
def convert_to_lowercase(text):
    return text.lower()
def remove_white_space(text):
    return text.strip()
def remove_short_words(text):
    text = re.sub(r'bw{1,2}b', '', text)
    return text

def tokenizing(text):
    stop_words = set(stopwords.words('english'))
    # print(stop_words)
    tokens = word_tokenize(text)
    result = [i for i in tokens if i not in stop_words] # remove stopwords from tokens
    return result

# clean the basic keywords and remove the spaces and noise
def clean_original_keywords(original_kw):
    original_kw_clean = []
    for doc_kw in original_kw:
        temp = []
        for t in doc_kw:
            tt = ' '.join(preprocess_text(t))
            if (len(tt.split()) > 0):
                temp.append(tt)
        original_kw_clean.append(temp)
    return original_kw_clean


