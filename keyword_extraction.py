from text_preprocessing import clean_original_keywords, preprocess_text

## PREPARE THE DATASET

import os
import pandas as pd
path = "/Users/maingo/Downloads/theses100/"
all_files = os.listdir(path + "docsutf8") # content file
all_keys = os.listdir(path + "keys") # file of keywords of the texts
# print(len(all_files), " files n", all_files, "n", all_keys)

all_documents = []
all_keys = []
all_files_names = []
for i, fname in enumerate(all_files):
    # print(fname)
    # print(path + 'docsutf8/' + fname)
    with open(path + 'docsutf8/' + fname) as f: # open each txt file
        lines = f.readlines() # get the content
    key_name = fname[:-4] # thesis holder's name
    # print(key_name)
    with open(path + 'keys/' + key_name + '.key') as f:
        k = f.readlines()
    all_text = ''.join(lines)
    keys = ''.join(k)
    # print(keys)
    all_documents.append(all_text) # add each thesis content to the all_documents list
    all_keys.append(keys.split("\n")) # split list of keywords by 'n'
    all_files_names.append(key_name)

dtf = pd.DataFrame({'goldkeys':all_keys, 'text': all_documents})
# print(dtf.head())


# print(dtf)
original_keywords = clean_original_keywords(dtf['goldkeys'])
# print(original_keywords)
dtf['cleaned_text'] = dtf.text.apply(lambda  x: ' '.join(preprocess_text(x)))

### Keywords Extraction using TFIDF
# 1. Generating n-grams (keyphrases) and weighing them

# import TFIDF Vectorizer from the text feature extraction package
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#set idf=true (we want to use the inverse document frequency IDF with the term frequency)
#max is 0.5 -- we only want terms that occur in 50 percent of the documents
# if a term appears in omre than 50 documents, it will be deleted because
# it is considered non-discriminatory at the corpus level

# specify the range of n-grams from one to three -- according to the statistics of the current dataset,
# the largest proportion if for keywords that are 1-3 in length
vectorizer = TfidfVectorizer(use_idf=True, max_df=0.5, min_df=1, ngram_range=(1,3))

# generates the documents' vectors
vectors = vectorizer.fit_transform(dtf['cleaned_text'])

# for each document, build a dictionary (dict_of_tokens) where the key is the word
# and the value is the TFIDF weight
# create a list of tfidf_vectors to store the dictionaries of all documents

# print("vectors: ", vectors)

dict_of_tokens = {i[1] : i[0] for i in vectorizer.vocabulary_.items()}
tfidf_vectors = []
for row in vectors:
    tfidf_vectors.append({dict_of_tokens[column]: value for (column, value) in zip(row.indices, row.data)})

# print(len(tfidf_vectors))

# let's see what this dictionary contains for the first document
# print('The dictionary of the first document: ', tfidf_vectors[0])

# 2. Sorting keyphrases by TFIDF weights (in descending order)
# set 'reverse=True' to make the order descend

doc_sorted_tfidfs = [] # list of doc features each with tfidf weight

# sort each dict of a document
for dn in tfidf_vectors:
    newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
    newD = dict(newD)
    doc_sorted_tfidfs.append(newD)

# get a list of keywords without their weights
tfidf_keywords = []
for doc_sorted_tfidf in doc_sorted_tfidfs:
    keyword_list = list(doc_sorted_tfidf.keys())
    tfidf_keywords.append(keyword_list)

# let's choose the Top 5 keywords for the first document
print(tfidf_keywords[0][:5])

# Performance evaluation

# note that we use an exact match for the evaluation, where the keyphrase automatically
# extracted from the document must exactly match the document's gold standard keyphrase

# keyword extraction is a ranking problem
# one of the most commonly used measures for ranking is Mean Average Precision at K, MAP@K
# To calculate MAP@K, the precision at K elements p@K is first considered as the basic metric of the ranking quality for one document

def calculate_apk(actual_keywords, predicted_keywords, top_k=10):
    # this function accepts 2 paramters: the list of keywords predicted by the TFIDF method
    # and the list of gold standard keywords -- the default value for k is 10
    if (len(predicted_keywords) > top_k):
        predicted_keywords = predicted_keywords[:top_k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted_keywords):
        if p in actual_keywords and p not in predicted_keywords[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual_keywords:
        return 0.0
    return score / min(len(actual_keywords), top_k)

def mapk(actual_keywords, predicted_keywords, top_k = 10):
    return np.mean([calculate_apk(a, p, top_k) for a, p in zip(actual_keywords, predicted_keywords)])

# print MAP value at k = [5, 10, 20, 40]
for k in [5, 10, 20, 40]:
    mpak = mapk(original_keywords, tfidf_keywords, k)
    print("Mean average precision at ", k, '= {0: .4g}'.format(mpak))

# This TFIDF method relies on corpus statistics to weight the extracted keywords, so
# it cannot be applied to a single text -- one of its drawbacks

