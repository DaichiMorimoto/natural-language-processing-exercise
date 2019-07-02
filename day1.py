import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

# 前処理
def preprocessing_text(text):
    def cleaning_text(text):
        pattern1 = '@|%'
        text = re.sub(pattern1, '', text)
        pattern2 = '\[[0-9 ]*\]'
        text = re.sub(pattern2, '', text)
        pattern3 = '\([a-z ]*\)'
        text = re.sub(pattern3, '', text)
        pattern4 = '[0-9]'
        text = re.sub(pattern4, '', text)
        return text

    def tokenize_text(text):
        text = re.sub('[.,]', '', text)
        return text.split()

    def lemmatize_word(word):
        # make words lower  example: Python =>python
        word=word.lower()
        # lemmatize  example: cooked=>cook
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    text = cleaning_text(text)
    tokens = tokenize_text(text)
    tokens = [lemmatize_word(word) for word in tokens]
    tokens = [remove_stopwords(word, en_stop) for word in tokens]
    tokens = [word for word in tokens if word is not None]
    return tokens

# ベクトル化
def tfidf_vectorizer(docs):
    def tf(word2id, doc):
        term_counts = np.zeros(len(word2id))
        for term in word2id.keys():
          term_counts[word2id[term]] = doc.count(term)
        tf_values = list(map(lambda x: x/sum(term_counts), term_counts))
        return tf_values

    def idf(word2id, docs):
        idf = np.zeros(len(word2id))
        for term in word2id.keys():
            idf[word2id[term]] = np.log(len(docs) / sum([bool(term in doc) for doc in docs]))
        return idf

    word2id = {}
    for doc in docs:
        for w in doc:
            if w not in word2id:
                word2id[w] = len(word2id)

    return [[_tf*_idf for _tf, _idf in zip(tf(word2id, doc), idf(word2id, docs))] for doc in docs], word2id

def bow_vectorizer(docs):
    word2id = {}
    for doc in docs:
        for w in doc:
            if w not in word2id:
                word2id[w] = len(word2id)

    result_list = []
    for doc in docs:
        doc_vec = [0] * len(word2id)
        for w in doc:
            doc_vec[word2id[w]] += 1
        result_list.append(doc_vec)

    return result_list, word2id

### 類似度計算 ###
def cosine_similarity(list_a, list_b):
    inner_prod = np.array(list_a).dot(np.array(list_b))
    norm_a = np.linalg.norm(list_a)
    norm_b = np.linalg.norm(list_b)
    try:
        return inner_prod / (norm_a*norm_b)
    except ZeroDivisionError:
        return 1.0

def minkowski_distance(list_a, list_b, p):
    diff_vec = np.array(list_a) - np.array(list_b)
    return np.linalg.norm(diff_vec, ord=p)

# テキストを読んでくる
def read_docs():
    return

def main():
    docs = read_docs()
    pp_docs = [preprocessing_text(text) for text in docs]

if __name__ == "__main__":
    main()
