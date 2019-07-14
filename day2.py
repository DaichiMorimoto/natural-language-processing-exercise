from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import collections

from nltk.corpus import reuters as corpus
from nltk.corpus import wordnet as wn

# STOP WORDS
en_stop = nltk.corpus.stopwords.words('english')
en_stop= ["``","/",",.",".,",";","--",":",")","(",'"','&',"'",'),',',"','-','.,','.,"','.-',"?",">","<"]                  \
         +["0","1","2","3","4","5","6","7","8","9","10","11","12","86","1986","1987","000"]                                                      \
         +["said","say","u","v","mln","ct","net","dlrs","tonne","pct","shr","nil","company","lt","share","year","billion","price"]          \
         +en_stop

# 前処理
def preprocess_word(word, stopwordset):
    #1.make words lower ex: Python =>python
    word=word.lower()

    #2.remove "," and "."
    if word in [",","."]:
        return None

    #3.remove stopword  ex: the => (None)
    if word in stopwordset:
        return None

    #4.lemmatize  ex: cooked=>cook
    lemma = wn.morphy(word)
    if lemma is None:
        return word

    elif lemma in stopwordset: #lemmatizeしたものがstopwordである可能性がある
        return None
    else:
        return lemma

def preprocess_document(document):
    document=[preprocess_word(w, en_stop) for w in document]
    document=[w for w in document if w is not None]
    return ' '.join(document)

def preprocess_documents(documents):
    return [preprocess_document(document) for document in documents]

### 類似度計算 ###
def cosine_similarity(list_a, list_b):
    inner_prod = list_a.dot(list_b)
    norm_a = np.linalg.norm(list_a)
    norm_b = np.linalg.norm(list_b)
    try:
        return inner_prod / (norm_a*norm_b)
    except ZeroDivisionError:
        return 1.0

def main():
    # データ
    k = 100
    docs=[corpus.words(fileid) for fileid in corpus.fileids()[:k]]

    # 前処理
    pp_docs = preprocess_documents(docs)

    # ベクトル化
    vectorizer = TfidfVectorizer(max_features=50, token_pattern=u'(?u)\\b\\w+\\b' )
    tf_idf = vectorizer.fit_transform(pp_docs)

    # K-means
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters, random_state=0, precompute_distances=True)

    clusters = km.fit_predict(tf_idf)
    categories=[','.join(corpus.categories(fileid)) for fileid in corpus.fileids()[:k]]
    keys = []
    for k, _ in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
        keys.append(k)

    w_df = pd.DataFrame({ 'class': clusters, 'category': categories })
    k_df = pd.DataFrame({ 'key': keys })
    print(k_df)
    print(w_df)
    w_df.to_csv('result/kmeans.csv')
    k_df.to_csv('result/tf_idf_key.csv')

if __name__ == "__main__":
    main()
