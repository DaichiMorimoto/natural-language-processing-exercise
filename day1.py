import re
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

# 前処理
en_stop = nltk.corpus.stopwords.words('english')

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
        pattern5 = '\(.*?\)'
        text = re.sub(pattern5, '', text)
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

    def remove_stopwords(word, stopwordset):
        if word in stopwordset:
            return None
        else:
            return word

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

    return np.array([[_tf*_idf for _tf, _idf in zip(tf(word2id, doc), idf(word2id, docs))] for doc in docs]), word2id

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

    return np.array(result_list), word2id

### 類似度計算 ###
def cosine_similarity(list_a, list_b):
    inner_prod = list_a.dot(list_b)
    norm_a = np.linalg.norm(list_a)
    norm_b = np.linalg.norm(list_b)
    try:
        return inner_prod / (norm_a*norm_b)
    except ZeroDivisionError:
        return 1.0

def minkowski_distance(list_a, list_b, p):
    diff_vec = list_a - list_b
    return np.linalg.norm(diff_vec, ord=p)

# テキストを読んでくる
def read_docs(path):
    df = pd.read_csv(path)
    print(df)
    return df

def calc_sim(vec, is_cos):
    if is_cos:
        sims = np.zeros((vec.shape[0], vec.shape[0]))
        for i in range(vec.shape[0]):
            for j in range(vec.shape[0]):
                sims[i][j] = cosine_similarity(vec[i], vec[j])
    else:
        sims = np.full((vec.shape[0], vec.shape[0]), np.inf)
        for i in range(vec.shape[0]):
            for j in range(vec.shape[0]):
                sims[i][j] = minkowski_distance(vec[i], vec[j], 2)

    return sims

def print_max_pair(names, vecs, is_cos):
    if is_cos:
        idx = np.unravel_index(np.argmax(vecs), vecs.shape)
    else:
        idx = np.unravel_index(np.argmin(vecs), vecs.shape)
    print(names[idx[0]] + ' and ' + names[idx[1]])

# 正方行列と X および Y のラベルの行列を渡す
def draw_heatmap(data, row_labels, column_labels, name, is_cos):
    # 描画する
    fig, ax = plt.subplots()
    if is_cos:
        heatmap = ax.pcolor(data, cmap=plt.cm.YlOrRd)
    else:
        heatmap = ax.pcolor(data, cmap=plt.cm.YlOrRd_r)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig('result/'+ name +'.png')

    return heatmap

def main():
    # CSV読み込み兼前処理
    df = read_docs('./data/lang.csv')
    docs = df["Abstract"].values
    pp_docs = [preprocessing_text(text) for text in docs]
    # print(pp_docs)

    # TF-IDF
    tfidf_vector, word2id = tfidf_vectorizer(pp_docs)
    print('======= TF-IDF =======')
    print(tfidf_vector)

    # BoW
    bow_vec, word2id = bow_vectorizer(pp_docs)
    print('====== BoW =======')
    print(bow_vec)

    # 類似度計算
    cos_tfidf = calc_sim(tfidf_vector, True)
    min_tfidf = calc_sim(tfidf_vector, False)
    cos_bow = calc_sim(bow_vec, True)
    min_bow = calc_sim(bow_vec, False)

    names = df["Name"].values
    draw_heatmap(cos_tfidf, names, names, 'cos_tfidf', True)
    draw_heatmap(min_tfidf, names, names, 'min_tfidf', False)
    draw_heatmap(cos_bow, names, names, 'cos_bow', True)
    draw_heatmap(min_bow, names, names, 'min_bow', False)

'''
    print_max_pair(df["Name"].values, cos_tfidf, True)
    print_max_pair(df["Name"].values, min_tfidf, False)
    print_max_pair(df["Name"].values, cos_bow, True)
    print_max_pair(df["Name"].values, min_bow, False)
'''

if __name__ == "__main__":
    main()
