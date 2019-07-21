import re
import nltk

import gensim
from gensim import corpora
from nltk.corpus import reuters as corpus
from nltk.corpus import wordnet as wn

import pyLDAvis.gensim

# STOP WORDS
en_stop = nltk.corpus.stopwords.words('english')
en_stop= ["said","say","mln","ct","net","dlrs","today","tonne","pct","shr","nil","company","lt","share","year","billion","price","anything","u","v","else","add","one","two","three","four","five","six","seven","eight","nine","ten","would"]+en_stop


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

    pattern1 = '[0-9]*'
    word = re.sub(pattern1, '', word)
    pattern2 = '\W'
    word = re.sub(pattern2, '', word)
    pattern3 = '^[a-z]$'
    word = re.sub(pattern3, '', word)

    if word is '':
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
    return document

def preprocess_documents(documents):
    return [preprocess_document(document) for document in documents]

def main():
    print(en_stop)
    # データ
    k = 100
    docs=[corpus.words(fileid) for fileid in corpus.fileids()[:k]]
    # 前処理
    pp_docs = preprocess_documents(docs)
    print(pp_docs)
    #documentを，gensim LDAが読み込めるデータ構造にする
    #辞書の作成
    dictionary = corpora.Dictionary(pp_docs)
    #コーパスの作成
    corpus_ = [dictionary.doc2bow(doc) for doc in pp_docs]
    #documentのcategory
    categories=[corpus.categories(fileid) for fileid in corpus.fileids()]

    alpha = 1
    eta = 1
    num_topics = 5
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus_, num_topics=num_topics, id2word=dictionary,
                                                alpha=0.1, eta=0.1, #minimum_probability=0.0
                                                )

    #(トピックID, 当該トピックにおける単語とそのprobability)  ※　のうち、上位num_words位
    topics = ldamodel.print_topics(num_words=15)
    for topic in topics:
        print(topic)

    #[(当該documentにおけるトピックIDとそのprobability　)]　 ※　のうち、minimum_probabilityの値を超えるもの
    for n,item in enumerate(corpus_[:10]):
        print("document ID "+str(n)+":" ,end="")
        print(ldamodel.get_document_topics(item))

    #全documentを学習に用いた場合結構時間がかかる(20min~)
    #gensimではK個のトピックに0~K-1のidが割り振られていたのに対し，pyLDAvisでは1~Kのidが割り振られていることに注意
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus_, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'result/lda-result-'+ str(num_topics) + '-' + str(alpha) + '-' + str(eta) + '.html')

if __name__ == "__main__":
    main()
