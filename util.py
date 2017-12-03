# core libraries
import random, re

# numerical libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# nlp libraries
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

# ml libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#
import gensim.corpora
from gensim.matutils import Sparse2Corpus, sparse2full
from gensim.utils import simple_preprocess
import pyLDAvis.gensim as gensimvis
import pyLDAvis

from IPython.display import HTML
from IPython.display import Image
from IPython.display import clear_output
%matplotlib inline

## install instructions
# install spacy with llvm support to access OpenMP ... ? 
# https://github.com/explosion/spaCy/issues/267
# install llvm and export the links as it says in the documentation
# then $ pip install -U spacy
# python -m spacy download en

def plot_counts(vec, counts, top_n=15, title='Word Counts', xlabel='Counts'):
    """Plot most common words in a corpus given CountVec and sparse matrix."""
    vocabulary = np.array(sorted(vec.vocabulary_.keys()))
    word_count_df = pd.DataFrame({
        'word': vocabulary,
        'count': np.array(counts.sum(axis=0))[0]
    })
    (word_count_df
     .sort_values('count', ascending=False)
     .iloc[:15]
     .set_index('word')['count']
     .sort_values()
     .plot(kind='barh')
    )
    plt.title(title)
    plt.xlabel(xlabel)

class FilterEntity(BaseEstimator, TransformerMixin):
    """Take document as set of (word, ENT) pairs and filter to a certain ENT."""
    def __init__(self, extract=None):
        self.extract = extract
    def fit(self, X):
        return self 
    def transform(self, X):
        X = X.apply(lambda x: filter(lambda word: word[1] == self.extract, x))
        return X.apply(lambda x: ' '.join(map(lambda word: word[0], x)))

def preprocess_text_POS(text_col):
    """Tag text with POS."""
    index = text_col.index
    return pd.Series({
        i: [(word.lemma_, word.pos_) for word in doc]
        for i, doc in enumerate(nlp.pipe(text_col, n_threads=10, batch_size=100))
    })
    
def preprocess_text_named_entity(text_col):
    """Tag text with Named Entities."""
    index = text_col.index
    docs = []
    for doc in nlp.pipe(text_col, n_threads=10, batch_size=10):
        docs.append([(e.text, e.label_) for e in doc.ents])
    return docs 

with open('stop_words.txt') as f:
    STOPWORDS = set([word.strip() for word in f])

def tokenize(text):
    """Preprocess and filter out STOPWORDS."""
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def doc_stream_iterator(doc_list):
    """Take list of docs or text col and tokenize/preprocess."""
    for doc in doc_list:
        yield(tokenize(doc))
        
class WikiLeaksLDACorpus(object):
    """Corpus wrapper for LDA. Takes lda.corpus.Dictionary and text col."""
    def __init__(self, df, dictionary):
        self.df = df
        self.dictionary = dictionary
    
    def __iter__(self):
        for doc in self.df:
            yield self.dictionary.doc2bow(tokenize(doc))

    def __len__(self):
        return len(self.df)

def get_gamma(text, id2word, size=10):
    """Method for extracting vectors."""
    # calculate topic distribution for the article to add
    bow_vector = id2word.doc2bow(tokenize(text))
    lda_vector = lda_model[bow_vector]
    gamma_full = np.array(sparse2full(lda_vector, size).tolist())
    return gamma_full
    
def make_pipe(filterer=None):
    """Filter Entity + Count Vec transformer pipe."""
    return Pipeline([
    ('fe', FilterEntity(filterer)),
    ('vec', CountVectorizer(
        min_df=.01,
        max_df=.8
    ))
])

RUN = False
if RUN:
    ## read in CSV
    records = pd.read_csv('cablegate_parsed_records.csv', index_col=0, engine='python')
    records = records[records.text.notnull()]

RUN = False
if RUN:
    ## POS Tagging
    pos_tagging = preprocess_text_POS(records.text.iloc[:1000].astype(unicode))
    pos_tagging.index = records.text.iloc[:1000].index

RUN = False
if RUN:
    ## Named Entity Tagging
    named_entities = preprocess_text_named_entity(records.text.iloc[:1000].astype(unicode))
    named_entities = pd.Series(named_entities, index=records.iloc[:1000].index)

RUN = False
if RUN:
    ## duplicates??
    records_slim = records.copy()
    records_slim.index = pd.Series(records_slim.index).apply(lambda x: '/'.join(x.split('/')[1:]))
    records_slim = records.reset_index().drop_duplicates('index').set_index('index')
    
    ## LDA
    # document stream
    doc_stream = doc_stream_iterator(records_slim.text)
    id2word = gensim.corpora.Dictionary(doc_stream)
#     id2word.filter_extremes(no_below=20, no_above=0.3)
    
    # train model
    corp = WikiLeaksLDACorpus(records_slim.text, id2word)
    lda_model = gensim.models.LdaModel(
        corp, 
        num_topics=40,
        id2word=id2word,
        passes=4
    )
    lda_model.save('lda_model.gensim')
    
    # score records
    theta = np.zeros((len(records_slim), 40))
    for i, text in enumerate(records_slim.text):
        theta[i] = get_gamma(text, id2word, size=40)
                
    theta_df = records_slim.copy()
    for i in range(40):
        theta_df['topic_%d' % (i + 1)] = theta[:, i]

# visualize
VISUALIZE = False
if VISUALIZE:
    vis_data = gensimvis.prepare(lda_model, corp, id2word)
    pyLDAvis.display(vis_data)
    
    
# active learning
def setup_active_learning(records_slim):
    df = records_slim.copy()
    df['y'] = np.nan
    df['max_p'] = .5
    df['class_label'] = np.nan

    pipe = Pipeline([
        ('vec', CountVectorizer(
            "content",
            stop_words="english",
            max_df=.3,
            min_df=.001
        )),
        ('model', LogisticRegressionCV())
    ])
    
    return df, pipe


def run_active_learning(df, pipe, batch_size=10):
    query_sample = df[df.y.isnull()].sort_values('max_p').iloc[:batch_size]
    for i, row in query_sample.iterrows():
        text = row.text
        print 
        print "..." + text[2000:5000] + "..."
        print
        answer = input("Is this interesting? [0 no, 1 yes]")
        df.set_value(i, 'y', answer)
        df.set_value(i, 'max_p', np.nan)
        clear_output()

    # split labelled/unlabelled
    labelled = df[df.y.notnull()].copy()
    unlabelled = df[df.y.isnull()].copy()

    # split training/validation
    X_train, X_test, y_train, y_test = train_test_split(
        labelled.text,
        labelled.y,
        test_size=.2
    )

    # train
    pipe.fit(X_train, y_train)

    # predict probabilities
    validation_probs = pipe.predict_proba(X_test)[:, 1]
    unlabelled_probs = pipe.predict_proba(unlabelled.text)

    df.loc[unlabelled.index, 'max_p'] = unlabelled_probs.max(axis=1)
    df.loc[unlabelled.index, 'class_label'] = unlabelled_probs.argmax(axis=1)
    return df, pipe, roc_auc_score(y_test, validation_probs)

import warnings
warnings.simplefilter('ignore')