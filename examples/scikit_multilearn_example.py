import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
import neattext as nt
import neattext.functions as nfx


# Data extraction and cleaning

df = pd.read_csv("dataset-tags.csv")

print(df.head())
print(df.dtypes)

df['mysql'] = df['mysql'].astype(float)
print(df.dtypes)

x = df['title'].apply(lambda x: nt.TextFrame(x).noise_scan())

print(x)

x = df['title'].apply(lambda x: nt.TextExtractor(x).extract_stopwords())

print(x)

corpus = df['title'].apply(nfx.remove_stopwords)

# Feature extraction

tfidf = TfidfVectorizer()

Xfeatures = tfidf.fit_transform(corpus).toarray()

y = df[['mysql', 'python', 'php']]

X_train, X_test, y_train, y_test = train_test_split(Xfeatures, y, test_size=0.3, random_state=42)

# Model

def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    clf_predictions = clf.predict(xtest)
    acc = accuracy_score(ytest, clf_predictions)
    ham = hamming_loss(ytest, clf_predictions)
    result = {"accuracy:": acc, "hamming_score": ham}
    return clf, result

# Technique 1: Binary relevance
clf_binary_rel_model, result_b = build_model(MultinomialNB(), BinaryRelevance, X_train, y_train, X_test, y_test)
print(result_b)

# Technique 2: Classifier chains
clf_chain_model, result_c = build_model(MultinomialNB(), ClassifierChain, X_train, y_train, X_test, y_test)
print(result_c)

# Technique 3: Label powerset
clf_labelP_model, result_lp = build_model(MultinomialNB(), LabelPowerset, X_train, y_train, X_test, y_test)
print(result_lp)

# Example

ex1 = df['title'].iloc[0]
vec_example = tfidf.transform([ex1])
res = clf_binary_rel_model.predict(vec_example).toarray()
print(res)
