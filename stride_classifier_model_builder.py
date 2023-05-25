import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from text_preprocessing import prepare_sentence
from skops.io import dump
# Load raw CAPEC data

df = pd.read_excel('data/raw_capec_data.xlsx', sheet_name="Threats")
df["NameDesc"] = df["Name"] + " " + df["Desc"]
df['NameDesc'] = df['NameDesc'].astype(str)

# Preprocessing (removing stopwords, stemming, etc.)
corpus = df['NameDesc'].apply(prepare_sentence)

# Feature extraction

# Initialize the TF-IDF vectorizer
tfidf = TfidfVectorizer()
# Fit vectorizer (aka create and update the vocabulary with the TF-IDF output for each word in the corpus)
tfidf.fit(corpus)
# Creates a matrix: every feature (word in the corpus in TF-IDF format) for every row (or text line)
X = tfidf.transform(corpus)
X = X.toarray()
# Can be simplified to Xfeatures = tfidf.fit_transform(corpus).toarray()
y = df[list("STRIDE")]

# Now we establish which estimators and models will be tested to get the best one based on the accuracy parameter
best_model = ""
max_accuracy = 0

estimators = [
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset
]
models = [
    MultinomialNB(), ComplementNB(), GaussianNB(), BernoulliNB(),
    LinearSVC(max_iter=10000, tol=0.000005, C=10, class_weight="balanced"),
    LogisticRegression()
]

# Here we create our train and test data sets.
# Usually it needs two lists: the list of inputs and the outputs


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=64)

for model in models:
    for estimator in estimators:
        clf = estimator(model)
        clf.fit(X_train, y_train)
        clf_predictions = clf.predict(X_test)

        scores = dict()

        if len(y.columns) == 1:
            clf_predictions = clf_predictions.toarray()
        else:
            scores = {
                "score": clf.score(X_test, y_test),
                "f1_avg": f1_score(y_test, clf_predictions, average='samples'),
                "f1_macro": f1_score(y_test, clf_predictions, average='macro')
            }

        result = {
            "accuracy": accuracy_score(y_test, clf_predictions),
            "hamming_score": hamming_loss(y_test, clf_predictions),
        }

        result = result | scores

        print(f"Testing {str(model)} -> {str(estimator)} -> {result}")
        if result.get('accuracy') > max_accuracy:
            # print(f"New best model! {str(model)} -> {str(estimator)}")
            best_model = clf
            max_accuracy = result.get('accuracy')

print(f"Best model: {str(best_model)} with {max_accuracy} accuracy")


# Export model
dump(best_model, "model/best_model.skops")
with open('model/vectorizer.pickle', 'wb') as fin:
    pickle.dump(tfidf, fin)
