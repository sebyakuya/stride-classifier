import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
nltk.download('averaged_perceptron_tagger')

# Define a list of sample sentences and their corresponding labels
sentences = [
    ("The quick brown fox jumps over the lazy dog.", "animal"),
    ("The cat in the hat.", "animal"),
    ("I love to eat pizza.", "food"),
    ("Sushi is my favorite food.", "food")
]

# Tokenize each sentence into individual words and perform POS tagging on them
tagged_sentences = []
for sentence, label in sentences:
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    tagged_sentences.append((" ".join([tag for word, tag in pos_tags]), label))

# Split the data into training and testing sets
train_data = tagged_sentences[:2] + tagged_sentences[2:]
test_data = tagged_sentences[:2]

# Convert the data into feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([sentence for sentence, label in train_data])
y_train = [label for sentence, label in train_data]
X_test = vectorizer.transform([sentence for sentence, label in test_data])
y_test = [label for sentence, label in test_data]

# Train a Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing data
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)