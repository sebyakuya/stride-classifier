import pickle
import pandas as pd
from text_preprocessing import prepare_sentence
from skops.io import load


best_model = load("model/best_model.skops", trusted=True)
tfidf = pickle.load(open("model/vectorizer.pickle", 'rb'))

df_unseen = pd.read_excel('data/unseen_data.xlsx')
df_unseen["NameDesc"] = df_unseen["Name"] + " " + df_unseen["Desc"]
corpus = df_unseen['NameDesc'].apply(prepare_sentence)

n_predicted = 0

n_sentences = 0
for sentence in corpus:
    vec_example = tfidf.transform([sentence])
    res = best_model.predict(vec_example).toarray()

    predicted = "".join(str(res).replace("[", "").replace("]", "").replace(" ", ""))
    expected = df_unseen['STRIDE'].iloc[n_sentences].replace("\"", "")
    print(f"{sentence[0:40]}...  -> Predicted: {predicted} -> Expected: {expected} -> {predicted == expected}")
    if predicted == expected:
        n_predicted += 1
    n_sentences += 1

print(f"Accuracy: {n_predicted / n_sentences}")

print("Write something")
sc = input()
print(sc)
sc = prepare_sentence(sc)
vec_example = tfidf.transform([sc])
res = best_model.predict(vec_example).toarray()
print(f"{sc[0:40]}...  -> {res}")
