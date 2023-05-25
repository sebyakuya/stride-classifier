import pickle
import pandas as pd
from text_preprocessing import prepare_sentence
from skops.io import load

best_model = load("model/best_model.skops", trusted=True)
tfidf = pickle.load(open("model/vectorizer.pickle", 'rb'))

df_unseen = pd.read_excel('data/mitre-attack-framework.xlsx', sheet_name="Threats")
df_unseen["NameDesc"] = df_unseen["Name"] + " " + df_unseen["Desc"]
corpus = df_unseen['NameDesc'].apply(prepare_sentence)


prd = []

n_sentences = 0
for sentence in corpus:
    vec_example = tfidf.transform([sentence])
    res = best_model.predict(vec_example).toarray()

    predicted = "".join(str(res).replace("[", "").replace("]", "").replace(" ", ""))
    prd.append(predicted)
    print(f"{sentence[0:40]}...  -> Predicted: {predicted}")

    n_sentences += 1

df_unseen["STRIDE"] = prd

with pd.ExcelWriter("data/mitre-classified.xlsx", engine="openpyxl") as writer:
    df_unseen.to_excel(writer, sheet_name="Threats")