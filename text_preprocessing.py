import nltk
from nltk import SnowballStemmer, word_tokenize
import neattext.functions as nfx
from nltk.stem import 	WordNetLemmatizer
import string
#nltk.download('averaged_perceptron_tagger')


wordnet_lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
translate_table = dict((ord(char), None) for char in string.punctuation)


def stemming(sentence):
    """
    Stemming function to stem the document to it's root
    :param sentence:
    :return:
    """
    stemSentence = ""
    for word in sentence.split():
        stemSentence += stemmer.stem(word)
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


def lemmatize(sentence):
    lemmSentence = ""
    for w in word_tokenize(sentence):
        lemmSentence += wordnet_lemmatizer.lemmatize(w)
        lemmSentence += " "
    lemmSentence = lemmSentence.strip()
    return lemmSentence


def pos_tags(sentence):
    pos_tags = nltk.pos_tag(word_tokenize(sentence))

    taggedSentence = ""
    for w, tag in pos_tags:
        taggedSentence += w
    for w, tag in pos_tags:
        taggedSentence += tag

    return taggedSentence


def prepare_sentence(sentence):
    s = sentence.lower()
    #s = s.translate(translate_table)
    #s = pos_tags(s)
    s = nfx.remove_stopwords(s)
    s = stemming(s)
    #s = lemmatize(s)
    return s
