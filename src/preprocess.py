import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

def preprocess(text):

    tokens = word_tokenize(text.lower())

    tokens = [t for t in tokens if t.isalpha()]

    tokens = [t for t in tokens if t not in stop_words]

    doc = nlp(" ".join(tokens))

    lemmas = [token.lemma_ for token in doc]

    return lemmas