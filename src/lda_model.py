from gensim.corpora import Dictionary
from gensim.models import LdaModel


def create_corpus(texts):
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

def train_lda(corpus, dictionary, num_topics=5):
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=42
    )
    return lda_model