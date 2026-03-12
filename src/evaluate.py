from gensim.models import CoherenceModel


def compute_coherence(lda_model, texts, dictionary):
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        processes=1
    )
    return coherence_model.get_coherence()