import pandas as pd
from src.preprocess import preprocess
from src.lda_model import create_corpus, train_lda
from src.evaluate import compute_coherence
import pyLDAvis
import pyLDAvis.gensim_models
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer



def main():
    data = pd.read_csv("data/Restaurant_Reviews.tsv", sep="\t", quoting=3)

    print(data.head())
    print("Columns:", data.columns.tolist())
    print("Total Reviews:", len(data))

    data["processed"] = data["Review"].astype(str).apply(preprocess)

    texts = data["processed"].tolist()

    dictionary, corpus = create_corpus(texts)
    results = []

    for k in [3, 4, 5, 6, 7, 8]:
        lda_model = train_lda(corpus, dictionary, num_topics=k)
        score = compute_coherence(lda_model, texts, dictionary)
        results.append((k, score))
        print(f"Topics: {k}, Coherence Score: {score:.4f}")
    best_k = max(results, key=lambda x: x[1])[0]

    print(f"\nBest number of topics: {best_k}")

    lda_model = train_lda(corpus, dictionary, num_topics=best_k)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, "lda_visualization.html")

    print("\nSaved visualization as lda_visualization.html")
    texts_joined = [" ".join(text) for text in texts]

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts_joined)

    nmf = NMF(n_components=best_k, random_state=42)
    nmf.fit(X)

    feature_names = vectorizer.get_feature_names_out()

    print("\nNMF Topics:")
    for i, topic in enumerate(nmf.components_):
        top_words = [feature_names[j] for j in topic.argsort()[-10:]]
        print(f"Topic {i}: {top_words}")

    print("\nFinal Topics:")
    for topic in lda_model.print_topics():
        print(topic)

    lda_model = train_lda(corpus, dictionary, num_topics=5)

    print("\nTopics:")
    for topic in lda_model.print_topics():
        print(topic)

    score = compute_coherence(lda_model, texts, dictionary)
    print("\nCoherence Score:", score)
    print("\nTuning number of topics...")

    


if __name__ == "__main__":
    main()