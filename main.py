import pandas as pd
from src.preprocess import preprocess
from src.lda_model import create_corpus, train_lda
from src.evaluate import compute_coherence


def main():
    data = pd.read_csv("data/Restaurant_Reviews.tsv", sep="\t", quoting=3)

    print(data.head())
    print("Columns:", data.columns.tolist())
    print("Total Reviews:", len(data))

    data["processed"] = data["Review"].astype(str).apply(preprocess)

    texts = data["processed"].tolist()

    dictionary, corpus = create_corpus(texts)

    lda_model = train_lda(corpus, dictionary, num_topics=5)

    print("\nTopics:")
    for topic in lda_model.print_topics():
        print(topic)

    score = compute_coherence(lda_model, texts, dictionary)
    print("\nCoherence Score:", score)


if __name__ == "__main__":
    main()