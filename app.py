"""
Embedding Models: From Text to Meaningful Vector Representations

Generates sentence embeddings with all-MiniLM-L6-v2, then uses cosine
similarity to find semantically similar sentence pairs.

Usage:
    python app.py                    # default threshold (0.50)
    python app.py --threshold 0.65   # show only pairs with similarity >= 0.65
"""

import argparse
from itertools import combinations

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"

SENTENCES = [
    "I enjoy coding.",
    "I like programming.",
    "The weather is sunny today.",
    "It is a bright and clear day outside.",
    "Machine learning models learn patterns from data.",
    "Neural networks are trained on large datasets.",
    "I love eating pizza.",
    "She is cooking pasta for dinner.",
    "The stock market closed higher today.",
    "Football is a popular sport around the world.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find semantically similar sentence pairs using embeddings."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum cosine similarity (between -1 and 1) to display a pair. Default: 0.50",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nNumber of sentences: {len(SENTENCES)}")
    for i, sentence in enumerate(SENTENCES, start=1):
        print(f"  {i}. {sentence}")

    # Convert sentences to embeddings (one vector per sentence).
    embeddings = model.encode(SENTENCES)
    print(f"\nEmbeddings shape    : {embeddings.shape}")
    print(f"Embedding dimension : {embeddings.shape[1]}")

    print("\nFirst 5 values of the first embedding:")
    print(embeddings[0][:5])

    # Cosine similarity between every pair of sentences.
    similarity_matrix = cosine_similarity(embeddings)

    pairs = [
        (similarity_matrix[i][j], SENTENCES[i], SENTENCES[j])
        for i, j in combinations(range(len(SENTENCES)), 2)
        if similarity_matrix[i][j] >= args.threshold
    ]
    pairs.sort(key=lambda p: p[0], reverse=True)

    print(f"\nSemantically similar sentence pairs (threshold = {args.threshold:.2f}):")
    print("-" * 60)
    if not pairs:
        print("No pairs found above the threshold. Try a lower --threshold value.")
    for score, s1, s2 in pairs:
        print(f"Sentence 1: {s1}")
        print(f"Sentence 2: {s2}")
        print(f"Similarity: {score:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
