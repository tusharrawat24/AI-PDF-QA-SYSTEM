import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    PDF chunks ki embeddings ko FAISS index mein store karta hai.
    """

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("FAISS index banane ke liye embeddings nahi mili.")

    embeddings = np.asarray(embeddings, dtype="float32")

    # Embedding dimension, jaise 384
    dimension = embeddings.shape[1]

    # Normalized embeddings ke liye inner-product index
    index = faiss.IndexFlatIP(dimension)

    # Embeddings index mein add karo
    index.add(embeddings)

    return index


def search_similar_chunks(
    question_embedding,
    index,
    chunks,
    top_k=3
):
    """
    Question ke sabse relevant PDF chunks retrieve karta hai.
    """

    if index is None:
        raise ValueError("FAISS index available nahi hai.")

    if not chunks:
        raise ValueError("Search karne ke liye chunks available nahi hain.")

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Chunks kam hon to available chunks hi return karega
    number_of_results = min(top_k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        number_of_results
    )

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index == -1:
            continue

        results.append(
            {
                "chunk": chunks[chunk_index],
                "score": float(score),
                "chunk_number": int(chunk_index + 1)
            }
        )

    return results