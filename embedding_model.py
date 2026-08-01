import streamlit as st


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load and cache the sentence transformer model.
    """

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        local_files_only=True
    )


def create_embeddings(texts):
    """
    Convert PDF chunks into normalized embeddings.
    """

    if not texts:
        raise ValueError(
            "No text chunks were provided for embedding generation."
        )

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype("float32")


def create_query_embedding(question):
    """
    Convert the user's question into a normalized embedding.
    """

    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    model = load_embedding_model()

    embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embedding.astype("float32")