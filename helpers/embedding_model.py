import streamlit as st


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load and cache the Sentence Transformer embedding model.

    The model is downloaded from Hugging Face on the first run.
    After loading, Streamlit keeps it cached for later requests.
    """

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            MODEL_NAME
        )

        return model

    except Exception as error:
        raise RuntimeError(
            "The embedding model could not be loaded. "
            "Please check the internet connection and verify that "
            "sentence-transformers is installed correctly. "
            f"Original error: {error}"
        ) from error


def create_embeddings(texts: list[str]):
    """
    Convert PDF text chunks into normalized numerical embeddings.

    Parameters:
        texts: List of PDF text chunks.

    Returns:
        A float32 NumPy array containing one embedding per chunk.
    """

    if not texts:
        raise ValueError(
            "No text chunks were provided for embedding generation."
        )

    valid_texts = [
        text.strip()
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    if not valid_texts:
        raise ValueError(
            "The provided text chunks are empty or invalid."
        )

    model = load_embedding_model()

    try:
        embeddings = model.encode(
            valid_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype("float32")

    except Exception as error:
        raise RuntimeError(
            "Document embeddings could not be generated. "
            f"Original error: {error}"
        ) from error


def create_query_embedding(question: str):
    """
    Convert a user question into a normalized numerical embedding.

    Parameters:
        question: The user's question.

    Returns:
        A float32 NumPy array with shape (1, embedding_dimension).
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "The question cannot be empty."
        )

    model = load_embedding_model()

    try:
        embedding = model.encode(
            [question.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype("float32")

    except Exception as error:
        raise RuntimeError(
            "The question embedding could not be generated. "
            f"Original error: {error}"
        ) from error