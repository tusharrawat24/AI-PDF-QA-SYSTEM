import os

import streamlit as st
from dotenv import load_dotenv
from google import genai


# Local computer par .env load hogi.
# Streamlit Cloud par .env na mile to error nahi aayega.
load_dotenv(override=False)

DEFAULT_MODEL = "gemini-3.6-flash"


def get_google_api_key() -> str:
    """
    Get the Gemini API key from Streamlit Secrets
    or the local environment variable.
    """

    api_key = None

    # Streamlit Cloud secrets
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        api_key = None

    # Local .env / environment variable
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. "
            "Add it in Streamlit Cloud → App settings → Secrets, "
            "or put it inside the local .env file."
        )

    return api_key.strip()


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """
    Create and cache the Gemini client.
    """

    return genai.Client(
        api_key=get_google_api_key()
    )


client = get_gemini_client()


def generate_content(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Generate text using Gemini.

    This compatibility function is used by qa_chain,
    notes_generator and other helper modules.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "Gemini prompt cannot be empty."
        )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt.strip(),
        )

        generated_text = getattr(
            response,
            "text",
            None,
        )

        if not generated_text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return generated_text.strip()

    except Exception as error:
        raise RuntimeError(
            f"Gemini could not generate content: {error}"
        ) from error