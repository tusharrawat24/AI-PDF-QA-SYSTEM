import os

import streamlit as st
from dotenv import load_dotenv
from google import genai


# Local computer par .env milegi to load hogi.
# Streamlit Cloud par .env na ho to koi error nahi aayega.
load_dotenv(override=False)


def get_google_api_key() -> str:
    """
    Get Gemini API key from:
    1. Streamlit Cloud Secrets
    2. Environment variables / local .env
    """

    api_key = None

    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it in "
            "Streamlit Cloud → App settings → Secrets."
        )

    return api_key.strip()


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """
    Create and cache the Google Gemini client.
    """

    return genai.Client(
        api_key=get_google_api_key()
    )


client = get_gemini_client()