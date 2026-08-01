import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai


# ==================================================
# Load local .env only when it exists
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(
        dotenv_path=ENV_FILE,
        override=False,
    )


# ==================================================
# Read API Key
# ==================================================
def get_google_api_key() -> str:
    """
    Read the Gemini API key.

    Local development:
        Reads GOOGLE_API_KEY from the root .env file.

    Streamlit Cloud:
        Reads GOOGLE_API_KEY from Streamlit Secrets.
    """

    # First try environment variable
    api_key = os.getenv("GOOGLE_API_KEY")

    # Then try Streamlit Cloud Secrets
    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY was not found. "
            "For local development, add it to the root .env file. "
            "For Streamlit Cloud, add it under App Settings → Secrets."
        )

    return api_key.strip()


# ==================================================
# Gemini Client
# ==================================================
@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """
    Create and cache the Gemini API client.
    """

    return genai.Client(
        api_key=get_google_api_key()
    )


# Existing helper files can import this directly:
# from helpers.gemini_client import client
client = get_gemini_client()