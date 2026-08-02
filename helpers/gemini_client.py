"""
NeuraDocs shared AI client.

Primary provider:
    Google Gemini

Fallback provider:
    Groq

Other helper modules should only call:
    generate_content(prompt)
"""

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from groq import Groq


# Load local environment variables when running on the developer's computer.
# Streamlit Cloud reads keys from st.secrets.
load_dotenv(override=False)


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def get_secret(secret_name: str) -> str:
    """
    Read a secret from Streamlit Secrets first and then from
    environment variables or the local .env file.
    """

    secret_value = ""

    try:
        secret_value = str(
            st.secrets.get(secret_name, "")
        ).strip()
    except Exception:
        secret_value = ""

    if not secret_value:
        secret_value = os.getenv(
            secret_name,
            "",
        ).strip()

    return secret_value


def get_google_api_key() -> str:
    """Return the configured Gemini API key."""

    return get_secret("GOOGLE_API_KEY")


def get_groq_api_key() -> str:
    """Return the configured Groq API key."""

    return get_secret("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """
    Create and cache the Gemini client.

    Returns None when the Gemini API key is unavailable.
    """

    api_key = get_google_api_key()

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key,
    )


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """
    Create and cache the Groq client.

    Returns None when the Groq API key is unavailable.
    """

    api_key = get_groq_api_key()

    if not api_key:
        return None

    return Groq(
        api_key=api_key,
    )


# Compatibility variable for any older module that still imports:
# from helpers.gemini_client import client
client = get_gemini_client()


def should_use_groq_fallback(error: Exception) -> bool:
    """
    Return True when Gemini failed because of quota, authentication,
    connectivity, timeout, or temporary service problems.
    """

    error_text = str(error).lower()

    fallback_markers = (
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "too many requests",
        "503",
        "service unavailable",
        "unavailable",
        "high demand",
        "timeout",
        "timed out",
        "connection",
        "network",
        "api key",
        "permission denied",
        "unauthenticated",
    )

    return any(
        marker in error_text
        for marker in fallback_markers
    )


def generate_with_gemini(
    prompt: str,
    model_name: str,
) -> str:
    """Generate text using Gemini."""

    gemini_client = get_gemini_client()

    if gemini_client is None:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured."
        )

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    generated_text = getattr(
        response,
        "text",
        "",
    )

    if not generated_text or not generated_text.strip():
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return generated_text.strip()


def generate_with_groq(
    prompt: str,
    model_name: str,
) -> str:
    """Generate text using Groq."""

    groq_client = get_groq_client()

    if groq_client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    completion = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are NeuraDocs, a careful AI PDF assistant. "
                    "Follow the supplied instructions and never invent "
                    "facts that are not present in the provided PDF context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    if not completion.choices:
        raise RuntimeError(
            "Groq returned no completion choices."
        )

    generated_text = (
        completion.choices[0]
        .message
        .content
    )

    if not generated_text or not generated_text.strip():
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return generated_text.strip()


def generate_content(
    prompt: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
    groq_model_name: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    Generate content using Gemini first.

    If Gemini quota is exhausted or Gemini is temporarily unavailable,
    automatically use Groq as the fallback provider.
    """

    if not isinstance(prompt, str):
        raise TypeError(
            "The AI prompt must be a string."
        )

    cleaned_prompt = prompt.strip()

    if not cleaned_prompt:
        raise ValueError(
            "The AI prompt cannot be empty."
        )

    gemini_error = "Gemini was not attempted."
    groq_error = "Groq was not attempted."

    # Try Gemini first.
    if get_google_api_key():
        try:
            return generate_with_gemini(
                prompt=cleaned_prompt,
                model_name=model_name,
            )
        except Exception as error:
            gemini_error = str(error)

            if not should_use_groq_fallback(error):
                raise RuntimeError(
                    f"Gemini could not generate content: {error}"
                ) from error
    else:
        gemini_error = (
            "GOOGLE_API_KEY is not configured."
        )

    # Use Groq as fallback.
    if get_groq_api_key():
        try:
            return generate_with_groq(
                prompt=cleaned_prompt,
                model_name=groq_model_name,
            )
        except Exception as error:
            groq_error = str(error)
    else:
        groq_error = (
            "GROQ_API_KEY is not configured."
        )

    raise RuntimeError(
        "Both AI providers were unavailable. "
        f"Gemini error: {gemini_error} "
        f"Groq error: {groq_error}"
    )