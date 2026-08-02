Shared AI client for NeuraDocs.

Primary provider:
    Google Gemini

Automatic fallback:
    Groq

Other helper modules should call:
    generate_content(prompt)
"""

import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai


# Load the local .env file when running on the developer's computer.
# Streamlit Cloud does not require a .env file.
load_dotenv(override=False)


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def _read_secret(secret_name: str) -> Optional[str]:
    """
    Read a secret from Streamlit Secrets first and then from
    environment variables / the local .env file.
    """

    secret_value = None

    try:
        secret_value = st.secrets.get(secret_name)
    except Exception:
        secret_value = None

    if not secret_value:
        secret_value = os.getenv(secret_name)

    if not secret_value:
        return None

    cleaned_value = str(secret_value).strip()

    return cleaned_value or None


def get_google_api_key() -> Optional[str]:
    """Return GOOGLE_API_KEY when configured."""

    return _read_secret("GOOGLE_API_KEY")


def get_groq_api_key() -> Optional[str]:
    """Return GROQ_API_KEY when configured."""

    return _read_secret("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """
    Create and cache the Gemini client.

    Returns None when GOOGLE_API_KEY has not been configured, allowing
    NeuraDocs to continue with the Groq fallback.
    """

    api_key = get_google_api_key()

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """
    Create and cache the Groq client.

    Returns None when GROQ_API_KEY has not been configured.
    """

    api_key = get_groq_api_key()

    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError(
            "The Groq package is not installed. "
            "Add 'groq' to requirements.txt and install it."
        ) from error

    return Groq(api_key=api_key)


# Kept for compatibility with older helper files.
# New helper files should use generate_content() instead of accessing
# client.models.generate_content() directly.
client = get_gemini_client()


def _is_retryable_gemini_error(error: Exception) -> bool:
    """
    Return True for errors where switching to Groq is appropriate.

    This includes quota, rate-limit, temporary server, timeout,
    authentication and connection-related failures.
    """

    error_text = str(error).lower()

    retryable_markers = (
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

    return any(marker in error_text for marker in retryable_markers)


def _generate_with_gemini(
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

    generated_text = getattr(response, "text", None)

    if not generated_text or not generated_text.strip():
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return generated_text.strip()


def _generate_with_groq(
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
                    "Follow the user's instructions and do not invent "
                    "information that is not present in the supplied context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    if not completion.choices:
        raise RuntimeError(
            "Groq returned no completion choices."
        )

    generated_text = completion.choices[0].message.content

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
    Generate content with Gemini first and automatically use Groq
    when Gemini is unavailable or its quota has been exhausted.

    Parameters:
        prompt:
            Complete prompt sent by qa_chain, summary_generator or
            notes_generator.

        model_name:
            Gemini model ID.

        groq_model_name:
            Groq fallback model ID.

    Returns:
        Generated response text.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "The AI prompt cannot be empty."
        )

    cleaned_prompt = prompt.strip()
    gemini_error = None
    groq_error = None

    # --------------------------------------------------
    # Primary provider: Gemini
    # --------------------------------------------------
    if get_google_api_key():
        try:
            return _generate_with_gemini(
                prompt=cleaned_prompt,
                model_name=model_name,
            )
        except Exception as error:
            gemini_error = error

            # Content or prompt errors should not be silently hidden.
            # Service, quota and authentication errors can use Groq.
            if not _is_retryable_gemini_error(error):
                raise RuntimeError(
                    f"Gemini could not generate content: {error}"
                ) from error
    else:
        gemini_error = RuntimeError(
            "GOOGLE_API_KEY is not configured."
        )

    # --------------------------------------------------
    # Fallback provider: Groq
    # --------------------------------------------------
    if get_groq_api_key():
        try:
            return _generate_with_groq(
                prompt=cleaned_prompt,
                model_name=groq_model_name,
            )
        except Exception as error:
            groq_error = error
    else:
        groq_error = RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    raise RuntimeError(
        "Both AI providers were unavailable. "
        f"Gemini error: {gemini_error}. 
"f"Groq error: {groq_error}."
    )