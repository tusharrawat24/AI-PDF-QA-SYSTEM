"""
Shared configuration and client for the Gemini API.

Keeps the API key loading and rate-limit retry logic in a single
place so every Gemini-powered feature behaves consistently.
"""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ------------------------------------------------------------------
# .env discovery
#
# The API key can live in helpers/.env (the canonical location for
# this project) or in the project root. helpers/.env is checked
# first so a freshly updated key there always wins. override=True
# also makes sure the .env value beats any stale GOOGLE_API_KEY that
# may already be set in the shell/process environment.
# ------------------------------------------------------------------
ENV_FILE_CANDIDATES = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]

loaded_env_file = None

for env_file in ENV_FILE_CANDIDATES:
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
        loaded_env_file = env_file
        break

if loaded_env_file is None:
    raise FileNotFoundError(
        "No .env file was found. Looked for: "
        + ", ".join(str(path) for path in ENV_FILE_CANDIDATES)
    )

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        f"GOOGLE_API_KEY was not found in {loaded_env_file}."
    )

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# ------------------------------------------------------------------
# Rate-limit retry settings
# ------------------------------------------------------------------
RETRY_LIMIT = 2
RETRY_FALLBACK_SECONDS = 30.0
RETRY_HARD_CAP_SECONDS = 60.0

client = genai.Client(api_key=API_KEY)


def _is_rate_limit_error(error_message: str) -> bool:
    """True for Gemini 429 RESOURCE_EXHAUSTED responses."""

    return "429" in error_message or "RESOURCE_EXHAUSTED" in error_message


def _extract_retry_delay(error_message: str) -> float:
    """Best-effort parse of the retry delay from a 429 error message."""

    patterns = (
        r"Please retry in\s+([\d.]+)s",
        r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"',
    )

    for pattern in patterns:
        match = re.search(pattern, error_message)

        if match:
            return float(match.group(1))

    return RETRY_FALLBACK_SECONDS


def generate_content(prompt: str) -> str:
    """
    Generate text from Gemini, retrying on 429 rate-limit errors.

    Returns the trimmed response text, or an empty string when Gemini
    returns no content. Raises RuntimeError when Gemini still fails
    after all retries.
    """

    if not prompt or not prompt.strip():
        raise ValueError("The prompt cannot be empty.")

    attempt = 0

    while True:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            try:
                return response.text.strip()
            except (AttributeError, ValueError):
                return ""

        except Exception as error:
            error_message = str(error)

            should_retry = (
                _is_rate_limit_error(error_message)
                and attempt < RETRY_LIMIT
            )

            if not should_retry:
                raise RuntimeError(
                    "Gemini could not generate content "
                    f"(model: {GEMINI_MODEL}). "
                    f"Details: {error_message}"
                ) from error

            retry_delay = min(
                _extract_retry_delay(error_message),
                RETRY_HARD_CAP_SECONDS,
            )

            attempt += 1

            time.sleep(retry_delay)
