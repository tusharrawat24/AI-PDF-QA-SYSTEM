"""
Study notes generation using the shared AI layer.

This module automatically uses:
1. Google Gemini as the primary provider.
2. Groq as the fallback provider when Gemini quota or service errors occur.
"""

from helpers.gemini_client import generate_content


def generate_study_notes(document_text: str) -> str:
    """
    Generate detailed and well-structured study notes from PDF text.
    """

    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError(
            "Document text cannot be empty."
        )

    prompt = f"""
You are NeuraDocs, an expert academic study-notes assistant.

Create detailed and well-structured study notes using ONLY the PDF content
provided below.

Instructions:
1. Begin with a short document overview.
2. Use clear headings and subheadings.
3. Explain important definitions and concepts.
4. Use bullet points for key facts.
5. Include comparisons wherever appropriate.
6. Add examples only when they are present in the PDF.
7. Include a section titled "Important Revision Points".
8. Add a section titled "Possible Exam Questions" at the end.
9. Do not introduce information outside the supplied PDF.
10. Keep the language clear and suitable for university students.
11. Avoid unnecessary repetition.

PDF Content:

{document_text}

Study Notes:
"""

    try:
        notes_text = generate_content(prompt)

        if not notes_text or not notes_text.strip():
            raise RuntimeError(
                "The AI provider returned empty study notes."
            )

        return notes_text.strip()

    except Exception as error:
        raise RuntimeError(
            f"Unable to generate study notes: {error}"
        ) from error