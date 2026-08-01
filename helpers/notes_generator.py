from helpers.gemini_client import generate_content


def generate_study_notes(document_text: str) -> str:
    """
    Generate detailed study notes from PDF text.
    """

    if (
        not isinstance(document_text, str)
        or not document_text.strip()
    ):
        raise ValueError(
            "Document text cannot be empty."
        )

    prompt = f"""
You are an expert academic study-notes assistant.

Create detailed and well-structured study notes using only
the PDF content provided below.

Instructions:

1. Begin with a short document overview.
2. Use clear headings and subheadings.
3. Explain important definitions.
4. Use bullet points for key facts.
5. Include comparisons wherever appropriate.
6. Include important revision points.
7. Add possible exam questions at the end.
8. Do not introduce information outside the PDF.
9. Keep the language clear and suitable for university students.

PDF Content:

{document_text}
"""

    try:
        return generate_content(prompt)

    except Exception as error:
        raise RuntimeError(
            f"Gemini could not generate study notes: {error}"
        ) from error