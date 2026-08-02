"""
PDF summary generation using the shared AI layer.

This module automatically uses:
1. Google Gemini as the primary provider.
2. Groq as the fallback provider when Gemini quota or service errors occur.
"""

from helpers.gemini_client import generate_content


def generate_pdf_summary(document_text: str) -> str:
    """
    Generate a detailed and well-structured summary of the PDF content.
    """

    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError("Document text cannot be empty.")

    prompt = f"""
You are NeuraDocs, an expert document summarization assistant.

Generate a detailed and well-structured summary using ONLY the PDF content
provided below.

Instructions:
1. Begin with a short document overview.
2. Include all major concepts and important supporting details.
3. Use clear headings and subheadings.
4. Use bullet points where they improve readability.
5. Keep the language simple and suitable for university students.
6. Add a section titled "Important Revision Points".
7. Do not introduce information that is not present in the PDF.
8. Avoid unnecessary repetition.

PDF Content:

{document_text}

Summary:
"""

    try:
        summary_text = generate_content(prompt)

        if not summary_text or not summary_text.strip():
            raise RuntimeError(
                "The AI provider returned an empty summary."
            )

        return summary_text.strip()

    except Exception as error:
        raise RuntimeError(
            f"Unable to generate the PDF summary: {error}"
        ) from error