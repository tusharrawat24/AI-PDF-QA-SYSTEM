from helpers.gemini_client import client


def generate_pdf_summary(document_text: str) -> str:
    """
    Generate a detailed summary using the shared Gemini client.
    """

    if not document_text or not document_text.strip():
        raise ValueError("Document text cannot be empty.")

    prompt = f"""
You are an expert document summarization assistant.

Generate a detailed and well-structured summary of the PDF content.

Instructions:
1. Include the main concepts and important details.
2. Use clear headings and subheadings.
3. Use bullet points where helpful.
4. Do not introduce information outside the document.
5. Keep the explanation easy to understand.
6. Include a document overview and important revision points.

PDF Content:

{document_text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty summary."
            )

        return response.text.strip()

    except Exception as error:
        raise RuntimeError(
            f"Gemini could not generate the PDF summary: {error}"
        ) from error