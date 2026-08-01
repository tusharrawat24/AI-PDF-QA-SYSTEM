"""
Study notes generation using the Gemini API.
"""

from helpers.gemini_client import generate_content


def generate_study_notes(document_text: str) -> str:
    """
    Generate structured study notes from PDF content.
    """

    if not document_text or not document_text.strip():
        raise ValueError("The PDF text cannot be empty.")

    prompt = f"""
You are an AI Study Notes Generator.

Create detailed, structured, and easy-to-revise study notes
using only the PDF content provided below.

Instructions:
1. Start with a clear title.
2. Divide the notes into meaningful topics and subtopics.
3. Include important definitions.
4. Explain key concepts in simple language.
5. Use bullet points wherever helpful.
6. Include examples only when they are available in the PDF.
7. Add important differences or comparisons when relevant.
8. Add a section titled "Important Points for Revision".
9. Add a section titled "Possible Exam Questions".
10. Do not invent information that is not present in the PDF.
11. Keep the notes detailed, organized, and suitable for exam preparation.

PDF Content:
{document_text}

Study Notes:
"""

    try:
        notes_text = generate_content(prompt)

        if not notes_text:
            return "No study notes were received from Gemini."

        return notes_text

    except Exception as error:
        raise RuntimeError(
            f"Gemini could not generate study notes: {error}"
        ) from error
