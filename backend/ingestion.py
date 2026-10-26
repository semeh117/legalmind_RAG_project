import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """step one: Extract text from the PDF document."""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        pages.append(
            {"page_number": page_num + 1, "text": page.get_text()}
        )
    return pages

"""step two: Split the extracted text into smaller chunks."""

# Chunks shorter than this (signature blocks, page footers) carry almost no
# content but embed close to every question, crowding out real clauses.
MIN_CHUNK_LENGTH = 200

def split_into_chunks(pages : list[dict]):
    splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],       
    chunk_size=1000,
    chunk_overlap=100,
    )
    chunks = []
    for page in pages:
        page_chunks = [
            c for c in splitter.split_text(page["text"])
            if len(c.strip()) >= MIN_CHUNK_LENGTH
        ]

        for i, chunk in enumerate(page_chunks):
            chunks.append(
                {"page_number": page["page_number"], "chunk_number": i + 1, "text": chunk}
            )
    return chunks
