import pdfplumber
import os
import re
from dataclasses import dataclass
from typing import Optional
import config

# ─── Data Contract ───────────────────────────────────────────
# Every downstream file (chunker, vector_store) expects this exact shape
@dataclass
class PageContent:
    page_number: int        # 1-indexed (human readable)
    text: str               # extracted text for this page
    word_count: int         # used to detect scanned pages
    filename: str           # original PDF filename

@dataclass
class ProcessedDocument:
    filename: str           # e.g. "attention_is_all_you_need.pdf"
    total_pages: int        # total pages in PDF
    pages: list[PageContent] # list of PageContent objects
    is_scanned: bool        # True if PDF has no extractable text
    total_words: int        # total word count across all pages
    error: Optional[str]    # None if successful, error message if failed


# ─── Core Functions ──────────────────────────────────────────

def extract_pdf(file_path: str) -> ProcessedDocument:
    """
    Main function. Takes a PDF file path, returns ProcessedDocument.
    Called by: vector_store.py when user uploads a PDF.

    Why pdfplumber over PyPDF2 or fitz?
    - Better accuracy on research papers with complex layouts
    - Handles multi-column text better
    - Native page-by-page extraction (critical for citations)

    Why extract_words() over extract_text()?
    - Rebuilds text from bounding boxes → preserves proper spacing
    - Fixes word-merging issue common in LaTeX/arXiv PDFs
    - use_text_flow=True handles multi-column research paper layouts
    - Zero extra dependencies or RAM cost
    """
    filename = os.path.basename(file_path)

    # Guard: file must exist
    if not os.path.exists(file_path):
        return ProcessedDocument(
            filename=filename,
            total_pages=0,
            pages=[],
            is_scanned=False,
            total_words=0,
            error=f"File not found: {file_path}"
        )

    try:
        pages = []

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                # extract_words() rebuilds text from bounding boxes
                # preserves spacing far better than extract_text() for LaTeX PDFs
                words = page.extract_words(
                    x_tolerance=3,        # horizontal gap tolerance between words
                    y_tolerance=3,        # vertical gap tolerance (same line)
                    keep_blank_chars=False,
                    use_text_flow=True    # follows reading order, handles columns
                )
                raw_text = " ".join(w["text"] for w in words) if words else ""

                # Clean the text if extraction succeeded
                if raw_text:
                    cleaned = _clean_text(raw_text)
                else:
                    cleaned = ""

                word_count = len(cleaned.split()) if cleaned else 0

                pages.append(PageContent(
                    page_number=i + 1,   # convert 0-index to 1-index
                    text=cleaned,
                    word_count=word_count,
                    filename=filename
                ))

        # Detect scanned PDF: if >80% of pages have <10 words
        # A real text PDF will always have substantial text per page
        # Documented tradeoff: scanned PDFs need OCR (not in scope v1.0)
        scanned_pages = sum(1 for p in pages if p.word_count < 10)
        is_scanned = (scanned_pages / total_pages) > 0.8 if total_pages > 0 else False

        total_words = sum(p.word_count for p in pages)

        return ProcessedDocument(
            filename=filename,
            total_pages=total_pages,
            pages=pages,
            is_scanned=is_scanned,
            total_words=total_words,
            error=None
        )

    except Exception as e:
        return ProcessedDocument(
            filename=filename,
            total_pages=0,
            pages=[],
            is_scanned=False,
            total_words=0,
            error=str(e)
        )


def _clean_text(text: str) -> str:
    """
    Cleans raw extracted text from pdfplumber.
    extract_words() already handles spacing, so this just
    normalizes whitespace and removes junk characters.
    """
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Normalize whitespace: collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize line endings
    text = re.sub(r'\r\n|\r', '\n', text)

    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace single newlines with space (mid-paragraph line breaks)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Final cleanup
    text = text.strip()

    return text


def get_pdf_summary(doc: ProcessedDocument) -> dict:
    """
    Returns a summary dict for the API response.
    Called by main.py after processing upload.
    """
    return {
        "filename": doc.filename,
        "total_pages": doc.total_pages,
        "total_words": doc.total_words,
        "is_scanned": doc.is_scanned,
        "status": "error" if doc.error else "success",
        "error": doc.error,
        "pages_with_content": sum(1 for p in doc.pages if p.word_count > 10)
    }


# ─── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: python pdf_processor.py path/to/file.pdf
    """
    import sys

    print(f"\n{'='*50}")
    print(f"  {config.APP_NAME} — PDF Processor Test")
    print(f"{'='*50}\n")

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        print("Usage: python pdf_processor.py path/to/your.pdf")
        print("\nNo PDF path provided.")
        sys.exit(0)

    print(f"Processing: {test_path}\n")
    result = extract_pdf(test_path)

    if result.error:
        print(f"❌ Error: {result.error}")
    else:
        print(f"✅ Success!")
        print(f"   Filename:           {result.filename}")
        print(f"   Total pages:        {result.total_pages}")
        print(f"   Total words:        {result.total_words}")
        print(f"   Pages with content: {sum(1 for p in result.pages if p.word_count > 10)}")
        print(f"   Is scanned:         {result.is_scanned}")
        print(f"\n   First 3 pages preview:")
        for page in result.pages[:3]:
            preview = page.text[:150] + "..." if len(page.text) > 150 else page.text
            print(f"\n   Page {page.page_number} ({page.word_count} words):")
            print(f"   {preview}")
