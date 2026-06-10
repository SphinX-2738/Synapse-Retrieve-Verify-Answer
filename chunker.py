import nltk
import re
from typing import Optional
import config
from pdf_processor import ProcessedDocument, PageContent

# Download nltk sentence tokenizer on first run
# punkt is the model that detects sentence boundaries
# quietly=True suppresses download messages after first time
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── Data Contract ───────────────────────────────────────────
# This is the exact shape every downstream file expects
# vector_store.py, agent.py, and evaluator.py all depend on this
class Chunk:
    def __init__(
        self,
        chunk_id: str,        # unique ID: "filename_p4_chunk_2"
        text: str,            # the actual chunk text
        page: int,            # page number (for citations)
        filename: str,        # source PDF filename
        word_count: int,      # words in this chunk
        total_chunks: int,    # total chunks in this document
        chunk_index: int      # position of this chunk (0-indexed)
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.page = page
        self.filename = filename
        self.word_count = word_count
        self.total_chunks = total_chunks
        self.chunk_index = chunk_index

    def to_dict(self) -> dict:
        """
        Converts chunk to dict for ChromaDB storage.
        ChromaDB stores metadata as flat dicts — no nested objects.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page": self.page,
            "filename": self.filename,
            "word_count": self.word_count,
            "total_chunks": self.total_chunks,
            "chunk_index": self.chunk_index
        }


# ─── Core Functions ──────────────────────────────────────────

def chunk_document(doc: ProcessedDocument) -> list[Chunk]:
    """
    Main function. Takes a ProcessedDocument, returns list of Chunks.
    Called by: vector_store.py after pdf_processor.py extracts pages.

    Strategy:
    1. Split each page into sentences using nltk
    2. Group sentences into chunks of ~CHUNK_SIZE words
    3. Add CHUNK_OVERLAP words from previous chunk at the start
    4. Never cut mid-sentence
    5. Attach page number to every chunk for citations
    """
    all_chunks = []
    chunk_index = 0

    for page in doc.pages:
        # Skip pages with no meaningful content
        if page.word_count < 10:
            continue

        # Split page text into sentences
        sentences = _split_into_sentences(page.text)

        if not sentences:
            continue

        # Group sentences into chunks
        page_chunks = _group_sentences_into_chunks(
            sentences=sentences,
            page_number=page.page_number,
            filename=doc.filename,
            chunk_index_start=chunk_index
        )

        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    # Now we know total_chunks — update each chunk
    total = len(all_chunks)
    for chunk in all_chunks:
        chunk.total_chunks = total

    return all_chunks


def _split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using nltk's punkt tokenizer.

    Why nltk over simple split on '.'?
    - Handles abbreviations: "Dr. Smith" doesn't split after "Dr."
    - Handles decimals: "0.6 threshold" doesn't split after "0.6"
    - Handles edge cases: ellipsis, quotes, parentheses
    """
    try:
        sentences = nltk.sent_tokenize(text)
        # Filter out empty or very short sentences (artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    except Exception:
        # Fallback: split on period+space if nltk fails
        return [s.strip() for s in text.split('. ') if len(s.strip()) > 10]


def _group_sentences_into_chunks(
    sentences: list[str],
    page_number: int,
    filename: str,
    chunk_index_start: int
) -> list[Chunk]:
    """
    Groups sentences into chunks respecting CHUNK_SIZE and CHUNK_OVERLAP.

    Overlap strategy:
    - After filling a chunk, carry the last OVERLAP words into the next chunk
    - This ensures sentences at boundaries appear in at least one full chunk
    - Prevents losing context when a key concept spans a chunk boundary
    """
    chunks = []
    current_sentences = []
    current_word_count = 0
    overlap_sentences = []    # sentences carried over from previous chunk
    local_index = chunk_index_start

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence exceeds CHUNK_SIZE, finalize current chunk
        if current_word_count + sentence_words > config.CHUNK_SIZE and current_sentences:
            # Build chunk text
            chunk_text = " ".join(current_sentences)

            chunk = Chunk(
                chunk_id=_build_chunk_id(filename, page_number, local_index),
                text=chunk_text,
                page=page_number,
                filename=filename,
                word_count=current_word_count,
                total_chunks=0,      # updated after all chunks are collected
                chunk_index=local_index
            )
            chunks.append(chunk)
            local_index += 1

            # Calculate overlap: carry last N words into next chunk
            overlap_sentences = _get_overlap_sentences(
                current_sentences,
                config.CHUNK_OVERLAP
            )

            # Start new chunk with overlap
            current_sentences = overlap_sentences.copy()
            current_word_count = sum(len(s.split()) for s in current_sentences)

        # Add sentence to current chunk
        current_sentences.append(sentence)
        current_word_count += sentence_words

    # Don't forget the last chunk (sentences that didn't fill a full chunk)
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk = Chunk(
            chunk_id=_build_chunk_id(filename, page_number, local_index),
            text=chunk_text,
            page=page_number,
            filename=filename,
            word_count=current_word_count,
            total_chunks=0,
            chunk_index=local_index
        )
        chunks.append(chunk)

    return chunks


def _get_overlap_sentences(sentences: list[str], overlap_words: int) -> list[str]:
    """
    Returns the last N sentences that together contain ~overlap_words words.
    Used to create overlap between consecutive chunks.
    """
    overlap = []
    word_count = 0

    # Walk backwards through sentences until we hit overlap_words
    for sentence in reversed(sentences):
        words = len(sentence.split())
        if word_count + words > overlap_words:
            break
        overlap.insert(0, sentence)
        word_count += words

    return overlap


def _build_chunk_id(filename: str, page: int, index: int) -> str:
    """
    Builds a unique, readable chunk ID.
    Format: "bert_paper_p3_chunk_5"

    Why readable IDs?
    - Easier to debug in ChromaDB
    - Interviewers can see the structure immediately
    - Cited in answers as source reference
    """
    # Remove extension and replace spaces/dots with underscores
    base = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
    base = re.sub(r'[\s\.\-]+', '_', base)
    return f"{base}_p{page}_chunk_{index}"


def get_chunk_stats(chunks: list[Chunk]) -> dict:
    """
    Returns stats about the chunking result.
    Called by main.py to include in upload response.
    """
    if not chunks:
        return {"total_chunks": 0, "avg_words": 0, "min_words": 0, "max_words": 0}

    word_counts = [c.word_count for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_words": round(sum(word_counts) / len(word_counts)),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "pages_covered": len(set(c.page for c in chunks))
    }


# ─── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: python chunker.py path/to/file.pdf
    Tests the full pipeline: pdf_processor → chunker
    """
    import sys
    from pdf_processor import extract_pdf

    print(f"\n{'='*50}")
    print(f"  {config.APP_NAME} — Chunker Test")
    print(f"{'='*50}\n")

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        print("Usage: python chunker.py path/to/your.pdf")
        sys.exit(0)

    # Step 1: Extract PDF
    print(f"Step 1: Extracting PDF...")
    doc = extract_pdf(test_path)

    if doc.error:
        print(f"❌ PDF extraction failed: {doc.error}")
        sys.exit(1)

    print(f"✅ Extracted {doc.total_pages} pages, {doc.total_words} words\n")

    # Step 2: Chunk document
    print(f"Step 2: Chunking document...")
    print(f"   Chunk size:    {config.CHUNK_SIZE} words")
    print(f"   Chunk overlap: {config.CHUNK_OVERLAP} words\n")

    chunks = chunk_document(doc)
    stats = get_chunk_stats(chunks)

    print(f"✅ Chunking complete!")
    print(f"   Total chunks:  {stats['total_chunks']}")
    print(f"   Avg words:     {stats['avg_words']}")
    print(f"   Min words:     {stats['min_words']}")
    print(f"   Max words:     {stats['max_words']}")
    print(f"   Pages covered: {stats['pages_covered']}")

    # Preview first 3 chunks
    print(f"\n   First 3 chunks preview:")
    for chunk in chunks[:3]:
        preview = chunk.text[:120] + "..." if len(chunk.text) > 120 else chunk.text
        print(f"\n   [{chunk.chunk_id}]")
        print(f"   Page {chunk.page} | {chunk.word_count} words")
        print(f"   {preview}")
