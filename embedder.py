import time
import config

# Suppress ChromaDB telemetry at import time
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# ─── Embedding Dimensions ────────────────────────────────────
EMBEDDING_DIMENSIONS = 768


# ─── Core Functions ──────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Main function. Takes a list of strings, returns list of embedding vectors.
    Each vector is a list of 768 floats.
    """
    if not texts:
        return []

    if config.EMBED_MODE == "gemini":
        return _embed_via_gemini(texts)
    else:
        return _embed_local(texts)


def embed_single(text: str) -> list[float]:
    """Embeds a single string. Used by agent.py for query embedding."""
    results = embed_texts([text])
    return results[0] if results else []


# ─── Gemini API Embedding ────────────────────────────────────

def _embed_via_gemini(texts: list[str]) -> list[list[float]]:
    """
    Calls Google Gemini API to generate embeddings.

    Rate limit strategy:
    - Gemini free tier: 1500 req/day, 100 req/minute
    - We process ONE text at a time with delays between calls
    - Small delay after every call prevents rate limit bursts
    - On rate limit: exponential backoff up to 5 retries
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    all_embeddings = []
    total = len(texts)

    for i, text in enumerate(texts):
        embedding = _embed_one_with_retry(client, types, text)
        all_embeddings.append(embedding)

        # Progress indicator for large batches
        if total > 10 and (i + 1) % 10 == 0:
            print(f"   Embedded {i + 1}/{total} chunks...")

        # Polite delay between every call — prevents rate limit bursts
        # 0.7s = ~85 requests/minute, safely under 100/min limit
        if i < total - 1:
            time.sleep(0.7)

    return all_embeddings


def _embed_one_with_retry(
    client,
    types,
    text: str,
    max_retries: int = 5,
    base_delay: float = 3.0
) -> list[float]:
    """
    Embeds a single text with exponential backoff retry.

    Retry schedule on rate limit:
    Attempt 1 fail → wait 3s
    Attempt 2 fail → wait 6s
    Attempt 3 fail → wait 12s
    Attempt 4 fail → wait 24s
    Attempt 5 fail → raise error
    """
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSIONS
                )
            )
            return result.embeddings[0].values

        except Exception as e:
            error_str = str(e).lower()

            if "api key" in error_str or "401" in error_str:
                raise ValueError("Gemini API key invalid. Check GEMINI_API_KEY in .env")

            if attempt < max_retries - 1:
                # Exponential backoff
                wait = base_delay * (2 ** attempt)
                if "quota" in error_str or "rate" in error_str or "429" in error_str:
                    print(f"   Gemini rate limit — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue

            raise RuntimeError(f"Gemini embedding failed after {max_retries} attempts: {e}")

    raise RuntimeError("Embedding failed — max retries exceeded")


# ─── Local Embedding (Development Fallback) ──────────────────

def _embed_local(texts: list[str]) -> list[list[float]]:
    """
    Local fallback. Only when EMBED_MODE=local.
    WARNING: ~400MB RAM — DO NOT use on Render free tier.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    except ImportError:
        raise ImportError("Set EMBED_MODE=gemini in .env for production use.")


# ─── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  {config.APP_NAME} — Embedder Test")
    print(f"{'='*50}\n")
    print(f"  Mode:  {config.EMBED_MODE}")
    print(f"  Model: {config.EMBED_MODEL}\n")

    test_texts = [
        "The transformer architecture uses self-attention mechanisms.",
        "Attention allows the model to focus on relevant parts of input.",
        "The recipe requires two cups of flour and one egg."
    ]

    print(f"  Testing {len(test_texts)} texts...\n")

    try:
        embeddings = embed_texts(test_texts)
        print(f"✅ Count: {len(embeddings)} vectors")
        print(f"   Dims:  {len(embeddings[0])}")

        def cosine_sim(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            mag_a = sum(x**2 for x in a) ** 0.5
            mag_b = sum(x**2 for x in b) ** 0.5
            return dot / (mag_a * mag_b) if mag_a and mag_b else 0

        sim_12 = cosine_sim(embeddings[0], embeddings[1])
        sim_13 = cosine_sim(embeddings[0], embeddings[2])
        print(f"\n   Similar pair:   {sim_12:.3f}")
        print(f"   Different pair: {sim_13:.3f}")
        print(f"\n✅ {'PASS' if sim_12 > sim_13 else 'UNEXPECTED'}")

    except Exception as e:
        print(f"❌ Error: {e}")
