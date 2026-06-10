# DECISIONS.md — Synapse

Engineering decisions made during development, with rationale.
This separates "followed a tutorial" from "made deliberate engineering choices."

---

## 1. Python 3.11.9 over 3.13

**Decision:** Install Python 3.11.9 alongside system Python 3.13, create venv with 3.11.

**Why:** ChromaDB and pdfplumber had compatibility issues with Python 3.13 at time of development. 3.11 is the production-stable sweet spot — fully supported by every Gen AI library and what most production systems target.

**Tradeoff:** Slightly older Python, but zero dependency conflicts. Worth it.

---

## 2. Google Gemini for embeddings over HuggingFace

**Decision:** Use `gemini-embedding-001` via `google-genai` SDK instead of `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Inference API.

**Why:** HuggingFace Inference API endpoint (`api-inference.huggingface.co`) is geo-blocked by some Indian ISPs. Confirmed via `nslookup` — even Google DNS returned no IP address. Gemini is unrestricted, free at 1500 requests/day, and the `gemini-embedding-001` model produces 768-dimensional vectors vs HuggingFace's 384 — better semantic accuracy.

**Tradeoff:** +200ms latency per embedding call vs local model. Acceptable since embeddings only happen at upload time, not at query time.

---

## 3. google-genai SDK over google-generativeai

**Decision:** Use `google-genai` package instead of `google-generativeai`.

**Why:** `google-generativeai` was officially deprecated and stopped receiving updates. The new `google-genai` package uses a different client API (`client.models.embed_content()` vs `genai.embed_content()`). Staying on deprecated packages is a production risk.

**Migration:** Updated embedder.py to use `from google import genai` and `genai.Client(api_key=...)`.

---

## 4. ChromaDB 0.4.24 over 0.5.15

**Decision:** Downgraded ChromaDB from 0.5.15 to 0.4.24.

**Why:** ChromaDB 0.5.x has a broken telemetry system that floods the console with `capture() takes 1 positional argument but 3 were given` on every collection access. The env var suppression (`ANONYMIZED_TELEMETRY=False`) doesn't work reliably on Windows because ChromaDB initializes its telemetry at import time. 0.4.24 has the same functionality without the broken telemetry code.

**Tradeoff:** Older version, but the vector storage API is identical and stable.

---

## 5. Manual agent loop over LangChain/LlamaIndex

**Decision:** Build the entire agent loop manually in `agent.py` without any orchestration framework.

**Why:** In technical interviews, you will be asked to explain your agent loop line by line. If you used LangChain you can't. Every step — planning, retrieval, generation, evaluation, retry — is explicit Python that can be explained, debugged, and modified without framework knowledge. This is also more deployable: no framework version conflicts, smaller dependency footprint, and full control over retry logic.

**Tradeoff:** More code to write. Worth it for interview credibility and production control.

---

## 6. ChromaDB in-process over Pinecone/Weaviate

**Decision:** Use ChromaDB's `PersistentClient` running in-process instead of a managed vector database.

**Why:** Render free tier has no external networking budget and charges for external service calls. ChromaDB in-process uses ~50MB RAM, persists to disk automatically, and requires zero external dependencies. At scale I would migrate to Pinecone or Weaviate — documented here as the known next step.

**Tradeoff:** No horizontal scaling, data lives on the Render instance disk. Fine for a portfolio project and early-stage production.

---

## 7. Session-per-user isolation in ChromaDB

**Decision:** Each user session gets its own ChromaDB collection (`session_{session_id}`).

**Why:** Without isolation, User A's uploaded documents would appear in User B's search results. This is a data privacy requirement, not just a nice-to-have. Collections are cleaned up after 24 hours on startup to manage disk usage.

**Implementation:** `get_or_create_collection()` with session ID in the collection name. Sanitized to alphanumeric + underscores to meet ChromaDB naming rules.

---

## 8. get_or_create_collection() over get → except → create pattern

**Decision:** Use ChromaDB's built-in `get_or_create_collection()` instead of `try: get_collection() except: create_collection()`.

**Why:** `get_collection()` throws `InvalidCollectionException` when the collection doesn't exist instead of returning None. The exception-based pattern is fragile — it catches all exceptions including legitimate errors. The built-in method is atomic, cleaner, and the correct ChromaDB API.

---

## 9. Cosine similarity over L2 distance for vector search

**Decision:** Configure ChromaDB collections with `hnsw:space: cosine`.

**Why:** L2 (Euclidean) distance measures raw geometric distance between vectors. Cosine similarity measures the angle between vectors — which captures semantic direction regardless of magnitude. For text embeddings, direction of meaning matters more than raw distance. A long detailed paragraph and a short sentence about the same topic should be similar — cosine captures this, L2 doesn't.

---

## 10. 768 dimensions capped from Gemini's 3072 max

**Decision:** Set `output_dimensionality=768` when calling Gemini embedding API instead of using the full 3072.

**Why:** `gemini-embedding-001` supports Matryoshka Representation Learning — you can truncate the output vector to any size and still get meaningful embeddings. 768 dimensions gives excellent semantic accuracy while keeping ChromaDB index size small and search fast. 3072 dimensions would be 4x the storage and search cost with marginal accuracy improvement for our use case.

---

## 11. extract_words() over extract_text() in pdfplumber

**Decision:** Use `page.extract_words(use_text_flow=True)` instead of `page.extract_text()` for PDF extraction.

**Why:** Some arXiv PDFs generated with LaTeX store characters without space metadata. `extract_text()` merges words together: "Providedproperattribution". `extract_words()` rebuilds text from bounding boxes, preserving word spacing. `use_text_flow=True` handles multi-column research paper layouts correctly.

**Remaining limitation:** Heavily encoded LaTeX PDFs may still have some word merging. The LLM handles merged text correctly as it was trained on noisy data. Documented as accepted tradeoff — no additional OCR library added to keep RAM under 512MB.

---

## 12. 0.7s delay between Gemini embedding calls

**Decision:** Add `time.sleep(0.7)` between every individual Gemini embedding API call.

**Why:** Gemini free tier allows 100 requests/minute for the embedding model. At 0.7s per call that's ~85 requests/minute — safely under the limit. Without the delay, batch embedding of large PDFs (50+ pages) hits the rate limit after ~15 chunks, causing failures and partial embeddings that crash ChromaDB's `ids must match embeddings count` validation.

**Tradeoff:** Upload of a 50-page PDF takes ~35 seconds instead of ~5 seconds. Acceptable since this is a one-time cost per document.

---

## 13. llm_survey.pdf removed from pre-loaded sample docs

**Decision:** Removed `llm_survey.pdf` (417 chunks) from `config.SAMPLE_DOCS` pre-load list.

**Why:** 417 embedding API calls on every fresh startup exhausts Gemini's daily quota (1500 req/day) in a single startup, leaving no budget for actual user queries. Pre-loading is meant for instant demo, not comprehensive indexing. The file remains in `sample_documents/` and can be uploaded manually.

**Kept:** `attention_is_all_you_need.pdf` (34 chunks) + `rag_paper.pdf` (19 chunks) = 53 total, loads in ~40 seconds.

---

## 14. Self-ping keep-alive over paid Render tier

**Decision:** Implement 3-layer keep-alive strategy instead of upgrading to Render paid tier.

**Why:** Render free tier sleeps after 15 minutes of inactivity causing 40-50 second cold starts. The 3-layer strategy solves this at $0 cost:
- Layer 1: UptimeRobot (free) pings `/health` every 14 minutes externally
- Layer 2: FastAPI async task self-pings `/health` every 14 minutes internally
- Layer 3: Frontend shows warm-up UI and polls until backend is ready

This is sufficient for a portfolio/demo context. Production would use paid tier or a dedicated uptime service.

---

## 15. SSE streaming over polling for chat responses

**Decision:** Implement `/chat` as Server-Sent Events (SSE) streaming endpoint instead of a regular JSON response.

**Why:** The agent loop takes 3-8 seconds (multiple LLM calls for planning, generation, evaluation). Without streaming, the user sees a blank loading state for the full duration. With SSE, status updates flow in real-time: "Planning..." → "Retrieving..." → "Generating..." → "Evaluating...". This dramatically improves perceived performance and lets the user understand what the agent is doing at each step.
