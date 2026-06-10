import json
import re
import time
from groq import Groq
from typing import Optional, Generator
import config
from vector_store import search, list_documents
from memory import get_history, add_turn

# ─── Groq Client ─────────────────────────────────────────────
_groq_client = None

def _get_groq() -> Groq:
    """
    Returns initialized Groq client.
    Lazy initialization — only created on first call.
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


# ─── Step 1: PLAN ────────────────────────────────────────────

def plan(question: str, session_id: str) -> dict:
    """
    Step 1 of the agentic loop.
    LLM decides: search documents OR answer from general knowledge.

    Why this makes it an agent:
    - A pipeline always retrieves. An agent DECIDES whether to retrieve.
    - "What is 2+2?" → answer_directly (no docs needed)
    - "What does the paper say about attention?" → search (docs needed)

    Returns:
    {
        "action": "search" | "answer_directly",
        "reasoning": "one sentence explanation"
    }
    """
    # Get list of uploaded documents for context
    docs = list_documents(session_id)
    doc_names = [d["filename"] for d in docs] if docs else []
    doc_list_str = ", ".join(doc_names) if doc_names else "no documents uploaded"

    # Get recent conversation history for context
    history = get_history(session_id)
    history_str = _format_history_for_prompt(history)

    plan_prompt = f"""You are a research assistant with access to uploaded documents.

Available documents: {doc_list_str}

Recent conversation:
{history_str}

Question: {question}

Decide whether to search the uploaded documents or answer from general knowledge.

Rules:
- Choose "search" if the question is about the uploaded documents or their content
- Choose "search" if the question asks about specific papers, findings, or details
- Choose "answer_directly" if it's a general knowledge question unrelated to the docs
- Choose "answer_directly" if no documents are uploaded
- Choose "answer_directly" for greetings, meta questions, or clarifications

Respond ONLY with valid JSON, no other text:
{{"action": "search" or "answer_directly", "reasoning": "one sentence"}}"""

    response = _call_llm_with_retry(
        messages=[{"role": "user", "content": plan_prompt}],
        temperature=0.0,    # zero temperature for deterministic decisions
        max_tokens=150
    )

    return _parse_plan_response(response)


def _parse_plan_response(response: str) -> dict:
    """
    Parses LLM plan response into structured dict.
    Handles edge cases where LLM adds extra text around the JSON.

    Fallback: if parsing fails, default to "search" —
    safer to over-retrieve than miss relevant content.
    """
    try:
        # Try direct parse first
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    try:
        # Try extracting JSON from response text
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: infer from response text
    if "answer_directly" in response.lower():
        return {"action": "answer_directly", "reasoning": "Inferred from response"}

    # Default to search — safer fallback
    return {"action": "search", "reasoning": "Defaulting to search for safety"}


# ─── Step 2: RETRIEVE ────────────────────────────────────────

def retrieve(question: str, session_id: str) -> dict:
    """
    Step 2 of the agentic loop.
    Searches ChromaDB for relevant chunks.

    Retry logic:
    - If no results above threshold, rephrase query and try again
    - Rephrasing uses LLM to generate alternative formulations
    - Max 2 retries before giving up and returning empty

    Returns:
    {
        "chunks": [...],           # list of relevant chunk dicts
        "query_used": "...",       # final query that found results
        "retries": 0,              # number of retries needed
        "success": True/False      # whether relevant chunks were found
    }
    """
    original_query = question
    current_query = question
    retries = 0

    for attempt in range(config.MAX_AGENT_RETRIES + 1):
        chunks = search(current_query, session_id)

        if chunks:
            # Found relevant chunks
            return {
                "chunks": chunks,
                "query_used": current_query,
                "original_query": original_query,
                "retries": retries,
                "success": True
            }

        # No results — rephrase and retry
        if attempt < config.MAX_AGENT_RETRIES:
            retries += 1
            print(f"   No results for query '{current_query}' — rephrasing (attempt {retries})...")
            current_query = _rephrase_query(original_query, current_query, attempt + 1)
            time.sleep(0.5)  # small delay between retries

    # All retries exhausted
    return {
        "chunks": [],
        "query_used": current_query,
        "original_query": original_query,
        "retries": retries,
        "success": False
    }


def _rephrase_query(original: str, failed_query: str, attempt: int) -> str:
    """
    Uses LLM to rephrase a query that returned no results.

    Why rephrasing helps:
    - User asks "What's the transformer architecture?" 
    - Embeddings don't match — paper uses "encoder-decoder structure"
    - Rephrased: "encoder decoder model architecture"
    - Now embeddings match the paper's vocabulary

    Returns the rephrased query string.
    """
    rephrase_prompt = f"""A search query returned no results. Rephrase it to use different vocabulary.

Original question: {original}
Failed query: {failed_query}
Attempt: {attempt}

Rules:
- Use synonyms and alternative phrasings
- Break into key concepts
- Use more general terms if specific ones failed
- Keep it concise (under 15 words)

Respond with ONLY the rephrased query, no explanation:"""

    try:
        response = _call_llm_with_retry(
            messages=[{"role": "user", "content": rephrase_prompt}],
            temperature=0.3,
            max_tokens=50
        )
        rephrased = response.strip().strip('"').strip("'")
        return rephrased if rephrased else original
    except Exception:
        # If rephrasing fails, return original
        return original


# ─── Step 3: GENERATE ────────────────────────────────────────

def generate(
    question: str,
    chunks: list[dict],
    session_id: str,
    direct: bool = False
) -> dict:
    """
    Step 3 of the agentic loop.
    LLM generates an answer grounded in retrieved chunks.

    Two modes:
    - direct=False: uses retrieved chunks, must cite sources
    - direct=True:  answers from general knowledge (no chunks needed)

    Citation format enforced in prompt:
    "According to [filename], page [N], ..."

    Constrained generation:
    The prompt explicitly tells the LLM to ONLY use the chunks.
    This is the first layer of hallucination prevention.
    The second layer is the evaluator (Step 4).

    Returns:
    {
        "answer": "...",
        "citations": [{"filename": "...", "page": N, "text": "..."}],
        "used_chunks": [...]
    }
    """
    history = get_history(session_id)
    history_str = _format_history_for_prompt(history)

    if direct:
        # Answer from general knowledge — no chunks
        prompt = f"""You are a helpful research assistant.

Recent conversation:
{history_str}

Question: {question}

Answer clearly and concisely. If you're not sure, say so."""

        answer = _call_llm_with_retry(
            messages=[{"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )

        return {
            "answer": answer,
            "citations": [],
            "used_chunks": [],
            "mode": "direct"
        }

    # Format chunks for the prompt
    chunks_text = _format_chunks_for_prompt(chunks)

    # Grounded generation prompt
    prompt = f"""You are a research assistant. Answer the question using ONLY the provided source chunks.

IMPORTANT RULES:
1. Use ONLY information from the source chunks below
2. Cite sources using format: "According to [filename], page [N], ..."
3. If the chunks don't contain enough information, say "The provided documents don't contain enough information to answer this fully"
4. Do NOT use outside knowledge — only what's in the chunks
5. Be precise and cite page numbers

Recent conversation:
{history_str}

SOURCE CHUNKS:
{chunks_text}

Question: {question}

Answer (with citations):"""

    answer = _call_llm_with_retry(
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )

    # Extract citations from chunks used
    citations = [
        {
            "filename": c["filename"],
            "page": c["page"],
            "score": c["score"],
            "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "citations": citations,
        "used_chunks": chunks,
        "mode": "retrieval"
    }


# ─── Step 4: EVALUATE ────────────────────────────────────────

def evaluate(question: str, answer: str, chunks: list[dict]) -> dict:
    """
    Step 4 of the agentic loop.
    Second LLM call judges the quality of the generated answer.

    This is the LLM-as-judge pattern from Project 1, extended with:
    - groundedness: is every claim supported by chunks?
    - citation_accuracy: are page numbers and filenames correct?

    Why a second LLM call?
    - The generator LLM is optimistic about its own output
    - A separate judge call with different context catches errors
    - Same pattern used in production AI systems (Constitutional AI)

    Returns scores dict with pass/fail determination.
    """
    if not chunks:
        # Direct answer — no chunks to evaluate against
        return {
            "groundedness": 10,
            "relevance": 8,
            "citation_accuracy": 10,
            "hallucination_detected": False,
            "overall": 9.3,
            "passed": True,
            "issues": "Direct answer — no retrieval needed",
            "mode": "direct"
        }

    chunks_text = _format_chunks_for_prompt(chunks)

    eval_prompt = f"""You are an evaluation judge. Score this answer based on the source chunks.

Question: {question}

Answer to evaluate:
{answer}

Source chunks the answer should be based on:
{chunks_text}

Score each dimension from 0-10:
- groundedness: Is every claim in the answer supported by the chunks? (10 = fully grounded)
- relevance: Does the answer address the question? (10 = perfectly relevant)
- citation_accuracy: Are page numbers and filenames cited correctly? (10 = all correct)
- hallucination_detected: true if answer contains claims NOT in the chunks

Respond ONLY with valid JSON:
{{
    "groundedness": 0-10,
    "relevance": 0-10,
    "citation_accuracy": 0-10,
    "hallucination_detected": true or false,
    "issues": "brief description of problems, or none"
}}"""

    response = _call_llm_with_retry(
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0,    # zero temperature for consistent scoring
        max_tokens=200
    )

    return _parse_eval_response(response)


def _parse_eval_response(response: str) -> dict:
    """
    Parses evaluation response into structured scores.
    Handles edge cases and malformed JSON gracefully.
    """
    default_scores = {
        "groundedness": 5,
        "relevance": 5,
        "citation_accuracy": 5,
        "hallucination_detected": False,
        "overall": 5.0,
        "passed": False,
        "issues": "Could not parse evaluation response"
    }

    try:
        # Try direct parse
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        try:
            # Extract JSON from response
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                return default_scores
        except Exception:
            return default_scores

    # Calculate overall score
    groundedness = float(data.get("groundedness", 5))
    relevance = float(data.get("relevance", 5))
    citation_accuracy = float(data.get("citation_accuracy", 5))
    hallucination = bool(data.get("hallucination_detected", False))

    # Weighted average: groundedness matters most for RAG
    overall = (groundedness * 0.5) + (relevance * 0.3) + (citation_accuracy * 0.2)

    # Hallucination is an automatic fail regardless of scores
    if hallucination:
        overall = min(overall, 5.0)

    passed = overall >= config.HALLUCINATION_PASS_SCORE and not hallucination

    return {
        "groundedness": groundedness,
        "relevance": relevance,
        "citation_accuracy": citation_accuracy,
        "hallucination_detected": hallucination,
        "overall": round(overall, 2),
        "passed": passed,
        "issues": data.get("issues", "none"),
        "mode": "evaluated"
    }


# ─── Full Agent Loop ─────────────────────────────────────────

def run(question: str, session_id: str) -> dict:
    """
    THE main function. Runs the complete agentic loop.
    Called by main.py for every user question.

    Full loop:
    1. PLAN    → decide: search or answer directly
    2. RETRIEVE → get relevant chunks (with retry + rephrase)
    3. GENERATE → create grounded answer with citations
    4. EVALUATE → judge answer quality
    5. RETRY   → if score < threshold, retry with rephrased query
    6. RETURN  → final answer + sources + scores + metadata

    Returns complete result dict for API response.
    """
    start_time = time.time()
    retries_used = 0

    # ── Step 1: PLAN ──────────────────────────────────────────
    print(f"\n[Agent] Step 1: Planning...")
    plan_result = plan(question, session_id)
    action = plan_result.get("action", "search")
    print(f"[Agent] Action: {action} — {plan_result.get('reasoning', '')}")

    # ── Direct answer path ────────────────────────────────────
    if action == "answer_directly":
        print(f"[Agent] Step 3: Generating direct answer...")
        gen_result = generate(question, [], session_id, direct=True)
        eval_result = evaluate(question, gen_result["answer"], [])

        elapsed = round(time.time() - start_time, 2)
        result = _build_result(
            question=question,
            answer=gen_result["answer"],
            citations=[],
            plan=plan_result,
            retrieval={"chunks": [], "success": False, "retries": 0},
            evaluation=eval_result,
            elapsed=elapsed,
            retries_used=0
        )

        # Store in memory
        add_turn(session_id, question, gen_result["answer"])
        return result

    # ── Retrieval path ────────────────────────────────────────
    current_question = question

    for attempt in range(config.MAX_AGENT_RETRIES + 1):

        # ── Step 2: RETRIEVE ──────────────────────────────────
        print(f"[Agent] Step 2: Retrieving (attempt {attempt + 1})...")
        retrieval_result = retrieve(current_question, session_id)

        if not retrieval_result["success"]:
            # No relevant chunks found even after internal retries
            # Fall back to direct answer with a note
            print(f"[Agent] No relevant chunks found — falling back to direct answer")
            gen_result = generate(
                f"{question} (Note: no relevant documents found, answering from general knowledge)",
                [],
                session_id,
                direct=True
            )
            eval_result = evaluate(question, gen_result["answer"], [])

            elapsed = round(time.time() - start_time, 2)
            result = _build_result(
                question=question,
                answer=gen_result["answer"],
                citations=[],
                plan=plan_result,
                retrieval=retrieval_result,
                evaluation=eval_result,
                elapsed=elapsed,
                retries_used=retries_used
            )
            add_turn(session_id, question, gen_result["answer"])
            return result

        chunks = retrieval_result["chunks"]
        print(f"[Agent] Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']})")

        # ── Step 3: GENERATE ──────────────────────────────────
        print(f"[Agent] Step 3: Generating answer...")
        gen_result = generate(question, chunks, session_id, direct=False)

        # ── Step 4: EVALUATE ──────────────────────────────────
        print(f"[Agent] Step 4: Evaluating answer...")
        eval_result = evaluate(question, gen_result["answer"], chunks)
        print(f"[Agent] Score: {eval_result['overall']}/10 — {'PASS' if eval_result['passed'] else 'FAIL'}")

        # ── Step 5: CHECK & RETRY ─────────────────────────────
        if eval_result["passed"]:
            # Answer is good — return it
            break

        if attempt < config.MAX_AGENT_RETRIES:
            # Score too low — retry with rephrased question
            retries_used += 1
            print(f"[Agent] Score below threshold — retrying ({retries_used}/{config.MAX_AGENT_RETRIES})...")
            current_question = _rephrase_query(question, current_question, attempt + 1)
            time.sleep(0.5)
            continue

        # Max retries reached — return best answer we have
        print(f"[Agent] Max retries reached — returning best answer")
        break

    # ── Step 6: RETURN ────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)
    result = _build_result(
        question=question,
        answer=gen_result["answer"],
        citations=gen_result["citations"],
        plan=plan_result,
        retrieval=retrieval_result,
        evaluation=eval_result,
        elapsed=elapsed,
        retries_used=retries_used
    )

    # Store turn in memory
    add_turn(session_id, question, gen_result["answer"])

    return result


# ─── Streaming Support ───────────────────────────────────────

def stream(question: str, session_id: str) -> Generator[dict, None, None]:
    """
    Streaming version of run().
    Yields status updates as the agent works through each step.
    Called by main.py's /chat endpoint via Server-Sent Events (SSE).

    Why streaming?
    - Agent loop takes 3-8 seconds (multiple LLM calls)
    - Without streaming: user sees blank screen for 8 seconds
    - With streaming: user sees "Planning... Retrieving... Generating..."
    - Much better UX, same result

    Yields dicts with type field:
    - "status": intermediate step update
    - "result": final complete answer
    - "error": if something went wrong
    """
    try:
        # Status: Planning
        yield {"type": "status", "message": "Planning...", "step": 1}
        plan_result = plan(question, session_id)
        action = plan_result.get("action", "search")
        yield {
            "type": "status",
            "message": f"{'Searching documents' if action == 'search' else 'Answering directly'}...",
            "step": 2,
            "action": action
        }

        if action == "answer_directly":
            yield {"type": "status", "message": "Generating answer...", "step": 3}
            gen_result = generate(question, [], session_id, direct=True)
            eval_result = evaluate(question, gen_result["answer"], [])
            add_turn(session_id, question, gen_result["answer"])

            yield {
                "type": "result",
                "answer": gen_result["answer"],
                "citations": [],
                "evaluation": eval_result,
                "plan": plan_result,
                "retries": 0
            }
            return

        # Retrieval path
        current_question = question
        retries_used = 0

        for attempt in range(config.MAX_AGENT_RETRIES + 1):
            yield {"type": "status", "message": "Retrieving relevant chunks...", "step": 2}
            retrieval_result = retrieve(current_question, session_id)

            if not retrieval_result["success"]:
                yield {"type": "status", "message": "No docs found — answering from knowledge...", "step": 3}
                gen_result = generate(question, [], session_id, direct=True)
                eval_result = evaluate(question, gen_result["answer"], [])
                add_turn(session_id, question, gen_result["answer"])
                yield {
                    "type": "result",
                    "answer": gen_result["answer"],
                    "citations": [],
                    "evaluation": eval_result,
                    "plan": plan_result,
                    "retries": retries_used
                }
                return

            yield {"type": "status", "message": "Generating grounded answer...", "step": 3}
            gen_result = generate(question, retrieval_result["chunks"], session_id)

            yield {"type": "status", "message": "Evaluating answer quality...", "step": 4}
            eval_result = evaluate(question, gen_result["answer"], retrieval_result["chunks"])

            if eval_result["passed"] or attempt >= config.MAX_AGENT_RETRIES:
                add_turn(session_id, question, gen_result["answer"])
                yield {
                    "type": "result",
                    "answer": gen_result["answer"],
                    "citations": gen_result["citations"],
                    "evaluation": eval_result,
                    "plan": plan_result,
                    "retries": retries_used
                }
                return

            retries_used += 1
            current_question = _rephrase_query(question, current_question, attempt + 1)
            yield {
                "type": "status",
                "message": f"Retrying with better query... (attempt {retries_used})",
                "step": 2
            }
            time.sleep(0.5)

    except Exception as e:
        yield {"type": "error", "message": str(e)}


# ─── Helper Functions ────────────────────────────────────────

def _call_llm_with_retry(
    messages: list[dict],
    temperature: float = None,
    max_tokens: int = None,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """
    Calls Groq LLM with retry logic.
    Returns the text content of the response.

    Handles:
    - Rate limits (429): wait and retry
    - Timeouts: retry
    - API errors: retry up to max_retries
    """
    client = _get_groq()
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = str(e)
            error_str = str(e).lower()

            if "rate limit" in error_str or "429" in error_str:
                wait = retry_delay * (attempt + 1)
                print(f"   Groq rate limit — waiting {wait}s")
                time.sleep(wait)
                continue

            elif "401" in error_str or "invalid api key" in error_str:
                raise ValueError("Groq API key invalid. Check GROQ_API_KEY in .env")

            elif attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue

            else:
                raise RuntimeError(f"Groq LLM failed after {max_retries} attempts: {last_error}")

    raise RuntimeError(f"Groq LLM failed after {max_retries} attempts: {last_error}")


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Formats chunks into readable text for LLM prompts.
    Each chunk labeled with source file and page number.
    """
    formatted = []
    for i, chunk in enumerate(chunks):
        formatted.append(
            f"[Source {i+1}] {chunk['filename']}, Page {chunk['page']}:\n{chunk['text']}"
        )
    return "\n\n".join(formatted)


def _format_history_for_prompt(history: list[dict]) -> str:
    """
    Formats conversation history for inclusion in prompts.
    Returns last MEMORY_WINDOW turns as readable text.
    """
    if not history:
        return "No previous conversation."

    lines = []
    for turn in history[-config.MEMORY_WINDOW:]:
        lines.append(f"User: {turn.get('question', '')}")
        lines.append(f"Assistant: {turn.get('answer', '')[:200]}...")
    return "\n".join(lines)


def _build_result(
    question: str,
    answer: str,
    citations: list,
    plan: dict,
    retrieval: dict,
    evaluation: dict,
    elapsed: float,
    retries_used: int
) -> dict:
    """
    Builds the final result dict returned by run().
    This is the exact shape main.py and the frontend expect.
    """
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "metadata": {
            "action": plan.get("action"),
            "reasoning": plan.get("reasoning"),
            "chunks_retrieved": len(retrieval.get("chunks", [])),
            "query_used": retrieval.get("query_used", question),
            "retries": retries_used,
            "elapsed_seconds": elapsed
        },
        "evaluation": {
            "overall": evaluation.get("overall", 0),
            "groundedness": evaluation.get("groundedness", 0),
            "relevance": evaluation.get("relevance", 0),
            "citation_accuracy": evaluation.get("citation_accuracy", 0),
            "hallucination_detected": evaluation.get("hallucination_detected", False),
            "passed": evaluation.get("passed", False),
            "issues": evaluation.get("issues", "none")
        }
    }


# ─── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: python agent.py
    Tests the full agent loop end to end.
    Requires documents already in vector store OR uploads one.
    """
    import sys
    from pdf_processor import extract_pdf
    from chunker import chunk_document
    from vector_store import store_chunks, delete_session

    print(f"\n{'='*50}")
    print(f"  {config.APP_NAME} — Agent Test")
    print(f"{'='*50}\n")

    TEST_SESSION = "agent_test_001"

    # Load a PDF for testing if provided
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Loading: {test_path}")
        doc = extract_pdf(test_path)
        if not doc.error:
            chunks = chunk_document(doc)
            result = store_chunks(chunks, TEST_SESSION)
            print(f"✅ Stored {result['stored']} chunks\n")
        else:
            print(f"❌ PDF error: {doc.error}")
            sys.exit(1)
    else:
        print("No PDF provided — testing with existing session data")
        print("Usage: python agent.py path/to/file.pdf\n")

    # Test questions
    test_questions = [
        "What is retrieval augmented generation?",
        "What are the main components of the system?",
        "What is 2 + 2?",   # should answer_directly
    ]

    for question in test_questions:
        print(f"\n{'─'*50}")
        print(f"Q: {question}")
        print(f"{'─'*50}")

        result = run(question, TEST_SESSION)

        print(f"\nAnswer: {result['answer'][:300]}...")
        print(f"\nMetadata:")
        print(f"  Action:    {result['metadata']['action']}")
        print(f"  Chunks:    {result['metadata']['chunks_retrieved']}")
        print(f"  Retries:   {result['metadata']['retries']}")
        print(f"  Time:      {result['metadata']['elapsed_seconds']}s")
        print(f"\nEvaluation:")
        print(f"  Overall:   {result['evaluation']['overall']}/10")
        print(f"  Passed:    {result['evaluation']['passed']}")
        print(f"  Issues:    {result['evaluation']['issues']}")

        if result['citations']:
            print(f"\nCitations:")
            for c in result['citations'][:2]:
                print(f"  {c['filename']}, Page {c['page']} (score: {c['score']})")

    # Cleanup
    delete_session(TEST_SESSION)
    print(f"\n✅ Test session cleaned up")
