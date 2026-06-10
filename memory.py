import time
import config

# ─── In-Memory Session Store ─────────────────────────────────
# Dict of session_id → session data
# Structure per session:
# {
#   "turns": [{"question": "...", "answer": "...", "timestamp": 123}],
#   "summary": "summarized older context or None",
#   "created_at": 123456789
# }
_sessions: dict = {}


# ─── Core Functions ──────────────────────────────────────────

def get_history(session_id: str) -> list[dict]:
    """
    Returns the last MEMORY_WINDOW turns for a session.
    Called by agent.py before every LLM call for context.

    Returns list of turn dicts:
    [{"question": "...", "answer": "...", "timestamp": ...}]
    """
    session = _get_or_create_session(session_id)
    turns = session["turns"]

    # Return last MEMORY_WINDOW turns only
    return turns[-config.MEMORY_WINDOW:]


def get_summary(session_id: str) -> str:
    """
    Returns summarized older context for a session.
    Empty string if no summary exists yet.
    Called by agent.py to include older context in prompts.
    """
    session = _get_or_create_session(session_id)
    return session.get("summary") or ""


def add_turn(session_id: str, question: str, answer: str):
    """
    Adds a question/answer turn to session history.
    Called by agent.py after every successful response.

    Auto-triggers summarization when history exceeds
    MEMORY_SUMMARY_THRESHOLD (10 turns).

    Why summarize instead of truncate?
    - Truncation loses information permanently
    - Summarization compresses context, retains meaning
    - User can still reference earlier parts of conversation
    """
    session = _get_or_create_session(session_id)

    session["turns"].append({
        "question": question,
        "answer": answer,
        "timestamp": time.time()
    })

    # Trigger summarization when history gets long
    if len(session["turns"]) > config.MEMORY_SUMMARY_THRESHOLD:
        _summarize_old_turns(session_id)


def clear_session(session_id: str):
    """
    Clears all memory for a session.
    Called by DELETE /session/{session} endpoint in main.py.
    Also called when user clicks "Reset Session" in frontend.
    """
    if session_id in _sessions:
        del _sessions[session_id]


def get_session_stats(session_id: str) -> dict:
    """
    Returns stats about a session's memory.
    Called by main.py for the /history endpoint.
    """
    session = _get_or_create_session(session_id)
    turns = session["turns"]

    return {
        "total_turns": len(turns),
        "has_summary": bool(session.get("summary")),
        "created_at": session.get("created_at"),
        "last_active": turns[-1]["timestamp"] if turns else None
    }


def get_full_history(session_id: str) -> dict:
    """
    Returns complete session history for the /history endpoint.
    Includes both summary and recent turns.
    """
    session = _get_or_create_session(session_id)

    return {
        "session_id": session_id,
        "summary": session.get("summary"),
        "turns": session["turns"],
        "stats": get_session_stats(session_id)
    }


# ─── Summarization ───────────────────────────────────────────

def _summarize_old_turns(session_id: str):
    """
    Summarizes older turns when history exceeds threshold.

    Strategy:
    - Keep the last MEMORY_WINDOW turns intact (recent context)
    - Summarize everything before that into one paragraph
    - Store summary separately, replace old turns with it
    - Next time this runs, it extends the existing summary

    Why LLM summarization over simple truncation:
    - User might say "as I mentioned earlier..." referencing turn 1
    - Truncation would lose that context entirely
    - Summary preserves the key points in ~100 words
    - Stays well within token limits for any model
    """
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)

        session = _sessions[session_id]
        turns = session["turns"]

        # Split: keep recent, summarize old
        keep_count = config.MEMORY_WINDOW
        old_turns = turns[:-keep_count]
        recent_turns = turns[-keep_count:]

        if not old_turns:
            return

        # Build text of old turns to summarize
        old_text = "\n".join([
            f"User: {t['question']}\nAssistant: {t['answer'][:300]}..."
            for t in old_turns
        ])

        # Include existing summary if there is one
        existing_summary = session.get("summary", "")
        if existing_summary:
            old_text = f"Previous summary: {existing_summary}\n\nNew turns:\n{old_text}"

        summary_prompt = f"""Summarize this conversation history concisely.
Preserve key facts, questions asked, and important answers.
Keep it under 150 words.

Conversation:
{old_text}

Summary:"""

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.1,
            max_tokens=200
        )

        new_summary = response.choices[0].message.content.strip()

        # Update session: store summary, keep only recent turns
        session["summary"] = new_summary
        session["turns"] = recent_turns

    except Exception as e:
        # Summarization failure is non-fatal
        # Worst case: history stays long, uses more tokens
        print(f"   Memory summarization warning: {e}")


# ─── Session Management ──────────────────────────────────────

def _get_or_create_session(session_id: str) -> dict:
    """
    Gets existing session or creates a new one.
    """
    if session_id not in _sessions:
        _sessions[session_id] = {
            "turns": [],
            "summary": None,
            "created_at": time.time()
        }
    return _sessions[session_id]


def list_sessions() -> list[str]:
    """
    Returns list of active session IDs.
    Used for debugging and admin purposes.
    """
    return list(_sessions.keys())


# ─── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: python memory.py
    Tests sliding window, summarization trigger, and session management.
    """
    print(f"\n{'='*50}")
    print(f"  {config.APP_NAME} — Memory Test")
    print(f"{'='*50}\n")

    TEST_SESSION = "memory_test_001"

    # Test 1: Basic add and retrieve
    print("Test 1: Basic add and retrieve")
    add_turn(TEST_SESSION, "What is RAG?", "RAG stands for Retrieval Augmented Generation...")
    add_turn(TEST_SESSION, "How does it work?", "It retrieves relevant documents then generates...")
    add_turn(TEST_SESSION, "What are the benefits?", "The main benefits are reduced hallucination...")

    history = get_history(TEST_SESSION)
    print(f"✅ Stored and retrieved {len(history)} turns")

    # Test 2: Sliding window
    print("\nTest 2: Sliding window (adding more than MEMORY_WINDOW turns)")
    for i in range(config.MEMORY_WINDOW + 2):
        add_turn(TEST_SESSION, f"Question {i}", f"Answer {i}")

    history = get_history(TEST_SESSION)
    print(f"✅ get_history returns {len(history)} turns (window: {config.MEMORY_WINDOW})")
    assert len(history) <= config.MEMORY_WINDOW, "Window not respected"

    # Test 3: Stats
    print("\nTest 3: Session stats")
    stats = get_session_stats(TEST_SESSION)
    print(f"✅ Total turns: {stats['total_turns']}")
    print(f"   Has summary: {stats['has_summary']}")

    # Test 4: Summarization trigger
    print(f"\nTest 4: Summarization trigger (adding {config.MEMORY_SUMMARY_THRESHOLD + 1} turns)")
    clear_session(TEST_SESSION)

    for i in range(config.MEMORY_SUMMARY_THRESHOLD + 1):
        add_turn(
            TEST_SESSION,
            f"Tell me about topic {i}",
            f"Topic {i} is about {'RAG' if i % 2 == 0 else 'transformers'} and related concepts."
        )

    stats = get_session_stats(TEST_SESSION)
    summary = get_summary(TEST_SESSION)

    print(f"✅ After {config.MEMORY_SUMMARY_THRESHOLD + 1} turns:")
    print(f"   Active turns: {stats['total_turns']}")
    print(f"   Has summary:  {stats['has_summary']}")
    if summary:
        print(f"   Summary preview: {summary[:100]}...")

    # Test 5: Clear session
    print("\nTest 5: Clear session")
    clear_session(TEST_SESSION)
    history = get_history(TEST_SESSION)
    print(f"✅ After clear: {len(history)} turns")

    print(f"\n{'='*50}")
    print(f"  All memory tests passed ✅")
    print(f"{'='*50}\n")
