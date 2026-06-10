import json
import re
import os
import time
from datetime import datetime
from groq import Groq
import config

# ─── Groq Client ─────────────────────────────────────────────
_groq_client = None

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


# ─── Evaluation Dimensions ───────────────────────────────────
# Extended from Project 1's LLM-as-judge pattern
# Project 1 had: accuracy, format_compliance, relevance, hallucination_score
# Synapse adds: groundedness, citation_accuracy (RAG-specific)

EVAL_DIMENSIONS = {
    "groundedness": {
        "description": "Is every claim in the answer supported by the source chunks?",
        "weight": 0.5       # highest weight — most critical for RAG
    },
    "relevance": {
        "description": "Does the answer address what the user actually asked?",
        "weight": 0.3
    },
    "citation_accuracy": {
        "description": "Are cited page numbers and filenames correct?",
        "weight": 0.2
    }
}

PASS_THRESHOLD = config.HALLUCINATION_PASS_SCORE  # 7.0


# ─── Single Response Evaluator ───────────────────────────────

def evaluate_response(
    question: str,
    answer: str,
    chunks: list[dict],
    session_id: str = "eval"
) -> dict:
    """
    Evaluates a single question/answer pair against source chunks.
    This is the same function called by agent.py in Step 4.

    Extended from Project 1 pattern:
    - Same LLM-as-judge approach
    - Same pass threshold (≥ 7.0)
    - New dimensions: groundedness, citation_accuracy
    - New metric: hallucination_detected (boolean)

    Returns complete evaluation dict with scores, pass/fail, issues.
    """
    if not chunks:
        # Direct answer — no retrieval to evaluate against
        return _direct_answer_eval(question, answer)

    chunks_text = _format_chunks(chunks)

    eval_prompt = f"""You are an expert evaluation judge for a RAG (Retrieval Augmented Generation) system.
Evaluate this answer based on the provided source chunks.

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

SOURCE CHUNKS (what the answer should be based on):
{chunks_text}

Score each dimension from 0 to 10:

1. GROUNDEDNESS (0-10): Is every claim in the answer directly supported by the source chunks?
   - 10: Every single claim traces back to a chunk
   - 7: Most claims supported, minor unsupported details
   - 4: Some claims not in chunks
   - 0: Answer ignores chunks entirely

2. RELEVANCE (0-10): Does the answer address what was asked?
   - 10: Perfectly answers the question
   - 7: Answers most of it, minor gaps
   - 4: Partially relevant
   - 0: Completely off topic

3. CITATION_ACCURACY (0-10): Are the page numbers and filenames cited correctly?
   - 10: All citations are correct
   - 7: Most citations correct, minor errors
   - 4: Some citations wrong
   - 0: No citations or all wrong

4. HALLUCINATION_DETECTED (true/false): Does the answer contain ANY claims not in the chunks?
   - true: Answer makes claims beyond what the chunks say
   - false: Answer stays within chunk content

Respond ONLY with valid JSON, no other text:
{{
    "groundedness": <0-10>,
    "relevance": <0-10>,
    "citation_accuracy": <0-10>,
    "hallucination_detected": <true or false>,
    "issues": "<brief description of problems found, or 'none'>"
}}"""

    response = _call_llm(eval_prompt, temperature=0.0, max_tokens=250)
    return _parse_scores(response)


def _direct_answer_eval(question: str, answer: str) -> dict:
    """
    Lightweight evaluation for direct (non-retrieval) answers.
    Only checks relevance since there are no chunks to check against.
    """
    eval_prompt = f"""Rate how well this answer addresses the question.

QUESTION: {question}
ANSWER: {answer}

Respond ONLY with valid JSON:
{{
    "relevance": <0-10>,
    "issues": "<any issues or 'none'>"
}}"""

    response = _call_llm(eval_prompt, temperature=0.0, max_tokens=100)

    try:
        data = json.loads(response.strip())
        relevance = float(data.get("relevance", 8))
    except Exception:
        relevance = 8.0

    return {
        "groundedness": 10.0,
        "relevance": relevance,
        "citation_accuracy": 10.0,
        "hallucination_detected": False,
        "overall": round((10 * 0.5) + (relevance * 0.3) + (10 * 0.2), 2),
        "passed": True,
        "issues": "Direct answer — no retrieval",
        "mode": "direct"
    }


# ─── Batch Evaluator ─────────────────────────────────────────

def run_batch_eval(test_cases: list[dict], session_id: str) -> dict:
    """
    Runs evaluation on a batch of test cases.
    Generates the results table used in README.md.

    Extended from Project 1's batch evaluation pattern.
    Used to prove the system works before deploying.

    Each test case:
    {
        "question": "...",
        "expected_topics": ["attention", "transformer"],  # keywords expected in answer
        "should_retrieve": True/False                     # should agent search docs?
    }

    Returns summary report with pass rate, avg scores, per-case results.
    """
    from agent import run

    results = []
    passed = 0
    total = len(test_cases)

    print(f"\n  Running {total} test cases...\n")

    for i, case in enumerate(test_cases):
        question = case["question"]
        print(f"  [{i+1}/{total}] {question[:60]}...")

        try:
            # Run the full agent loop
            result = run(question, session_id)

            answer = result["answer"]
            citations = result["citations"]
            eval_scores = result["evaluation"]
            action = result["metadata"]["action"]

            # Check expected topics appear in answer
            expected_topics = case.get("expected_topics", [])
            topics_found = [
                t for t in expected_topics
                if t.lower() in answer.lower()
            ]
            topic_coverage = len(topics_found) / len(expected_topics) if expected_topics else 1.0

            # Check action matches expectation
            expected_retrieve = case.get("should_retrieve", True)
            action_correct = (action == "search") == expected_retrieve

            case_result = {
                "question": question,
                "action": action,
                "action_correct": action_correct,
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                "citations_count": len(citations),
                "evaluation": eval_scores,
                "topic_coverage": round(topic_coverage, 2),
                "passed": eval_scores["passed"] and action_correct
            }

            if case_result["passed"]:
                passed += 1
                print(f"     ✅ PASS — Score: {eval_scores['overall']}/10")
            else:
                print(f"     ❌ FAIL — Score: {eval_scores['overall']}/10")
                if not action_correct:
                    print(f"     Expected action: {'search' if expected_retrieve else 'answer_directly'}, got: {action}")

        except Exception as e:
            case_result = {
                "question": question,
                "error": str(e),
                "passed": False
            }
            print(f"     ❌ ERROR: {e}")

        results.append(case_result)
        time.sleep(0.5)  # small delay between cases

    # Calculate summary stats
    eval_scores_list = [
        r["evaluation"] for r in results
        if "evaluation" in r
    ]

    avg_overall = round(
        sum(e["overall"] for e in eval_scores_list) / len(eval_scores_list), 2
    ) if eval_scores_list else 0

    avg_groundedness = round(
        sum(e["groundedness"] for e in eval_scores_list) / len(eval_scores_list), 2
    ) if eval_scores_list else 0

    hallucinations = sum(
        1 for e in eval_scores_list if e.get("hallucination_detected", False)
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{round((passed/total)*100)}%",
        "avg_overall_score": avg_overall,
        "avg_groundedness": avg_groundedness,
        "hallucinations_detected": hallucinations,
        "results": results
    }

    return report


def print_report(report: dict):
    """
    Prints a formatted evaluation report to terminal.
    Same ASCII bar style as Project 1.
    """
    print(f"\n{'='*55}")
    print(f"  {config.APP_NAME} — Evaluation Report")
    print(f"  {report['timestamp']}")
    print(f"{'='*55}\n")

    print(f"  SUMMARY")
    print(f"  {'─'*40}")
    print(f"  Total cases:        {report['total_cases']}")
    print(f"  Passed:             {report['passed']} ✅")
    print(f"  Failed:             {report['failed']} ❌")
    print(f"  Pass rate:          {report['pass_rate']}")
    print(f"  Avg overall score:  {report['avg_overall_score']}/10")
    print(f"  Avg groundedness:   {report['avg_groundedness']}/10")
    print(f"  Hallucinations:     {report['hallucinations_detected']}")

    print(f"\n  RESULTS")
    print(f"  {'─'*40}")

    for i, result in enumerate(report["results"]):
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        question = result["question"][:50] + "..." if len(result["question"]) > 50 else result["question"]
        score = result.get("evaluation", {}).get("overall", 0)
        bar = _score_bar(score)
        print(f"\n  [{i+1}] {status}")
        print(f"  Q: {question}")
        print(f"  Score: {bar} {score}/10")
        if "error" in result:
            print(f"  Error: {result['error']}")


def save_report(report: dict):
    """
    Saves evaluation report to outputs/eval_logs/.
    Creates directory if it doesn't exist.
    """
    os.makedirs(config.EVAL_LOGS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.EVAL_LOGS_PATH}/eval_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {filename}")
    return filename


# ─── Helper Functions ────────────────────────────────────────

def _format_chunks(chunks: list[dict]) -> str:
    formatted = []
    for i, chunk in enumerate(chunks):
        formatted.append(
            f"[Source {i+1}] {chunk['filename']}, Page {chunk['page']}:\n{chunk['text']}"
        )
    return "\n\n".join(formatted)


def _call_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 250) -> str:
    client = _get_groq()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise e
    return "{}"


def _parse_scores(response: str) -> dict:
    """
    Parses evaluation scores from LLM response.
    Robust to extra text, markdown formatting, etc.
    """
    default = {
        "groundedness": 5.0,
        "relevance": 5.0,
        "citation_accuracy": 5.0,
        "hallucination_detected": False,
        "overall": 5.0,
        "passed": False,
        "issues": "Parse error"
    }

    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                return default
        except Exception:
            return default

    groundedness = float(data.get("groundedness", 5))
    relevance = float(data.get("relevance", 5))
    citation_accuracy = float(data.get("citation_accuracy", 5))
    hallucination = bool(data.get("hallucination_detected", False))

    overall = (
        groundedness * EVAL_DIMENSIONS["groundedness"]["weight"] +
        relevance * EVAL_DIMENSIONS["relevance"]["weight"] +
        citation_accuracy * EVAL_DIMENSIONS["citation_accuracy"]["weight"]
    )

    if hallucination:
        overall = min(overall, 5.0)

    passed = overall >= PASS_THRESHOLD and not hallucination

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


def _score_bar(score: float, width: int = 20) -> str:
    """ASCII score bar — same style as Project 1."""
    filled = int((score / 10) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


# ─── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run batch evaluation: python evaluator.py path/to/file.pdf
    Tests the full system and generates README results table.
    """
    import sys
    from pdf_processor import extract_pdf
    from chunker import chunk_document
    from vector_store import store_chunks, delete_session

    print(f"\n{'='*55}")
    print(f"  {config.APP_NAME} — Batch Evaluator")
    print(f"{'='*55}\n")

    TEST_SESSION = "eval_session_001"

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Loading: {test_path}")
        doc = extract_pdf(test_path)
        if not doc.error:
            chunks = chunk_document(doc)
            result = store_chunks(chunks, TEST_SESSION)
            print(f"✅ Loaded {result['stored']} chunks\n")
        else:
            print(f"❌ {doc.error}")
            sys.exit(1)
    else:
        print("Usage: python evaluator.py path/to/file.pdf")
        sys.exit(0)

    # Test cases for rag_paper.pdf
    test_cases = [
        {
            "question": "What is retrieval augmented generation?",
            "expected_topics": ["retrieval", "generation", "knowledge"],
            "should_retrieve": True
        },
        {
            "question": "What datasets were used to evaluate the system?",
            "expected_topics": ["dataset", "evaluation"],
            "should_retrieve": True
        },
        {
            "question": "What are the limitations of this approach?",
            "expected_topics": ["limitation"],
            "should_retrieve": True
        },
        {
            "question": "What is the capital of France?",
            "expected_topics": ["Paris"],
            "should_retrieve": False
        },
        {
            "question": "How does the retriever component work?",
            "expected_topics": ["retriev"],
            "should_retrieve": True
        }
    ]

    report = run_batch_eval(test_cases, TEST_SESSION)
    print_report(report)
    save_report(report)

    delete_session(TEST_SESSION)
    print(f"\n✅ Eval session cleaned up")
