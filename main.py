import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import asyncio
import json
import uuid
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import config
from pdf_processor import extract_pdf, get_pdf_summary
from chunker import chunk_document, get_chunk_stats
from vector_store import (
    store_chunks, search, list_documents,
    delete_document, delete_session, cleanup_old_sessions
)
from agent import run, stream
from memory import get_full_history, clear_session


# ─── Request Models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str

class CompareRequest(BaseModel):
    question: str
    session_id: str


# ─── Lifespan ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n{'='*50}")
    print(f"  {config.APP_NAME} — {config.APP_TAGLINE}")
    print(f"  Version {config.VERSION}")
    print(f"{'='*50}\n")

    print("  Starting up...")
    cleanup_old_sessions()
    asyncio.create_task(_preload_sample_docs())
    asyncio.create_task(_self_ping())
    print("  Ready.\n")
    yield
    print(f"\n  {config.APP_NAME} shutting down.")


# ─── App Setup ───────────────────────────────────────────────

app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_TAGLINE,
    version=config.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Frontend ────────────────────────────────────────────────

FRONTEND_FILE = "Synapse-Retrieve Verify Answer.html"

@app.get("/")
async def serve_frontend():
    if os.path.exists(FRONTEND_FILE):
        return FileResponse(FRONTEND_FILE)
    return {"message": f"{config.APP_NAME} API is running", "docs": "/docs"}


# ─── Health ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.VERSION,
        "tagline": config.APP_TAGLINE
    }


# ─── Upload ──────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    if not session_id:
        session_id = str(uuid.uuid4())

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_path = f"./temp_{file.filename}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        doc = extract_pdf(temp_path)
        if doc.error:
            raise HTTPException(status_code=422, detail=f"PDF extraction failed: {doc.error}")

        if doc.is_scanned:
            raise HTTPException(status_code=422, detail="Scanned PDFs are not supported. Please use text-based PDFs.")

        chunks = chunk_document(doc)
        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from this PDF")

        store_result = store_chunks(chunks, session_id)

        return {
            "success": True,
            "session_id": session_id,
            "filename": doc.filename,
            "pdf": get_pdf_summary(doc),
            "chunks": get_chunk_stats(chunks),
            "storage": store_result
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ─── Chat (Streaming SSE) ────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID required")

    async def generate():
        try:
            loop = asyncio.get_event_loop()

            def run_stream():
                return list(stream(request.question, request.session_id))

            chunks = await loop.run_in_executor(None, run_stream)

            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ─── Compare ─────────────────────────────────────────────────

@app.post("/compare")
async def compare(request: CompareRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    docs = list_documents(request.session_id)
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found in session.")

    responses = []

    for doc in docs:
        filename = doc["filename"]
        chunks = search(request.question, request.session_id, filename_filter=filename)

        if not chunks:
            responses.append({
                "filename": filename,
                "answer": "No relevant content found in this document.",
                "citations": [],
                "confidence": 0
            })
            continue

        from agent import generate as agent_generate, evaluate as agent_evaluate
        gen_result = agent_generate(request.question, chunks, request.session_id)
        eval_result = agent_evaluate(request.question, gen_result["answer"], chunks)

        responses.append({
            "filename": filename,
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "confidence": eval_result["overall"]
        })

    synthesis = ""
    if len(responses) > 1:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        answers_text = "\n\n".join([
            f"From {r['filename']}:\n{r['answer'][:500]}"
            for r in responses
            if r["answer"] != "No relevant content found in this document."
        ])

        if answers_text:
            try:
                response = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"Compare these answers about: '{request.question}'\n\n{answers_text}\n\nWrite a 2-3 sentence synthesis."
                    }],
                    temperature=0.1,
                    max_tokens=300
                )
                synthesis = response.choices[0].message.content.strip()
            except Exception:
                synthesis = "Could not generate synthesis."

    return {
        "question": request.question,
        "responses": responses,
        "synthesis": synthesis,
        "total_docs": len(docs)
    }


# ─── History ─────────────────────────────────────────────────

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return get_full_history(session_id)


# ─── Documents ───────────────────────────────────────────────

@app.get("/documents/{session_id}")
async def get_documents(session_id: str):
    docs = list_documents(session_id)
    return {"session_id": session_id, "documents": docs}


@app.delete("/documents/{session_id}/{filename}")
async def remove_document(session_id: str, filename: str):
    result = delete_document(filename, session_id)
    if result.get("deleted", 0) == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


# ─── Session ─────────────────────────────────────────────────

@app.delete("/session/{session_id}")
async def remove_session(session_id: str):
    delete_session(session_id)
    clear_session(session_id)
    return {"success": True, "session_id": session_id}


# ─── Keep-Alive Self-Ping ────────────────────────────────────

async def _self_ping():
    await asyncio.sleep(60)
    app_url = os.getenv("APP_URL", "")
    if not app_url or "localhost" in app_url:
        return
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{app_url}/health", timeout=10)
        except Exception:
            pass
        await asyncio.sleep(840)


# ─── Sample Docs Preloader ───────────────────────────────────

async def _preload_sample_docs():
    DEMO_SESSION = "demo_session"

    existing = list_documents(DEMO_SESSION)
    existing_files = [d["filename"] for d in existing]
    docs_to_load = [d for d in config.SAMPLE_DOCS if d not in existing_files]

    if not docs_to_load:
        print("  Sample docs already loaded.")
        return

    print(f"  Pre-loading {len(docs_to_load)} sample document(s)...")

    for filename in docs_to_load:
        path = os.path.join(config.SAMPLE_DOCS_PATH, filename)
        if not os.path.exists(path):
            print(f"  ⚠️  Not found: {path}")
            continue

        try:
            doc = extract_pdf(path)
            if doc.error:
                print(f"  ⚠️  {filename}: {doc.error}")
                continue

            chunks = chunk_document(doc)
            result = store_chunks(chunks, DEMO_SESSION)
            print(f"  ✅ {filename} ({result['stored']} chunks)")

        except Exception as e:
            print(f"  ⚠️  {filename}: {e}")

    print("  Sample docs ready.\n")


# ─── Run ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
