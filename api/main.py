"""
FastAPI backend — thin API layer around the existing RAG pipeline,
with SQLite Chat History and Security Guardrails.
"""

import sys
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure Python can find the existing "implementation" package.
# The project root is one level above this file (d:/InsureLLM).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing RAG function — unchanged
from implementation.answer_langchain import answer_question

# Import security guardrails
from api.guardrails import (
    run_input_guardrails,
    run_output_guardrails,
    redact_sensitive_data,
)

# Import SQLite database operations
from api.database import (
    init_db,
    get_all_sessions,
    create_session,
    get_session,
    update_session_title,
    delete_session,
    get_session_messages,
    add_message,
)

# ---------------------------------------------------------------------------
# App setup & Database Initialization
# ---------------------------------------------------------------------------
app = FastAPI(title="InsureLLM RAG API")

# Initialize SQLite tables on startup
init_db()

# Allow the React dev server (port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Request / Response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


class ChatRequest(BaseModel):
    question: str
    session_id: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    chunks: list[dict]


# ---------------------------------------------------------------------------
# SQLite Session Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    """Return all conversation sessions from SQLite."""
    sessions = get_all_sessions()
    # If no sessions exist yet, create a default first one
    if not sessions:
        new_s = create_session(title="New Chat")
        sessions = [new_s]
    return sessions


@app.post("/api/sessions")
async def create_new_session(req: CreateSessionRequest = CreateSessionRequest()):
    """Create a new conversation session in SQLite."""
    return create_session(title=req.title)


@app.get("/api/sessions/{session_id}/messages")
async def get_messages_for_session(session_id: str):
    """Retrieve all messages for a specific session from SQLite."""
    return get_session_messages(session_id)


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a session and its messages from SQLite."""
    delete_session(session_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# The Chat API endpoint (with SQLite persistence & Guardrails)
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Receive question + session_id:
      1. Ensure session exists in SQLite.
      2. Fetch conversation history from SQLite.
      3. Check guardrails.
      4. Call RAG pipeline.
      5. Save user turn and assistant reply into SQLite.
    """
    session_id = request.session_id

    # Ensure session exists in SQLite
    current_s = get_session(session_id)
    if not current_s:
        create_session(session_id=session_id, title="New Chat")
        current_s = {"id": session_id, "title": "New Chat"}

    # 1. Fetch previous history from SQLite for this session
    previous_history = get_session_messages(session_id)

    # 2. Input Guardrails Check
    is_safe, refusal_reason = run_input_guardrails(request.question)
    if not is_safe:
        # Save both user question and refusal into SQLite so history stays consistent
        add_message(session_id, "user", request.question)
        add_message(session_id, "assistant", refusal_reason)

        return ChatResponse(
            answer=refusal_reason,
            chunks=[],
        )

    # 3. Sanitize input
    sanitized_question = redact_sensitive_data(request.question)

    try:
        # Pass a copy of the previous history to RAG
        history_copy = list(previous_history)

        # Call existing RAG pipeline
        answer, chunks = answer_question(
            question=sanitized_question,
            history=history_copy,
        )

        # 4. Output Guardrails Check
        safe_answer = run_output_guardrails(answer)

        # 5. Persist this turn in SQLite!
        add_message(session_id, "user", request.question)
        add_message(session_id, "assistant", safe_answer)

        # Update session title if it was still "New Chat"
        if current_s.get("title") == "New Chat":
            preview_title = request.question.strip()[:25]
            if len(request.question.strip()) > 25:
                preview_title += "..."
            update_session_title(session_id, preview_title)

        # Convert chunks to dicts
        chunks_as_dicts = []
        for chunk in chunks:
            chunks_as_dicts.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
            })

        return ChatResponse(answer=safe_answer, chunks=chunks_as_dicts)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Server Boot Identifier
# ---------------------------------------------------------------------------
SERVER_BOOT_ID = str(time.time())


@app.get("/health")
async def health():
    return {"status": "ok", "boot_id": SERVER_BOOT_ID}
