# 🛡️ InsureLLM — Conversational RAG Insurance Agent

An enterprise-grade, conversational Retrieval-Augmented Generation (RAG) assistant designed for **Insurellm**, an insurance technology company. The system features a modern ChatGPT-style React frontend, a FastAPI backend, an SQLite-backed session history, LangChain RAG with ChromaDB vector search and LLM reranking, and enterprise-grade security guardrails.

---

## 🏗️ System Architecture

```
                                  USER BROWSER
                         (React + Vite + TanStack Query)
                                       │
                                       ▼ HTTP (Axios)
                   ┌───────────────────────────────────────┐
                   │          FastAPI Backend (8000)       │
                   └──────────────────┬────────────────────┘
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
┌───────────────────────────┐                           ┌───────────────────────────┐
│   Security Guardrails     │                           │   SQLite Database Layer   │
│  (api/guardrails.py)      │                           │   (chat_history.db)       │
│  • Anti-DoS Length Limits │                           │  • sessions table         │
│  • Prompt Injection Block │                           │  • messages table         │
│  • PII & Secret Redaction │                           │  • Full conversation log  │
└─────────────┬─────────────┘                           └───────────────────────────┘
              │ (Passed)
              ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline (answer_langchain.py)                      │
│                                                                                   │
│  1. Query Rewriter (LLM)   ──► Synthesizes multi-turn dialogue into exact query  │
│  2. Dual Vector Search     ──► Original query + Rewritten query in ChromaDB       │
│  3. Chunk Deduplication    ──► Merges unique retrieved chunks                     │
│  4. LLM Reranking          ──► Sorts extracts by structured relevance order       │
│  5. Context Synthesis      ──► Assembles knowledge base context + dialogue        │
│  6. Answer Generation      ──► Grounded response generated with OpenRouter LLM    │
└─────────────────────────────────────┬─────────────────────────────────────────────┘
                                      │
                                      ▼
                             Frontend Chat View
```

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: React 19 + Vite
- **Language**: JavaScript (ES6+)
- **API State Management**: TanStack Query (`@tanstack/react-query`)
- **HTTP Client**: Axios
- **Styling**: Plain, responsive CSS (no heavyweight UI libraries)

### Backend
- **Framework**: FastAPI (Python 3.12)
- **Server**: Uvicorn ASGI
- **Data Validation**: Pydantic v2
- **Database**: SQLite (via Python built-in `sqlite3`)
- **Security**: Custom regex guardrails engine

### RAG & AI Pipeline
- **Orchestration**: LangChain (`langchain-core`, `langchain-openai`, `langchain-chroma`)
- **Vector Database**: ChromaDB (persistent local SQLite + vector index)
- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **Language Model**: `nvidia/nemotron-3.5-super-vl:free` via OpenRouter (or standard OpenAI models)

---

## ✨ Key Features

1. **ChatGPT-Style Multi-Session Interface**:
   - Sidebar listing previous conversations.
   - **+ New Chat** button for fresh dialogues.
   - Automatically renames chats based on your first question preview.
   - Ability to delete unwanted conversation sessions.

2. **Persistent Chat History with SQLite**:
   - Conversations are not lost on page reload.
   - Zero reliance on brittle browser `localStorage` for messages.
   - Auto-reconnects to the session you were currently viewing on refresh.
   - Automatic new chat creation when the backend server restarts.

3. **Contextual Query Rewriting**:
   - Handles multi-turn questions like *"Where did they work before?"* by resolving pronouns against preceding dialogue turns.

4. **Two-Stage Retrieval & Reranking**:
   - Stage 1: Similarity search across knowledge base chunks using embeddings.
   - Stage 2: LLM document reranker evaluates chunks and ranks by highest relevance.

5. **ChatGPT-Grade Security Guardrails**:
   - **Anti-DoS**: Enforces length bounds (1–2,000 characters).
   - **Prompt Injection Defense**: Blocks system overrides, jailbreak personas (*DAN*, *Developer Mode*), and instruction-leaking tricks.
   - **Harmful Content Filtering**: Deflects malware, exploits, and attack requests.
   - **PII & Credential Redaction**: Automatically masks Social Security Numbers, Credit Card numbers, and API keys (`sk-...`) from both user input and AI output.

6. **Markdown Formatting**:
   - Automatically formats text wrapped in `**bold**` or `***bold***` into clean HTML `<strong>` tags without displaying raw markdown asterisks.

---

## 📁 Project Directory Structure

```
InsureLLM/
├── api/
│   ├── main.py               # FastAPI application & API endpoints
│   ├── database.py           # SQLite database operations (sessions & messages)
│   ├── guardrails.py         # Security & safety policies
│   └── requirements.txt      # Backend Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── chatApi.js    # Axios client & API caller functions
│   │   ├── components/
│   │   │   ├── Sidebar.jsx   # Multi-session conversations sidebar
│   │   │   ├── ChatMessage.jsx# Message bubble with bold text formatting
│   │   │   └── ChatInput.jsx # Input bar with send button
│   │   ├── App.jsx           # Root application component
│   │   ├── App.css           # Clean CSS styling
│   │   └── main.jsx          # TanStack Query & React DOM entry point
│   ├── index.html            # Vite HTML shell
│   └── package.json          # Frontend dependencies & scripts
│
├── implementation/
│   ├── answer_langchain.py   # Core RAG pipeline (retrieval, rerank, generation)
│   └── ingest_langchain.py   # Ingestion script to build Chroma vector database
│
├── knowledge-base/           # Markdown domain documents
│   ├── company/              # About, careers, culture, overview
│   ├── contracts/            # Enterprise client contracts & agreements
│   ├── employees/            # Staff directories and executive biographies
│   └── products/             # Carllm, Healthllm, Lifellm, Homellm, Bizllm, etc.
│
├── preprocessed_db/          # Persistent Chroma vector database
├── chat_history.db           # SQLite database for conversation history
├── .env.example              # Template environment variables
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python**: 3.12 or higher
- **Node.js**: v18 or higher (includes `npm`)
- **OpenRouter / OpenAI API Key**

---

### 2. Backend Setup

1. **Navigate to the root directory**:
   ```bash
   cd d:\InsureLLM
   ```

2. **Activate the virtual environment**:
   - In PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - In Command Prompt:
     ```cmd
     .venv\Scripts\activate.bat
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory (refer to `.env.example`):
   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_MODEL=nvidia/nemotron-3.5-super-vl:free
   EMBEDDING_MODEL=text-embedding-3-small
   ```

5. **Start the FastAPI Server**:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   - API is live at: `http://localhost:8000`
   - Interactive Swagger Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

---

### 3. Frontend Setup

1. **Open a new terminal and navigate to `frontend`**:
   ```bash
   cd d:\InsureLLM\frontend
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Configure API URL** (optional, defaults to `http://localhost:8000`):
   Verify `frontend/.env`:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

4. **Start Vite Dev Server**:
   ```bash
   npm run dev
   ```

5. **Open in Browser**:
   Open **`http://localhost:5173`** in your browser.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check and boot ID timestamp |
| `GET` | `/api/sessions` | Fetch all conversation sessions ordered by newest first |
| `POST` | `/api/sessions` | Create a new conversation session |
| `GET` | `/api/sessions/{session_id}/messages` | Retrieve all message turns for a session from SQLite |
| `DELETE` | `/api/sessions/{session_id}` | Delete a session and its associated message turns |
| `POST` | `/api/chat` | Send a question; triggers guardrails, SQLite logging, and RAG |

### Sample Chat Request:
```json
POST /api/chat
Content-Type: application/json

{
  "question": "Who is the CEO of Insurellm?",
  "session_id": "8f27fa13-3c8a-437c-9836-e0428071a5dd"
}
```

### Sample Chat Response:
```json
{
  "answer": "Avery Lancaster is the Co-Founder and CEO of Insurellm. She co-founded the company in 2015 and has served in that role since.",
  "chunks": [
    {
      "content": "Avery Lancaster is the Co-Founder and Chief Executive Officer...",
      "source": "knowledge-base\\employees\\Avery Lancaster.md"
    }
  ]
}
```

---

## 🔒 Security Policies & Guardrails

The application intercepts inputs and outputs via [`api/guardrails.py`](api/guardrails.py) before requests reach the RAG pipeline:

```
User Query ──► [Length Validation] ──► [Jailbreak Detection] ──► [Malicious Content Filter] ──► [PII Masking] ──► RAG
```

- **Policy Refusal Message**:
  > *"I cannot fulfill this request. My role is to safely assist with questions regarding Insurellm, its products, employees, and contracts, in accordance with security and safety policies."*
- **Safety Testing**:
  - `DAN mode / jailbreak` ➔ **Refused immediately (0ms latency, zero API costs)**
  - `Ignore previous instructions` ➔ **Refused**
  - `Reveal system prompt` ➔ **Refused**
  - `Sk-123456789...` ➔ **Masked to `[REDACTED_API_KEY]`**

---

## 📄 License
This project is developed for educational and portfolio demonstration purposes. All rights reserved.

