import os
import sys
from pathlib import Path

# Redirect LOCALAPPDATA / patch certifi-win32 for Windows
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

local_appdata = project_root / ".localappdata"
local_appdata.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(local_appdata)
try:
    import certifi_win32.wincerts
    certifi_win32.wincerts.PEM_PATH = str(local_appdata / ".certifi" / "cacert.pem")
except Exception:
    pass

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv(override=True)

# Where the Chroma database is saved
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")

COLLECTION_NAME = "docs"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MODEL = os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-super-vl:free")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY")

RETRIEVAL_K = 20   # chunks to fetch from the database per search
FINAL_K = 10       # chunks to give to the LLM at the end

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0,
    max_retries=5,
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_NAME,
    embedding_function=embeddings,
)


# ---------------------------------------------------------------------------
# Load Prompts from JSON using LangChain load_prompt
# ---------------------------------------------------------------------------
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

system_prompt_template = load_prompt(PROMPTS_DIR / "system_prompt.json")
rewrite_prompt_template = load_prompt(PROMPTS_DIR / "rewrite_prompt.json")
rerank_prompt_template = load_prompt(PROMPTS_DIR / "rerank_prompt.json")

SYSTEM_PROMPT = system_prompt_template.template
REWRITE_SYSTEM_PROMPT = rewrite_prompt_template.template
RERANK_PROMPT = rerank_prompt_template.template


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_messages(history) -> list[BaseMessage]:
    """Convert raw dict history or message list into LangChain BaseMessage objects."""
    if not history:
        return []
    messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(item, BaseMessage):
            messages.append(item)
    return messages


def merge_chunks(chunks, other_chunks):
    """Join two lists of chunks, skipping duplicates."""
    merged = list(chunks)
    seen_texts = {chunk.page_content for chunk in chunks}

    for chunk in other_chunks:
        if chunk.page_content not in seen_texts:
            merged.append(chunk)
            seen_texts.add(chunk.page_content)

    return merged


def rerank(question, chunks):
    """Ask the LLM to sort the chunks by relevance, most relevant first."""
    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks

    user_prompt = f"The user has asked the following question:\n\n{question}\n\n"
    user_prompt += "Order all the chunks of text by relevance to the question, from most relevant to least relevant. "
    user_prompt += "Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"

    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"

    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    try:
        structured_llm = llm.with_structured_output(RankOrder)
        result = structured_llm.invoke(
            [
                SystemMessage(content=RERANK_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        reranked = []
        used_ids = set()
        for chunk_id in result.order:
            if 1 <= chunk_id <= len(chunks) and chunk_id not in used_ids:
                reranked.append(chunks[chunk_id - 1])
                used_ids.add(chunk_id)

        # Add any missed chunks at the end so nothing is lost
        for i in range(len(chunks)):
            if (i + 1) not in used_ids:
                reranked.append(chunks[i])

        return reranked
    except Exception as e:
        print(f"Rerank failed: {e}. Returning unranked chunks.")
        return chunks


def fetch_context_unranked(question):
    """Search the vector database and return matching chunks."""
    return vectorstore.similarity_search(question, k=RETRIEVAL_K)


def fetch_context(original_question, rewritten_question=None):
    """Retrieve unranked chunks for original + rewritten query, merge, and rerank."""
    if not rewritten_question:
        rewritten_question = original_question

    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)

    merged = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, merged)

    return reranked[:FINAL_K]


def format_context_for_prompt(chunks):
    """Format chunks into a clean context string for the system prompt."""
    context_str = ""
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        context_str += f"Extract from {source}:\n{chunk.page_content}\n\n"
    return context_str


# ---------------------------------------------------------------------------
# LCEL Chains (LangChain Expression Language)
# ---------------------------------------------------------------------------

# 1. Query Rewrite Sub-Chain
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# 2. Final RAG Generation Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


# 3. Master LCEL Chain with Sources
# Orchestrates: Normalization -> Rewrite -> Dual Retrieval & Rerank -> Prompt -> LLM -> Output
rag_chain_with_sources = (
    RunnablePassthrough.assign(
        history=RunnableLambda(lambda x: to_messages(x.get("history", [])))
    )
    | RunnablePassthrough.assign(
        rewritten_question=rewrite_chain
    )
    | RunnablePassthrough.assign(
        chunks=RunnableLambda(lambda x: fetch_context(x["question"], x["rewritten_question"]))
    )
    | RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: format_context_for_prompt(x["chunks"]))
    )
    | RunnablePassthrough.assign(
        answer=(rag_prompt | llm | StrOutputParser())
    )
)

# 4. Standard streamable / invokable chain that outputs the answer string directly
rag_chain = rag_chain_with_sources | RunnableLambda(lambda x: x["answer"])


# ---------------------------------------------------------------------------
# Public API Function (Maintains 100% backward compatibility with api/main.py)
# ---------------------------------------------------------------------------

def answer_question(question, history=None):
    """
    Answer a question using the LCEL RAG pipeline.
    Maintains exact signature and behavior expected by api/main.py and eval scripts.
    Returns: (answer text, list of chunks used)
    """
    if history is None:
        history = []

    # Run through the LCEL chain
    result = rag_chain_with_sources.invoke({
        "question": question,
        "history": history,
    })

    answer = result["answer"]
    chunks = result["chunks"]

    # Record current turn in history list
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    return answer, chunks


if __name__ == "__main__":
    conversation_history = []

    q1 = "Who is the CEO of Insurellm?"
    print(f"\nUser: {q1}")
    answer1, used_chunks1 = answer_question(q1, history=conversation_history)
    print("\nANSWER:\n", answer1)
    print("\nNumber of chunks used:", len(used_chunks1))

    # Follow-up question relying on history
    q2 = "Where did they work before joining?"
    print(f"\nUser: {q2}")
    answer2, used_chunks2 = answer_question(q2, history=conversation_history)
    print("\nANSWER:\n", answer2)
    print("\nNumber of chunks used:", len(used_chunks2))
