"""
Simple LangChain version of the Insurellm ingestion code.

What this code does:
1. Read all .md files from the knowledge-base folder
2. Ask the LLM to split each file into chunks (headline + summary + original text)
3. Turn every chunk into a LangChain "Document"
4. Save all the Documents in the Chroma database (with embeddings)

Install first:
pip install langchain-core langchain-openai langchain-chroma python-dotenv pydantic tqdm

Your .env file needs:
OPENAI_API_KEY=...   (used for the chat model and for embeddings)

Run it with:
python ingest_langchain.py
"""

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
from tqdm import tqdm

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


# 
# Step 1: Settings
# 

# Load the keys from the .env file
load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-super-vl:free")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
USE_LLM_CHUNKING = os.getenv("USE_LLM_CHUNKING", "false").lower() in ("true", "1")

# Same folders as before
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

COLLECTION_NAME = "docs"
AVERAGE_CHUNK_SIZE = 600



# Step 2: The shape of the LLM answer


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )


class Chunks(BaseModel):
    chunks: list[Chunk]


# 
# Step 3: Create the LangChain objects
# 

# The chat model.
# max_retries = if a call fails (for example rate limit), LangChain tries again by itself.
# This replaces the "tenacity" retry code from before.
llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0,
    max_retries=5,
)

# This makes the LLM reply in the "Chunks" shape
structured_llm = llm.with_structured_output(Chunks)

# The embedding model (turns text into numbers so we can search)
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
)


# 
# Step 4: Functions
# 

def fetch_documents():
    """Read all .md files from the knowledge-base folder."""
    documents = []

    # Every folder inside knowledge-base is a "type" (for example: employees, products ...)
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name

        # Find all .md files inside this folder (also in sub-folders)
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append(
                {"type": doc_type, "source": file.as_posix(), "text": text}
            )

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    """Build the prompt that tells the LLM how to split one document."""
    # A rough idea of how many chunks we want
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1

    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def split_document_sections(document, max_chars=800, overlap=150):
    """Split markdown document into semantic chunks with headings and overlap."""
    text = document["text"]
    lines = text.split("\n")

    doc_title = document["type"].capitalize()
    sections = []
    current_heading = doc_title
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = stripped.lstrip("#").strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    docs = []
    for heading, content in sections:
        if not content:
            continue
        if len(content) <= max_chars:
            page_content = f"{heading}\n\n{content}"
            docs.append(Document(page_content=page_content, metadata={"source": document["source"], "type": document["type"]}))
        else:
            start = 0
            while start < len(content):
                end = min(start + max_chars, len(content))
                piece = content[start:end]
                page_content = f"{heading}\n\n{piece}"
                docs.append(Document(page_content=page_content, metadata={"source": document["source"], "type": document["type"]}))
                if end == len(content):
                    break
                start += max_chars - overlap
    return docs or [Document(page_content=text, metadata={"source": document["source"], "type": document["type"]})]


def process_document(document):
    """Split ONE document into chunks, and return them as LangChain Documents."""
    if USE_LLM_CHUNKING:
        try:
            prompt = make_prompt(document)
            result = structured_llm.invoke([HumanMessage(content=prompt)])

            langchain_docs = []
            for chunk in result.chunks:
                page_content = chunk.headline + "\n\n" + chunk.summary + "\n\n" + chunk.original_text
                metadata = {"source": document["source"], "type": document["type"]}
                langchain_docs.append(Document(page_content=page_content, metadata=metadata))

            if langchain_docs:
                return langchain_docs
        except Exception as e:
            print(f"\nWarning: LLM chunking failed for {document['source']}: {e}. Using fast section chunking.")

    return split_document_sections(document)


def create_chunks(documents):
    """Go through all documents, one by one, and collect all chunks."""
    all_chunks = []

    # tqdm shows a progress bar
    for document in tqdm(documents):
        chunks = process_document(document)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks")
    return all_chunks


def create_embeddings(chunks, batch_size=50):
    """Save all chunks in the Chroma database in safe batches."""
    # First delete the old collection, so we do not get duplicates.
    try:
        old_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_NAME,
            embedding_function=embeddings,
        )
        old_store.delete_collection()
    except Exception:
        pass

    # Now make a fresh one and populate in batches
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_NAME,
        embedding_function=embeddings,
    )

    print(f"Creating embeddings for {len(chunks)} chunks in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Ingested {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")

    count = vectorstore._collection.count()
    print(f"Vectorstore created with {count} documents")


# 
# Step 5: Run everything
# 

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
