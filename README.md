# EmbedOps — AI Data Engineering Platform for RAG

**EmbedOps** is a production-minded AI data engineering platform that powers **semantic retrieval and Retrieval-Augmented Generation (RAG)** workflows.

This project focuses on the **hard parts most demos skip**:

- ingestion  
- chunking  
- embeddings  
- vector indexing  
- grounded retrieval  
- clean LLM integration  

This is **not a chatbot demo**.  
It is a **retrieval platform** that LLMs can safely consume.

---

## Why this project exists

Most RAG examples tightly couple:

- ingestion  
- embeddings  
- vector search  
- LLM inference  

into a single script.

That approach does not scale.

**EmbedOps separates concerns**:

- batch pipelines for data & embeddings  
- an online API for retrieval  
- optional LLMs as *consumers*, not dependencies  

This mirrors real-world AI platform design.

---

## High-level Architecture

```text
                ┌──────────────────┐
                │  Raw Documents   │
                │  (PDFs)          │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Pipeline        │
                │  (Batch Job)     │
                │                  │
                │  • Ingest PDFs   │
                │  • Chunk text    │
                │  • Embed chunks  │
                │  • Upsert vectors│
                │  • Write chunks  │
                └────────┬─────────┘
                         │
        ┌────────────────┴───────────────┐
        │                                │
        ▼                                ▼
┌──────────────────┐           ┌──────────────────┐
│  Pinecone        │           │  Chunk Store     │
│  (Vectors + MD) │           │  (JSONL)         │
└──────────────────┘           └──────────────────┘
                                         ▲
                                         │
                                ┌────────┴────────┐
                                │  Retrieval API  │
                                │  (FastAPI)     │
                                │                │
                                │  /query        │
                                │  /rag/answer   │
                                └────────┬────────┘
                                         │
                                ┌────────┴────────┐
                                │  LLM Runtime    │
                                │  (Ollama)      │
                                │  llama3.1      │
                                └─────────────────
```

## Key Design Principles

- Retrieval is infrastructure
- LLMs are optional consumers
- No hallucination without context
- Batch + service separation
- Docker-first, reproducible execution
- Minimal Python dependencies

---

## Tech Stack

### Core
- **Python 3.11**
- **FastAPI** – retrieval & RAG API
- **SentenceTransformers** – embeddings
- **Pinecone** – vector database
- **Docker & Docker Compose**

### LLM (Free, Local, Cross-platform)
- **Ollama** (Dockerized)
- **llama3.1**

> No OpenAI / ChatGPT required  
> No paid APIs  
> No Python LLM dependencies

---

## Prerequisites (from scratch)

### 1️⃣ Install Docker & Create Pinecone API key
Required for all platforms.

- **Mac / Windows**  
  https://www.docker.com/products/docker-desktop

- **Linux**  
  https://docs.docker.com/engine/install/

- **Create pinecone vector DB API - Free 2 GB serverless storage**

  https://docs.pinecone.io/reference/api/authentication

Verify installation:
```bash
docker --version
docker compose version
````

### 2️⃣ Clone the repository

```bash
git clone https://github.com/mrugesh1989/embedops-ai-data-platform.git
cd embedops-ai-data-platform
```

### 3️⃣ Add documents
- **Place PDFs here:**
```text 
-  data/raw/
```

### 4️⃣ Configure environment variables
- **Create .env file at the root level:**
```.dotenv
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://host.docker.internal:11434"
OLLAMA_MODEL="llama3.1"
PINECONE_API_KEY="{Your_API_KEY}"
PINECONE_INDEX_NAME="embedops-rag-docs"
PINECONE_CLOUD="aws"
PINECONE_REGION="us-east-1"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
EMBEDDING_NAMESPACE="emb_v1"
UPSERT_BATCH_SIZE=200
```
### Run Everything (One Command)
- **Execute the below command at the root level of Repository**
```bash
docker compose up --build 
or
make up
```

### What happens automatically

- **Ollama starts (LLM runtime)**
- **Model is pulled automatically (llama3.1)**
- **Pipeline runs once**
	•	ingests documents
	•	chunks text
	•	generates embeddings
	•	upserts vectors
	•	writes chunk store
- **starts only after data is ready**

### Verify Services
```bash 
docker compose ps
```
- **Health check:**
```bash
curl http://localhost:8000/health
```
- **Health check Expected response:**
```json
{
  "status": "ok",
  "retrieval_ready": true,
  "llm_enabled": true
}
```

### Using the API - swagger UI

- **Open in Browser**
```text
http://localhost:8000/docs
```
- **Retrieval only (/query) from pinecone DB - semantic search Result without LLM**
```json
{
  "query": "Summarize designing notification system",
  "top_k": 5,
  "score_threshold": 0.4
}
```
- **Returns:**
	•	ranked similarity results
	•	document provenance
	•	grounded text previews

- **RAG answer (/rag/answer) - Using Ollama LLM**
```json
{
  "query": "Summarize designing notification system",
  "top_k": 5,
  "score_threshold": 0.4
}
```

## This Project demonstrates:
	•	AI-ready data pipelines
	•	Vector database integration
	•	Chunking & embedding strategies
	•	Metadata-aware retrieval
	•	Grounded RAG (no hallucination)
	•	Clean separation of concerns
	•	Dockerized, reproducible systems

- ***This is AI Data Engineering / ML Platform Engineering, not prompt engineering.***

## What’s intentionally excluded
	•	No LangChain
	•	No LlamaIndex
	•	No OpenAI SDK
	•	No hidden notebooks
	•	No tight LLM coupling

    This keeps the system:
	•	portable
	•	debuggable
	•	production-friendly

## Future Extensions (Optional)
	•	Multi-tenant namespaces
	•	Embedding version rollouts
	•	Streaming responses
	•	Metrics & monitoring
	•	External LLM providers
    None of the above require re-architecting.

## Author

- **Mrugesh Patel**
- **Senior Data Engineer → AI Data Engineer**