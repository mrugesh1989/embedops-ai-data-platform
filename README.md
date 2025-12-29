# embedops-ai-data-platform
Production-grade AI data engineering platform for document ingestion, embedding pipelines, and vector search using Pinecone and Docker.

## Architecture (End-to-End Flow)

```mermaid
flowchart LR
  A[PDFs in data/raw/] --> B[Pipeline Container<br/>Ingest + Chunk + Embed]
  B --> C[Pinecone Serverless Index<br/>Vectors + Metadata]
  B --> D[Chunk Store<br/>data/processed/chunks.jsonl]
  D --> E[API Container<br/>FastAPI Retrieval Service]
  C --> E
  E --> F[Client<br/>curl / Swagger UI / app]

  subgraph Docker Compose
    B
    E
    D
  end