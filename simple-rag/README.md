# AI Simple RAG

A minimal **Retrieval-Augmented Generation (RAG)** API built with TypeScript and Node.js.

This project demonstrates the core RAG pipeline without relying on a large framework abstraction:

**Document → Text Extraction → Chunking → Embeddings → Vector Search → Context → LLM → Answer**

The application allows documents to be uploaded, indexed into PostgreSQL using `pgvector`, and queried using natural-language questions. Relevant document chunks are retrieved using vector similarity search and provided to a vLLM-served LLM (vllm-mlx) to generate grounded answers with source references.

## ✨ Features

* 📄 Upload and process **PDF, Markdown, and plain-text documents**
* ✂️ Split documents into overlapping text chunks
* 🧠 Generate vector embeddings for document chunks
* 🗄️ Store embeddings in **PostgreSQL + pgvector**
* 🔎 Perform semantic similarity search using cosine distance
* 🎯 Filter retrieved chunks using a configurable similarity threshold
* 🤖 Generate answers using **vLLM (vllm-mlx)**
* 📚 Return source information with generated answers
* 🚫 Prevent the LLM from answering from outside knowledge
* ⚡ Fast HTTP API built with **Fastify**
* 🐳 PostgreSQL and pgAdmin available through Docker Compose
* 🔐 Environment-based configuration with `.env`

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │     Document    │
                    │  PDF / MD / TXT │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Text Extraction│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Chunking     │
                    │ 450 words/chunk │
                    │ 70 word overlap │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Embedding    │
                    │      Model      │
                    └────────┬────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ PostgreSQL + pgvector  │
                 │                         │
                 │ Documents               │
                 │ Document Chunks         │
                 │ Embeddings              │
                 └────────────┬────────────┘
                              │
                     User Question
                              │
                              ▼
                    ┌─────────────────┐
                    │ Query Embedding │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Vector Similarity│
                    │     Search      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevant Chunks │
                    │ + Similarity    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   vLLM LLM      │
                    │ Context + Query │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Grounded Answer │
                    │ + Sources       │
                    └─────────────────┘
```

## 🔄 How RAG Works

### 1. Document ingestion

The API accepts supported documents and extracts their text.

Currently supported formats:

* PDF
* Markdown
* Plain text

The extracted text is stored in the `documents` table.

### 2. Text chunking

Large documents are divided into smaller chunks before generating embeddings.

The current implementation uses:

* **Chunk size:** 450 words
* **Overlap:** 70 words

The overlap helps preserve context between neighboring chunks.

### 3. Generate embeddings

Each chunk is converted into a numerical vector using the configured embedding model.

The embeddings are stored in PostgreSQL using the `pgvector` extension.

### 4. Semantic retrieval

When a user asks a question, the question is converted into an embedding.

The system then searches the stored vectors using PostgreSQL's vector similarity functionality and returns the most relevant chunks.

Similarity is calculated from cosine distance:

```text
similarity = 1 - cosine_distance
```

The number of retrieved chunks is controlled by `RAG_TOP_K`.

### 5. Similarity filtering

Retrieved chunks are filtered using `RAG_MIN_SIMILARITY`.

If no chunk passes the threshold, the system returns:

```text
I could not find this in the provided documents.
```

This prevents the LLM from being given unrelated context.

### 6. Answer generation

The retrieved chunks are assembled into a context and sent to the vLLM LLM.

The model is instructed to:

1. Answer only using the supplied context.
2. Avoid using outside knowledge.
3. Explicitly state when the answer cannot be found.
4. Cite retrieved context using `[Source N]`.

This keeps the generated response grounded in the indexed documents.

## 🛠️ Tech Stack

| Technology     | Purpose                              |
| -------------- | ------------------------------------ |
| TypeScript     | Application language                 |
| Node.js        | Runtime                              |
| Fastify        | HTTP API                             |
| PostgreSQL     | Persistent database                  |
| pgvector       | Vector storage and similarity search |
| vLLM (vllm-mlx) | LLM and embedding provider (OpenAI-compatible API) |
| `pdf-parse`    | PDF text extraction                  |
| Zod            | Environment/input validation         |
| Docker Compose | Local infrastructure                 |

The project dependencies include Fastify, `@fastify/multipart`, the AI SDK OpenAI-compatible provider, PostgreSQL's `pg` client, `pdf-parse`, Zod, and related TypeScript tooling.

## 📁 Project Structure

```text
ai-simple-rag/
├── src/
│   ├── ai.ts          # Embedding and LLM integration
│   ├── db.ts          # PostgreSQL connection
│   ├── env.ts         # Environment configuration
│   ├── migrate.ts     # Database migrations
│   ├── rag.ts         # Document ingestion, retrieval and RAG logic
│   └── server.ts      # Fastify HTTP server
│
├── samples/           # Example documents (txt, pdf)
├── docker-compose.yml # PostgreSQL + pgAdmin
├── .env.example       # Environment variable template
├── package.json
├── tsconfig.json
└── README.md
```

The repository separates the RAG logic, database connection, AI integration, environment configuration, migrations, and HTTP server into individual modules.

## 🚀 Getting Started

### Prerequisites

Make sure you have installed:

* Node.js
* npm
* Docker
* Docker Compose
* A running vllm-mlx server (`make run` in the parent directory, port 8100)

### 1. Clone the repository

```bash
git clone https://github.com/sunflowerIU/ai-simple-rag.git

cd ai-simple-rag
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start PostgreSQL and pgAdmin

```bash
docker compose up -d
```

The project uses the `pgvector/pgvector:pg17` image for PostgreSQL.

Default local ports:

```text
PostgreSQL → localhost:5433
pgAdmin    → localhost:5050
```

PostgreSQL data is persisted using a Docker volume.

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then configure your database and vLLM endpoint.

Example:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ai_rag

VLLM_BASE_URL=http://localhost:8100/v1
VLLM_MODEL=mlx-community/Qwen3-8B-4bit
EMBEDDING_MODEL=mlx-community/bge-m3-mlx-fp16
EMBEDDING_DIMENSIONS=1024

RAG_TOP_K=5
RAG_MIN_SIMILARITY=0.5
```

> Do not commit your `.env` file or API keys to Git.

### 5. Run database migrations

```bash
npm run migrate
```

### 6. Start the development server

```bash
npm run dev
```

The server runs using Fastify and `tsx` in development mode.

## 🔌 RAG Pipeline

The main RAG implementation lives in `src/rag.ts`.

### Ingestion

```text
File
 ↓
Extract text
 ↓
Create document
 ↓
Split into chunks
 ↓
Generate embeddings
 ↓
Store chunks + embeddings
```

### Query

```text
Question
 ↓
Generate query embedding
 ↓
Vector similarity search
 ↓
Retrieve top K chunks
 ↓
Apply similarity threshold
 ↓
Build context
 ↓
Send context + question to LLM
 ↓
Return answer + sources
```

The implementation uses separate embedding prefixes for document search and query search, matching the expected format of the configured embedding model.

## 📡 API

The application exposes HTTP endpoints through Fastify.

### Upload a document

Upload a supported document to index it for retrieval.

```http
POST /documents
Content-Type: multipart/form-data
```

### Ask a question

Ask a question against the indexed documents.

```http
POST /ask
Content-Type: application/json
```

Example request:

```json
{
  "question": "What is the company's leave policy?"
}
```

You can optionally restrict retrieval to a particular document:

```json
{
  "question": "How many days of annual leave are provided?",
  "documentId": "document-id"
}
```

The RAG layer supports both global retrieval across documents and retrieval restricted to a specific document.

## 📚 Response Sources

Responses include information about the chunks used to generate the answer.

Example:

```json
{
  "answer": "Employees are entitled to ... [Source 1]",
  "sources": [
    {
      "label": "Source 1",
      "documentId": "document-id",
      "chunkId": "chunk-id",
      "chunkIndex": 2,
      "similarity": 0.87,
      "preview": "Employees are entitled to..."
    }
  ]
}
```

This makes the generated answer more transparent by showing which retrieved document chunks contributed to the response.

## 🧠 Why RAG?

A language model's built-in knowledge is not a reliable database for private documents.

RAG solves this by retrieving relevant information from an external knowledge source and placing it into the model's context at query time.

For example:

```text
User:
"What is our refund policy?"

        ↓

Vector Search

        ↓

Relevant policy chunks

        ↓

LLM + Retrieved Context

        ↓

"According to the policy, refunds are..."
```

The model doesn't need to memorize the company's policy. It retrieves the relevant information when the question is asked.

## ⚙️ Configuration

The RAG pipeline exposes configuration for retrieval behavior, including:

```env
RAG_TOP_K=5
RAG_MIN_SIMILARITY=0.5
```

### `RAG_TOP_K`

Controls how many candidate chunks are retrieved from the vector database.

A larger value provides more context but can also introduce irrelevant information.

### `RAG_MIN_SIMILARITY`

Controls the minimum similarity required for a retrieved chunk to be considered relevant.

This acts as a simple guard against answering questions using weakly related documents.

## 🐳 Docker

The included Docker Compose configuration provides:

```text
PostgreSQL + pgvector
        +
      pgAdmin
```

PostgreSQL uses the `pgvector/pgvector:pg17` image, while pgAdmin provides a browser-based database administration interface.

Start:

```bash
docker compose up -d
```

Stop:

```bash
docker compose down
```

Stop and remove volumes:

```bash
docker compose down -v
```

> Removing volumes permanently deletes the local PostgreSQL data.

## 🧪 Type Checking

Run TypeScript type checking with:

```bash
npm run typecheck
```

## 🎯 Project Goals

This project is intentionally minimal.

The goal is to understand the fundamental pieces behind a RAG application:

* Document ingestion
* Text extraction
* Chunking
* Embeddings
* Vector databases
* Similarity search
* Retrieval
* Context construction
* LLM generation
* Source attribution
* Similarity thresholds

Rather than hiding these concepts behind a high-level RAG framework, the implementation keeps the core pipeline visible in the application code.

## 🚧 Possible Improvements

Some natural next steps for the project include:

* [ ] Add authentication
* [ ] Add document deletion
* [ ] Add duplicate document detection
* [ ] Add metadata filtering
* [ ] Add HNSW indexing for larger collections
* [ ] Add hybrid keyword + vector search
* [ ] Add reranking
* [ ] Add conversation memory
* [ ] Add streaming LLM responses
* [ ] Add automated RAG evaluation
* [ ] Add a frontend chat interface
* [ ] Add document/page-level citations
* [ ] Add support for DOCX and other formats
* [ ] Add production Docker configuration
* [ ] Add automated tests

## 📄 License

ISC

```
```
