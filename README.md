# Pluang AI Knowledge Base Copilot

A possible solution could be an internal AI assistant developed with the Retrieval-Augmented Generation model that could help the employees search company documents in an efficient way.

The system retrieves relevant data from internal documents and gives answers based on Large Language Models, along with citations.

<img width="1914" height="966" alt="image" src="https://github.com/user-attachments/assets/5ddfa263-9502-4d55-ad51-e5c90666d02a" />


https://github.com/user-attachments/assets/83d9cc63-4df3-4c40-ae81-94775612da04

## Motivation

Employees of large organizations are required to spend a considerable amount of time going through policies, guidelines, and manuals within the organizations.

Traditional Keyword Searches

- No semantic meaning
- Returns irrelevant results
- Does not summarize any information

The goal of this project is to create an **AI copilot** that:

- Understands User intent
- Retrieve relevant internal content
- Generates dependable answers
- Prevents confusions
- Reveals source references

## Core Concept: Retrieval-Augmented Generation (RAG)

This project wokrs on RAG architecture.

Instead of directly asking an LLM:

> "Answer from your training data"

We use:

> "Answer only from verified company documents"

---

## System Architecture

### High-Level Architecture

```

Frontend (Streamlit)
↓
Query Handler
↓
Embedding Model
↓
FAISS Vector Database
↓
Document Retriever
↓
Prompt Builder
↓
Groq LLM API
↓
Answer Generator

```

---

## Project Structure

```

Pluang Assignment/
│
├── backend/
│   ├── vector_db/
│   │   ├── index.faiss
│   │   └── meta.pkl
│   └── ingest.py
│
├── frontend/
│   └── app.py
│
├── .env
├── requirements.txt
└── README.md

```

---

## Backend: Document Processing & Indexing

### 1️ Document Ingestion

All the documents are loaded from defined folders.

Each document is:

- Read
- Cleaned
- Chunked

Because: Large language models, as well as embedding models, tend to perform best with coherent blocks of text, provided they are kept smaller.

---

### 2️ Text Chunking

Document broken down into overlapping sections.

Example:

```

Chunk Size: 150 tokens
Overlap: 30 tokens

```

Because: It does not lose its context, Improves semantic retrieval.

---

### 3️ Embedding Generation

Used Model:

```

all-MiniLM-L6-v2 (Sentence Transformers)

```

Why?

- Lightweight
- High semantic accuracy
- Low Latency of Inference

Each chunk → 384-dimension vector.

## These vectors are representations of semantic meaning.

### 4️ Vector Storage (FAISS)

FAISS is used as the vector database.

Why FAISS?

Optimized for similarity search, Fast nearest neighbor search, Scales to millions of vectors, Industry-wide implementation

Index Type:

```
IndexFlatL2

```

Used because of its

- High accuracy
- Suitablity with medium datasets

Metadata is stored separately using pickle.

---

## Frontend: Query Processing

### 1️ User Query Input

I have used Streamlit for this.

Features:

- Highlighted input
- Persistent chat history

## Session state is used to preserve conversations.

### 2️ Query Embedding

The user query is embedded using the same embedding model.

## Because: Comparison of a vector is applicable only if it is in the same embedding space.

### 3️ Similarity Search

FAISS carries out k-nearest neighbor search.

```
Top-K = 4

```

## Returns most relevant chunks.

### 4️ Context Construction

Retrieved data is organized in the following format:

```
[Source 1] ...
[Source 2] ...

```

This increases traceability.

---

## LLM Integration (Groq API)

### Model Used

```

llama-3.1-8b-instant

```

Why?

- Low latency
- Cost-efficient
- High reasoning quality
- Good instruction following

---

### Prompt Engineering

The model is given strict instructions:

- Use only provided context
- Avoid outside knowledge
- Cite sources
- Provide clear answers

Example Prompt:

```

You are an internal knowledge assistant.
Use ONLY the information below...

```

This reduces the confusions.

---

## Conversation Management

Chat history is stored using:

```

st.session_state

```

Benefits:

- Multi-turn conversations
- Context continuity.

---

## Setup Instructions

### 1️ Clone Repository

```bash
git clone <repo-url>
cd Pluang-Assignment
```

### 2️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️ Configure Environment

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 4️ Build Vector Database

```bash
python backend/ingest.py
```

### 5️ Run Application

```bash
streamlit run frontend/app.py
```

## Challenges & Solutions

### 1️ API Deprecations

Issue:
Multiple LLM models were deprecated. I tried the Gemini API it was giving limit issue.

Solution:
Implemented dynamic model listing and migrated to Groq-supported models.

---

### 2 Hallucination Control

Issue:
LLM generating external info.

Solution:
Strict prompt design + limited context.

---

## Future Improvements

- Role-based authentication
- Document upload UI
- Docker containerization
- Query analytics dashboard
- Hybrid search (BM25 + vectors)

---

```

```
