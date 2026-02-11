# InvoiceInsight - AI Invoice Question Answering

🔗 **Live Demo:** https://invoice-insight-cloudlamp.streamlit.app

## What It Does
Upload invoice PDFs and ask questions in natural language. Built with RAG (Retrieval-Augmented Generation) to provide accurate answers based on document content.

## Features
- PDF text extraction and intelligent chunking
- Semantic search with OpenAI embeddings
- Context-aware question answering with GPT-4
- Production safeguards (rate limiting, file validation, cost controls)

## Tech Stack
- **Backend:** Python
- **UI:** Streamlit
- **AI:** OpenAI Embeddings API + GPT-4o-mini
- **Vector Search:** Cosine similarity with NumPy

## Architecture
[Add a simple diagram or explanation of your RAG pipeline]

## What I Learned
- RAG implementation from scratch
- Working with embeddings and vector similarity
- Production-ready API usage (rate limiting, error handling)
- Deploying AI applications

## Future Improvements
- Vector database integration (Pinecone/pgvector)
- Structured data extraction
- Multi-document querying
- Export capabilities
