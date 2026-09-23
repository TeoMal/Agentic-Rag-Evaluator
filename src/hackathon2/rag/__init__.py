"""Retrieval over the NFS knowledge pack (Deep Agent / RAG engineer).

FR03 index + retrieve, FR04 cite sources, FR05 evidence vs inference vs missing,
FR11 contradictions/UNKNOWN. Corpus lives in knowledge/ (Settings.knowledge_dir).
Start from course units:
  section-12-rag/60-doc-loaders-splitters  load + chunk, keep `source` metadata for citations
  section-12-rag/62-postgres-pgvector      PGVector store (Settings.sqlalchemy_database_url)
  section-12-rag/63-agentic-rag            retriever exposed as a tool
  section-12-rag/64-advanced-retrieval     MMR + LLM rerank
Embeddings: hackathon2.llm.get_embeddings() (Azure OpenAI, not Cohere as in class).
"""
