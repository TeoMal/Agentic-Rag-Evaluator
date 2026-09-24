"""Treat document content as data, never as instructions (FR09).

Scans chunks for prompt-injection patterns at ingestion (sets SearchHit.suspicious)
and wraps chunk text in <untrusted_document> tags before it reaches an agent.
"""
