---
name: document-tools
description: Extract text and tables from common local document formats.
allowed-tools: extract_document_text extract_document_tables
metadata:
  version: "1.0.0"
  tags:
    - documents
    - pdf
    - docx
    - csv
    - text
---
Use document-tools to inspect user documents.

Rules:
- Document extraction is read-only and can run directly for explicit local paths.
- TXT and CSV work with the base install.
- PDF and DOCX require optional Python dependencies; if missing, report the dependency error clearly.
- Return concise extracted text or table rows; do not invent content.
