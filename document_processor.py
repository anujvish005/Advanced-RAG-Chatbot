"""
Document Processor - PDF, TXT, DOCX
Place this file in the SAME folder as app.py
"""

import io
from typing import List, Dict, Any


class DocumentProcessor:

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, uploaded_file) -> List[Dict[str, Any]]:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            return self._process_pdf(uploaded_file, uploaded_file.name)
        elif ext == "txt":
            return self._process_txt(uploaded_file, uploaded_file.name)
        elif ext == "docx":
            return self._process_docx(uploaded_file, uploaded_file.name)
        raise ValueError(f"Unsupported file type: {ext}")

    def _process_pdf(self, file, name):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file.read()))
        chunks = []
        for page_num, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                for c in self._split(text):
                    chunks.append({"text": c, "source": name, "page": page_num + 1})
        return chunks

    def _process_txt(self, file, name):
        content = file.read().decode("utf-8", errors="ignore")
        return [{"text": c, "source": name, "page": "N/A"} for c in self._split(content)]

    def _process_docx(self, file, name):
        from docx import Document
        doc  = Document(io.BytesIO(file.read()))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": c, "source": name, "page": "N/A"} for c in self._split(text)]

    def _split(self, text: str) -> List[str]:
        """Recursive character splitter with overlap."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        for sep in ["\n\n", "\n", ". ", " "]:
            parts = text.split(sep)
            if len(parts) == 1:
                continue

            chunks, current = [], ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    current = part

            if current.strip():
                chunks.append(current.strip())

            if len(chunks) > 1:
                # Apply overlap
                if self.chunk_overlap > 0:
                    overlapped = [chunks[0]]
                    for i in range(1, len(chunks)):
                        tail = chunks[i - 1][-self.chunk_overlap:]
                        overlapped.append(tail + " " + chunks[i])
                    return overlapped
                return chunks

        # Hard split fallback
        step = self.chunk_size - self.chunk_overlap
        return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]