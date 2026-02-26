"""
RAG Engine - Core retrieval and generation logic
Place this file in the SAME folder as app.py
"""

import time
import hashlib
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq


class RAGEngine:
    """
    Advanced RAG Engine:
    - Groq LLM (ultra-fast inference)
    - Pinecone vector store
    - HuggingFace embeddings
    - Query rewriting using chat history
    - MMR + Semantic retrieval strategies
    - Source attribution
    """

    SYSTEM_PROMPT = """You are an expert AI assistant that answers questions strictly using the provided document context.

Rules:
1. ONLY use the given context. Do NOT use outside knowledge.
2. If the answer is not in context, say: "I don't have enough information in the uploaded documents."
3. Cite which chunk supports your answer (e.g., [Chunk 1]).
4. Be concise, accurate, and well-structured.
5. Use conversation history to understand follow-up questions.
"""

    def __init__(
        self,
        groq_api_key: str,
        pinecone_api_key: str,
        pinecone_index: str,
        llm_model: str = "llama-3.3-70b-versatile",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        temperature: float = 0.2,
        top_k: int = 5,
        retrieval_strategy: str = "Semantic Search",
    ):
        self.llm_model          = llm_model
        self.temperature        = temperature
        self.top_k              = top_k
        self.retrieval_strategy = retrieval_strategy
        self.index_name         = pinecone_index

        # Groq client
        self.groq_client = Groq(api_key=groq_api_key)

        # Embedding model
        print(f"⏳ Loading embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.embed_dim   = self.embed_model.get_sentence_embedding_dimension()
        print(f"✅ Embedding dim: {self.embed_dim}")

        # Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)
        print("✅ RAGEngine ready!")

    # ── Index management ──────────────────────────────────────────────────────

    def _ensure_index(self):
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"Creating Pinecone index '{self.index_name}' ...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(5)  # wait for index to be ready
        else:
            print(f"Using existing index '{self.index_name}'.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        texts = [c["text"] for c in chunks]
        print(f"Embedding {len(texts)} chunks ...")
        embeddings = self.embed_model.encode(texts, show_progress_bar=True, batch_size=32)

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            uid = hashlib.md5(f"{chunk.get('source','doc')}_{i}".encode()).hexdigest()
            vectors.append({
                "id":     uid,
                "values": emb.tolist(),
                "metadata": {
                    "text":   chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "page":   chunk.get("page", 0),
                }
            })

        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i + batch_size])

        print(f"✅ Indexed {len(vectors)} chunks.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _semantic_search(self, query_emb: List[float]) -> List[Dict]:
        res = self.index.query(vector=query_emb, top_k=self.top_k, include_metadata=True)
        return res.get("matches", [])

    def _mmr_search(self, query_emb: List[float]) -> List[Dict]:
        """Max Marginal Relevance — relevance + diversity."""
        import numpy as np
        res = self.index.query(vector=query_emb, top_k=self.top_k * 3, include_metadata=True)
        candidates = res.get("matches", [])
        if not candidates:
            return []

        cand_embs = np.array([
            self.embed_model.encode(c["metadata"]["text"])
            for c in candidates
        ])
        q_emb = np.array(query_emb)

        selected, remaining = [], list(range(len(candidates)))
        for _ in range(min(self.top_k, len(candidates))):
            if not remaining:
                break
            if not selected:
                scores = cand_embs[remaining] @ q_emb
                best   = remaining[int(scores.argmax())]
            else:
                sel_embs   = cand_embs[selected]
                rel_scores = cand_embs[remaining] @ q_emb
                div_scores = (cand_embs[remaining] @ sel_embs.T).max(axis=1)
                mmr        = 0.6 * rel_scores - 0.4 * div_scores
                best       = remaining[int(mmr.argmax())]
            selected.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected]

    def _retrieve(self, query: str) -> List[Dict]:
        emb = self.embed_model.encode(query).tolist()
        if self.retrieval_strategy == "MMR (Diversity)":
            return self._mmr_search(emb)
        return self._semantic_search(emb)

    # ── Query rewriting ───────────────────────────────────────────────────────

    def _rewrite_query(self, query: str, chat_history: List[Dict]) -> str:
        if not chat_history:
            return query
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in chat_history[-4:]
        )
        prompt = f"""Given the conversation history and a follow-up question, rewrite the question to be standalone for document retrieval. Return ONLY the rewritten question, nothing else.

History:
{history_text}

Follow-up: {query}
Standalone question:"""
        resp = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()

    # ── Main query pipeline ───────────────────────────────────────────────────

    def query(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        chat_history = chat_history or []

        # 1. Rewrite query
        search_q = self._rewrite_query(query, chat_history)

        # 2. Retrieve chunks
        matches = self._retrieve(search_q)
        if not matches:
            return {
                "answer":  "I couldn't find relevant information in the uploaded documents.",
                "sources": [],
            }

        # 3. Build context + sources
        context_parts, sources = [], []
        for i, m in enumerate(matches):
            meta  = m.get("metadata", {})
            text  = meta.get("text", "")
            src   = meta.get("source", "unknown")
            page  = meta.get("page", "N/A")
            score = round(m.get("score", 0), 3)
            context_parts.append(f"[Chunk {i+1} | {src} | Page {page} | Score {score}]\n{text}")
            sources.append(f"📄 {src} — Page {page} (score: {score})")

        context = "\n\n---\n\n".join(context_parts)

        # 4. Build messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": f"""Context:
{context}

Question: {query}

Answer using ONLY the context above. Cite chunk numbers."""})

        # 5. Generate
        resp   = self.groq_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=2048,
        )
        answer = resp.choices[0].message.content

        return {"answer": answer, "sources": sources}