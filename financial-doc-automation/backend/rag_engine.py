"""
RAG Engine Module
FAISS-based vector store with lightweight hashed embeddings and Gemini LLM
for retrieval-augmented generation Q&A over financial documents.
"""

import os
import time
import hashlib
import re

import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a financial document. "
    "Answer ONLY based on the context below. If the answer is not in the context, "
    'say "I could not find this information in the document."'
)

RAG_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

RAG_MODEL_CANDIDATES = (
    os.getenv("RAG_GEMINI_MODEL", "gemini-2.0-flash"),
    "gemini-1.5-flash-latest",
)
RAG_MAX_RETRIES_PER_MODEL = 2


class RAGEngine:
    """Retrieval-Augmented Generation engine using FAISS and Gemini."""

    def __init__(self):
        """Initialize embedding pipeline, FAISS index placeholder, and Gemini model."""
        self.embedding_dim = 512
        self.index = None
        self.chunks: list[str] = []

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.llm_models = [
            genai.GenerativeModel(model_name=model_name, system_instruction=RAG_SYSTEM_PROMPT)
            for model_name in RAG_MODEL_CANDIDATES
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Lightweight deterministic hashing embeddings to keep memory low in free hosting.
        """
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in self._tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], byteorder="little") % self.embedding_dim
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                vectors[i, index] += sign
        return vectors

    def build_index(self, chunks: list[str]) -> None:
        """
        Build a FAISS index from document chunks.

        Uses inner product search with L2 normalization to achieve
        cosine similarity behavior.

        Args:
            chunks: List of text chunks to index.
        """
        if not chunks:
            self.index = None
            self.chunks = []
            return

        self.chunks = chunks

        # Encode chunks into embeddings
        embeddings = self._encode_texts(chunks)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Create FAISS index with inner product (cosine after normalization)
        dimension = self.embedding_dim
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a question using RAG over the indexed document chunks.

        Args:
            question: The user's question.
            top_k: Number of top matching chunks to retrieve (default 5).

        Returns:
            Dictionary with 'answer' and 'sources' (top 3 chunks).
        """
        if self.index is None or not self.chunks:
            return {"answer": "No document loaded.", "sources": []}

        try:
            # Embed and normalize the question
            q_embedding = self._encode_texts([question])
            faiss.normalize_L2(q_embedding)

            # Search the FAISS index
            safe_top_k = max(1, min(top_k, len(self.chunks)))
            distances, indices = self.index.search(q_embedding, safe_top_k)

            # Retrieve matching chunks
            matched_chunks = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    matched_chunks.append(self.chunks[idx])

            # Build the RAG prompt
            context = "\n\n---\n\n".join(matched_chunks)
            user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
                context=context, question=question
            )

            # Call Gemini LLM
            answer = None
            last_error = None
            for model in self.llm_models:
                for attempt in range(RAG_MAX_RETRIES_PER_MODEL):
                    try:
                        response = model.generate_content(
                            user_prompt,
                            generation_config={"temperature": 0},
                        )
                        answer = (response.text or "").strip()
                        if answer:
                            break
                        last_error = RuntimeError("Model returned empty response.")
                    except Exception as e:
                        last_error = e
                        # Retry same model for transient errors, then fallback.
                        if "not found" in str(e).lower():
                            break
                        if attempt < RAG_MAX_RETRIES_PER_MODEL - 1:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                        break
                if answer:
                    break

            if not answer:
                if last_error:
                    raise RuntimeError(str(last_error))
                raise RuntimeError("RAG model returned an empty response.")

            return {
                "answer": answer,
                "sources": matched_chunks[:3],  # Return top 3 source chunks
            }

        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
            }
