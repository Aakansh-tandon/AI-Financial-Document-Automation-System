"""
RAG Engine Module
FAISS-based vector store with sentence-transformers embeddings and Gemini LLM
for retrieval-augmented generation Q&A over financial documents.
"""

import os

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
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


class RAGEngine:
    """Retrieval-Augmented Generation engine using FAISS and Gemini."""

    def __init__(self):
        """Initialize the embedding model, FAISS index placeholder, and Gemini model."""
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
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
        embeddings = self.embed_model.encode(chunks)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Create FAISS index with inner product (cosine after normalization)
        dimension = 384  # all-MiniLM-L6-v2 output dimension
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
            q_embedding = self.embed_model.encode([question])
            q_embedding = np.array(q_embedding, dtype=np.float32)
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
                try:
                    response = model.generate_content(
                        user_prompt,
                        generation_config={"temperature": 0.1},
                    )
                    answer = (response.text or "").strip()
                    if answer:
                        break
                except Exception as e:
                    last_error = e
                    # Try next model if current one is unavailable.
                    if "not found" in str(e).lower():
                        continue
                    raise

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
