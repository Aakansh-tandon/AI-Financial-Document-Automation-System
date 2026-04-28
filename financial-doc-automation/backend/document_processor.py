"""
Document Processor Module
Handles PDF text extraction and text chunking for downstream processing.
"""

import re

import fitz  # PyMuPDF


class DocumentProcessor:
    """Processes PDF documents by extracting text and splitting into chunks."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize PDF text while preserving important key/value cues.
        """
        if not text:
            return ""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse repeated spaces/tabs.
        normalized = re.sub(r"[ \t]+", " ", normalized)
        # Keep paragraph boundaries, but avoid noisy vertical spacing.
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        # Merge hard-wrapped lines while preserving blank-line paragraph splits.
        normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
        return normalized.strip()

    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF file provided as bytes.

        Args:
            pdf_bytes: Raw bytes of the PDF file.

        Returns:
            Full extracted text with pages separated by double newlines.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages_text.append(self.normalize_text(text))
            doc.close()
            return self.normalize_text("\n\n".join(pages_text))
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: The full document text.
            chunk_size: Maximum number of characters per chunk (default 500).
            overlap: Number of overlapping characters between chunks (default 50).

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap.")

        normalized_text = self.normalize_text(text)
        if not normalized_text:
            return []

        chunks = []
        start = 0
        text_length = len(normalized_text)
        step = chunk_size - overlap

        while start < text_length:
            end = min(start + chunk_size, text_length)
            # Prefer cutting on whitespace to avoid slicing tokens mid-word.
            if end < text_length:
                next_space = normalized_text.find(" ", end)
                if 0 <= next_space <= end + 80:
                    end = next_space
            chunk = normalized_text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += step

        return chunks
