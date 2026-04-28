"""
FastAPI Main Application
Orchestrates all modules: document processing, extraction, RAG Q&A,
automation alerts, and storage.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.document_processor import DocumentProcessor
from backend.extractor import StructuredExtractor
from backend.rag_engine import RAGEngine
from backend.automation_engine import AutomationEngine
from backend.storage import StorageManager

# ── Initialize FastAPI app ──────────────────────────────────────────
app = FastAPI(
    title="AI Financial Document Automation",
    description="Automated financial document processing with LLM extraction and RAG Q&A",
    version="1.0.0",
)

# ── CORS middleware (allow all origins for Streamlit) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize all modules ─────────────────────────────────────────
processor = DocumentProcessor()
extractor = StructuredExtractor()
rag = RAGEngine()
automation = AutomationEngine()
storage = StorageManager()

# ── Module-level state ─────────────────────────────────────────────
current_text: str = ""
current_filename: str = ""


# ── Request models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing.
    Extracts text, chunks it, and builds the FAISS index for RAG.
    """
    global current_text, current_filename

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        pdf_bytes = await file.read()

        # Extract text from PDF
        text = processor.extract_text(pdf_bytes)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the PDF. The file may be image-based or empty.",
            )

        # Chunk text and build FAISS index
        chunks = processor.chunk_text(text)
        rag.build_index(chunks)

        # Store for later extraction
        current_text = text
        current_filename = file.filename

        return {
            "message": "Document processed successfully",
            "chunks": len(chunks),
            "text_preview": text[:200],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/extract")
async def extract_data():
    """
    Extract structured financial data from the current document.
    Runs LLM extraction, automation rules, and saves the result.
    """
    global current_text, current_filename

    if not current_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a PDF first.",
        )

    try:
        # Extract structured data via LLM
        extracted = extractor.extract(current_text)

        # Evaluate automation rules
        alerts = automation.evaluate(extracted)

        # Save to storage
        storage.save(
            filename=current_filename or "unknown.pdf",
            extracted=extracted,
            alerts=alerts,
        )

        return {
            "extracted": extracted,
            "alerts": alerts,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during extraction: {str(e)}")


@app.post("/query")
async def query_document(request: QueryRequest):
    """
    Ask a question about the uploaded document using RAG.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag.query(request.question)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/history")
async def get_history():
    """
    Retrieve all past extraction records.
    """
    try:
        records = storage.get_all()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")
