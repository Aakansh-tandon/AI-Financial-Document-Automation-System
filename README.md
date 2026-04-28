# AI Financial Document Automation

A full-stack system for processing financial PDFs with:

- **FastAPI backend** for ingestion, extraction, RAG Q&A, and history APIs
- **Streamlit frontend** for upload, structured extraction, alerts, and chat-like querying
- **FAISS + sentence-transformers** for semantic retrieval
- **Google Gemini** for structured extraction and answer generation

---

## 1. What this project does

This project automates common financial-document workflows:

1. Upload a PDF (invoice/statement/receipt)
2. Extract text from the PDF
3. Build semantic chunks + embeddings for retrieval
4. Extract structured fields:
   - `invoice_number`
   - `vendor`
   - `amount`
   - `due_date`
   - `confidence_score`
5. Apply automation rules (due alerts, high-value alerts, missing fields, low confidence)
6. Ask questions over the uploaded document via RAG
7. Persist extraction records in local JSON storage

---

## 2. Key features

- **Structured extraction with LLM + fallback logic**
  - Gemini extraction prompt with invoice-specific mapping rules
  - robust JSON parsing (handles fenced/wrapped JSON output)
  - regex fallback for critical fields when LLM misses data
- **RAG Q&A over uploaded document**
  - chunking + embedding (`all-MiniLM-L6-v2`)
  - FAISS cosine-like search (normalized vectors + inner product)
  - Gemini answer generation from retrieved context only
- **Automation rule engine**
  - overdue / urgent due-date alerts
  - high-value invoice detection
  - missing-field detection
  - low-confidence extraction alert
- **History tracking**
  - append-only record persistence to `data/extracted/results.json`

---

## 3. Architecture overview

### Backend modules (`backend/`)

- `main.py`  
  FastAPI app entrypoint and API routes orchestration.

- `document_processor.py`  
  PDF extraction + text normalization + chunking.

- `extractor.py`  
  Structured field extraction using Gemini with deterministic fallback parsing/rules.

- `automation_engine.py`  
  Rule-based alert/severity engine over extracted fields.

- `rag_engine.py`  
  FAISS indexing + semantic retrieval + Gemini answer generation.

- `storage.py`  
  Local JSON storage manager for extraction results.

### Frontend (`frontend/`)

- `app.py`  
  Streamlit UI for:
  - upload/process
  - extract/analyze
  - RAG Q&A
  - history browsing

### Data folder (`data/`)

- `data/extracted/results.json`  
  Persistent extraction history file.

---

## 4. Tech stack

- **Python** 3.10+
- **FastAPI** + **Uvicorn**
- **Streamlit**
- **PyMuPDF (fitz)** for PDF text extraction
- **sentence-transformers** (`all-MiniLM-L6-v2`)
- **FAISS (CPU)** for vector search
- **Google Gemini API** (`google-generativeai`)
- **python-dotenv**

---

## 5. Prerequisites

- Windows PowerShell (commands below use Windows paths)
- Python installed and available in PATH
- Gemini API key

---

## 6. Setup (recommended: project-local virtual environment)

From project root:

```powershell
cd "D:\Repositries\rag-doc-qa-endee\financial-doc-automation"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create/update `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Important:** Never commit real API keys. If a key was exposed, rotate it immediately.

---

## 7. Run the application

Open **two terminals**.

### Terminal 1 — Backend

```powershell
cd "D:\Repositries\rag-doc-qa-endee\financial-doc-automation"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Backend URLs:

- Health: `http://localhost:8000/health`
- API docs (Swagger): `http://localhost:8000/docs`

### Terminal 2 — Frontend

```powershell
cd "D:\Repositries\rag-doc-qa-endee\financial-doc-automation"
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py --server.port 8501
```

Frontend URL:

- `http://localhost:8501`

---

## 7.1 Deploy backend on Render

This repo includes `render.yaml` for one-click backend deployment.

1. Push this repository to GitHub.
2. In Render, create a **Blueprint** from the repo (it will detect `render.yaml`).
3. Set required env var:
   - `GEMINI_API_KEY`
4. Deploy.

The backend will run with:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Health check:

- `https://<your-render-backend>.onrender.com/health`

Then set Streamlit secret:

```toml
BACKEND_URL = "https://<your-render-backend>.onrender.com"
```

---

## 8. API reference

### `GET /health`

Checks backend availability.

**Response**
```json
{"status": "ok"}
```

### `POST /upload`

Uploads and processes a PDF.

- Validates extension
- extracts text
- chunks text
- builds FAISS index
- stores current in-memory document for extraction/query

**Form-data**
- `file`: PDF file

**Response (example)**
```json
{
  "message": "Document processed successfully",
  "chunks": 8,
  "text_preview": "..."
}
```

### `POST /extract`

Extracts structured fields from the currently uploaded document and runs automation rules.

**Response shape**
```json
{
  "extracted": {
    "invoice_number": "INV-1001",
    "vendor": "Acme Services",
    "amount": "$1,250.00",
    "due_date": "2026-05-10",
    "confidence_score": 1.0
  },
  "alerts": {
    "alerts": [
      "WARNING: Payment due in 5 days"
    ],
    "severity": "MEDIUM"
  }
}
```

### `POST /query`

Runs RAG question answering on the currently indexed document.

**Request**
```json
{"question": "What is the amount due?"}
```

**Response**
```json
{
  "answer": "...",
  "sources": ["chunk text 1", "chunk text 2", "chunk text 3"]
}
```

### `GET /history`

Returns stored extraction records from `results.json`.

---

## 9. Extraction logic details

`backend/extractor.py` uses a layered strategy:

1. **LLM extraction prompt** with field-mapping guidance:
   - invoice synonyms (`Invoice #`, `Inv No`, `Reference`)
   - amount label priorities (`Balance Due` > `Grand Total` > `Total` > `Subtotal`)
   - vendor inference from top section if not labeled
2. **Strict JSON response mode** + robust parser fallback
3. **Regex fallback extraction** if LLM output is incomplete
4. **Date normalization** to ISO format when possible
5. **Dynamic confidence scoring**
   - `confidence_score = populated_fields / 4`

---

## 10. RAG pipeline details

`backend/rag_engine.py`:

1. Receives chunks from processor
2. Encodes chunks via `SentenceTransformer("all-MiniLM-L6-v2")`
3. L2 normalizes embeddings
4. Stores vectors in `faiss.IndexFlatIP` (cosine-like behavior)
5. On query:
   - embeds question
   - retrieves top-K chunks
   - builds context prompt
   - asks Gemini and returns answer + source chunks

Model fallback strategy is included for Gemini model availability handling.

---

## 11. Automation rules

`backend/automation_engine.py` evaluates:

- **Due date**
  - overdue → `CRITICAL`
  - due in 0–3 days → `HIGH`
  - due in 4–7 days → `MEDIUM`
- **High value amount**
  - above threshold (default: `$10,000`) → `HIGH`
- **Missing fields**
  - per required field
- **Low confidence**
  - below `0.6` triggers manual review recommendation

Overall severity is computed using rule severity hierarchy.

---

## 12. Text extraction and chunking behavior

`backend/document_processor.py`:

- extracts page text via PyMuPDF
- normalizes noisy PDF text:
  - collapses repeated spaces
  - removes excessive blank lines
  - merges hard-wrapped lines
- chunks with overlap:
  - default `chunk_size=500`
  - default `overlap=50`
  - validates `chunk_size > overlap`
  - attempts whitespace-aware chunk boundaries

---

## 13. Configuration

Environment variables:

- `GEMINI_API_KEY` (required)
- `EXTRACTION_GEMINI_MODEL` (optional override)
- `RAG_GEMINI_MODEL` (optional override)

Defaults use Gemini 2.0 Flash with fallback to a latest 1.5 model variant.

---

## 14. Project structure

```text
financial-doc-automation/
├── .env
├── .venv/
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── document_processor.py
│   ├── extractor.py
│   ├── automation_engine.py
│   ├── rag_engine.py
│   └── storage.py
├── frontend/
│   └── app.py
└── data/
    └── extracted/
        └── results.json
```

---

## 15. Troubleshooting

### Backend not reachable

- Ensure backend terminal is running on port `8000`
- Check `http://localhost:8000/health`

### Streamlit cannot connect to backend

- Confirm `BACKEND_URL` in `frontend/app.py` is `http://localhost:8000`
- Ensure no firewall blocks localhost ports

### Extraction returns null fields

- Re-upload the document after backend restart
- Verify PDF contains selectable text (not scanned images)
- Check `.env` and API key validity

### Model not found errors

- Use model env overrides:
  - `EXTRACTION_GEMINI_MODEL=gemini-1.5-flash-latest`
  - `RAG_GEMINI_MODEL=gemini-1.5-flash-latest`
- Restart backend after env changes

### Low chunk count (e.g., 1 chunk)

- This usually means extracted text is short
- Verify `text_preview` and source PDF content length/quality

### No history visible

- Check `data/extracted/results.json` exists and is writable

---

## 16. Current limitations

- No OCR pipeline for image-only/scanned PDFs
- Single active document in memory per backend process
- Local JSON storage is not multi-user transactional storage
- Extraction schema is invoice-oriented (can be extended for statements/POs)

---

## 17. Suggested production hardening (next steps)

- Add OCR fallback (e.g., Tesseract/Azure/AWS Textract) for scanned PDFs
- Move storage from JSON file to database (PostgreSQL)
- Add authentication/authorization for API + UI
- Add request IDs and structured logging
- Add tests:
  - extractor unit tests with real invoice fixtures
  - API integration tests
  - regression tests for prompt/parsing edge cases
- Add document/session isolation for multi-user usage

---

## 18. License / usage

This repository currently has no explicit license file. Add one if you plan to distribute publicly.

