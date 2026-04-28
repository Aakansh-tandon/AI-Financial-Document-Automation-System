"""
Streamlit Frontend
UI for the AI Financial Document Automation System.
Communicates with the FastAPI backend via HTTP requests.
"""

import json

import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI Financial Document Automation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium styling ─────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    div[data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">📄 AI Financial Document Automation</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload financial documents • Extract structured data • Ask questions with RAG</p>', unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────
def check_backend():
    """Check if the backend is running."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ── Sidebar: Document Upload ──────────────────────────────────────
with st.sidebar:
    st.markdown("### 📤 Upload Document")
    st.divider()

    # Backend status indicator
    if check_backend():
        st.success("✅ Backend connected")
    else:
        st.error("❌ Backend not running. Start it with:\n`uvicorn backend.main:app --reload --port 8000`")

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a financial document (invoice, receipt, statement) in PDF format.",
    )

    if uploaded_file is not None:
        st.info(f"📎 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("🚀 Upload & Process", use_container_width=True, type="primary"):
        if uploaded_file is None:
            st.warning("Please select a PDF file first.")
        else:
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["uploaded"] = True
                        st.session_state["filename"] = uploaded_file.name
                        st.session_state["chunks"] = data["chunks"]
                        st.session_state["text_preview"] = data["text_preview"]

                        st.success(f"✅ Document processed! **{data['chunks']} chunks** created.")
                        with st.expander("📝 Text Preview"):
                            st.text(data["text_preview"])
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Upload failed: {error_detail}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Show upload status
    if st.session_state.get("uploaded"):
        st.divider()
        st.markdown("#### 📊 Current Document")
        st.markdown(f"**File:** {st.session_state.get('filename', 'N/A')}")
        st.markdown(f"**Chunks:** {st.session_state.get('chunks', 0)}")


# ── Main Area: Two Columns ─────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")


# ── LEFT COLUMN: Extract Structured Data ───────────────────────────
with col_left:
    st.markdown('<div class="section-header">🔍 Extract Structured Data</div>', unsafe_allow_html=True)

    if st.button("⚡ Extract & Analyze", use_container_width=True, type="primary"):
        if not st.session_state.get("uploaded"):
            st.warning("⚠️ Please upload a document first.")
        else:
            with st.spinner("Extracting financial data with AI..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/extract", timeout=120)

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["extraction_result"] = result

                        extracted = result["extracted"]
                        alerts = result["alerts"]

                        # ── Extracted Data ──
                        st.markdown("#### 📋 Extracted Fields")
                        st.json(extracted)

                        # ── Confidence Score ──
                        confidence = extracted.get("confidence_score", 0.0)
                        if confidence is not None:
                            try:
                                confidence = float(confidence)
                            except (ValueError, TypeError):
                                confidence = 0.0
                        else:
                            confidence = 0.0

                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence:.0%}",
                            delta="High" if confidence >= 0.8 else ("Medium" if confidence >= 0.6 else "Low"),
                        )

                        # ── Alerts ──
                        st.markdown("#### 🚨 Alerts")
                        severity = alerts.get("severity", "LOW")
                        alert_list = alerts.get("alerts", [])

                        if not alert_list:
                            st.info("✅ No alerts. Document looks clean.")
                        else:
                            # Severity badge
                            severity_colors = {
                                "CRITICAL": "🔴",
                                "HIGH": "🟠",
                                "MEDIUM": "🟡",
                                "LOW": "🟢",
                            }
                            st.markdown(f"**Overall Severity:** {severity_colors.get(severity, '⚪')} **{severity}**")

                            for alert_msg in alert_list:
                                if any(kw in alert_msg for kw in ["OVERDUE", "CRITICAL"]):
                                    st.error(f"🔴 {alert_msg}")
                                elif any(kw in alert_msg for kw in ["URGENT", "HIGH VALUE"]):
                                    st.error(f"🟠 {alert_msg}")
                                elif any(kw in alert_msg for kw in ["WARNING"]):
                                    st.warning(f"🟡 {alert_msg}")
                                else:
                                    st.info(f"🔵 {alert_msg}")
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Extraction failed: {error_detail}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Show previous extraction result if available
    if "extraction_result" in st.session_state and not st.session_state.get("_just_extracted"):
        result = st.session_state["extraction_result"]
        with st.expander("📄 Previous Extraction Result", expanded=False):
            st.json(result)


# ── RIGHT COLUMN: RAG Q&A ─────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-header">💬 Ask a Question (RAG)</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Enter your question about the document:",
        placeholder="e.g., What is the total amount due?",
        key="rag_question",
    )

    if st.button("🧠 Get Answer", use_container_width=True, type="primary"):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        elif not st.session_state.get("uploaded"):
            st.warning("⚠️ Please upload a document first.")
        else:
            with st.spinner("Searching document and generating answer..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"question": question},
                        timeout=120,
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.success(f"**Answer:** {result['answer']}")

                        sources = result.get("sources", [])
                        if sources:
                            with st.expander(f"📚 Source Chunks ({len(sources)})", expanded=False):
                                for i, chunk in enumerate(sources, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text(chunk)
                                    if i < len(sources):
                                        st.divider()
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Query failed: {error_detail}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ── Bottom Section: Extraction History ─────────────────────────────
st.divider()
st.markdown('<div class="section-header">📜 Extraction History</div>', unsafe_allow_html=True)

if st.button("📂 Load History", use_container_width=True):
    with st.spinner("Loading history..."):
        try:
            response = requests.get(f"{BACKEND_URL}/history", timeout=30)

            if response.status_code == 200:
                records = response.json()

                if not records:
                    st.info("No extraction history found. Process a document to get started.")
                else:
                    st.markdown(f"**{len(records)} record(s) found**")
                    for i, record in enumerate(reversed(records)):
                        filename = record.get("filename", "Unknown")
                        timestamp = record.get("timestamp", "N/A")
                        with st.expander(f"📄 {filename} — {timestamp}", expanded=False):
                            st.markdown("**Extracted Data:**")
                            st.json(record.get("extracted", {}))
                            st.markdown("**Alerts:**")
                            alerts = record.get("alerts", {})
                            st.json(alerts)
            else:
                st.error("Failed to load history.")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is it running?")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align: center; color: #9ca3af; font-size: 0.85rem;">'
    "AI Financial Document Automation System • Powered by Gemini 2.0 Flash + FAISS + Sentence-Transformers"
    "</p>",
    unsafe_allow_html=True,
)
