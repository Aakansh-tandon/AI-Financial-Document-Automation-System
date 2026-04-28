"""
Structured Extractor Module
Uses Google Gemini to extract structured financial data from document text.
"""

import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# SAMPLE TEST INVOICE (for testing/reference)
# ═══════════════════════════════════════════════════════════════════
#
# INVOICE
# -------
# Invoice Number: INV-2024-00742
# Date: 2024-08-15
#
# From:
#   Acme Cloud Services Ltd.
#   123 Tech Park Drive
#   San Francisco, CA 94105
#
# Bill To:
#   Global Enterprises Inc.
#   456 Corporate Blvd
#   New York, NY 10001
#
# Description                    Qty    Unit Price     Total
# ─────────────────────────────────────────────────────────────
# Cloud Hosting (Annual)          1      $8,500.00    $8,500.00
# Premium Support Package         1      $2,400.00    $2,400.00
# Data Migration Service          1      $1,200.00    $1,200.00
# ─────────────────────────────────────────────────────────────
# Subtotal:                                          $12,100.00
# Tax (8.5%):                                         $1,028.50
# ─────────────────────────────────────────────────────────────
# TOTAL DUE:                                         $13,128.50
#
# Payment Terms: Net 30
# Due Date: 2024-09-14
#
# Please remit payment to:
#   Bank: First National Bank
#   Account: 9876543210
#   Routing: 021000021
#
# Thank you for your business!
# ═══════════════════════════════════════════════════════════════════


SYSTEM_PROMPT = (
    "You are an expert financial document parser. "
    "Extract structured invoice fields with high precision. "
    "Return valid JSON only. No markdown. No commentary."
)

USER_PROMPT_TEMPLATE = """Extract the following fields from this financial document.
If a field is not found, use null for that field.

Field mapping rules:
- invoice_number: look for "Invoice Number", "Invoice #", "Inv No", "Reference".
- vendor: prefer explicit seller labels ("From", "Vendor", "Seller", "Billed By").
  If not labeled, infer vendor from the top section (typically first business/entity name).
- amount: pick the FINAL payable amount with this priority:
  1) Balance Due / Amount Due / Total Due
  2) Grand Total
  3) Total
  4) Subtotal (only if no higher-priority value exists)
- due_date: look for "Due Date", "Payment Due", "Pay By", and return ISO YYYY-MM-DD if possible.

Return ONLY this JSON structure, no other text:
{{
  "invoice_number": "<string or null>",
  "vendor": "<string or null>",
  "amount": "<string with currency symbol or null>",
  "due_date": "<ISO date string YYYY-MM-DD or null>"
}}

Document text:
{text}"""

EXTRACTION_MODEL_CANDIDATES = (
    os.getenv("EXTRACTION_GEMINI_MODEL", "gemini-2.0-flash"),
    "gemini-1.5-flash-latest",
)


class StructuredExtractor:
    """Extracts structured financial data from document text using Gemini LLM."""

    def __init__(self):
        """Initialize the Gemini model with API key from environment."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.models = [
            genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)
            for model_name in EXTRACTION_MODEL_CANDIDATES
        ]

    @staticmethod
    def _null_result() -> dict:
        return {
            "invoice_number": None,
            "vendor": None,
            "amount": None,
            "due_date": None,
            "confidence_score": 0.0,
        }

    @staticmethod
    def _prepare_text_for_llm(text: str, max_chars: int = 8000) -> str:
        """Keep head+tail so footer fields (e.g., due date/total) are not dropped."""
        clean = re.sub(r"\r\n?", "\n", text).strip()
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        clean = re.sub(r"(?<!\n)\n(?!\n)", " ", clean)
        if len(clean) <= max_chars:
            return clean
        half = max_chars // 2
        return f"{clean[:half]}\n\n...[truncated middle]...\n\n{clean[-half:]}"

    @staticmethod
    def _parse_json_from_response(raw: str) -> dict | None:
        """Parse model output as JSON, including responses wrapped in extra text."""
        if not raw:
            return None
        cleaned = raw.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _to_iso_date(value: str | None) -> str | None:
        """Normalize common date formats to YYYY-MM-DD."""
        if value is None:
            return None
        candidate = str(value).strip()
        if not candidate:
            return None

        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%b %d, %Y",
            "%B %d, %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
        return candidate if re.match(r"^\d{4}-\d{2}-\d{2}$", candidate) else None

    def _regex_fallback_extract(self, text: str) -> dict:
        """
        Deterministic fallback extraction when LLM output is invalid/unavailable.
        """
        result = self._null_result()

        normalized_text = self._prepare_text_for_llm(text, max_chars=12000)

        invoice_match = re.search(
            r"(?im)\b(?:invoice\s*(?:number|no|#)?|inv[#\s-]*|reference)\s*[:#-]?\s*([A-Z0-9-]{3,})\b",
            normalized_text,
        )
        if invoice_match:
            result["invoice_number"] = invoice_match.group(1).strip()

        vendor_match = re.search(
            r"(?is)\b(?:from|vendor|seller|billed\s+by)\b\s*:\s*\n?\s*([^\n,]{2,80})",
            normalized_text,
        )
        if not vendor_match:
            vendor_match = re.search(
                r"(?im)^\s*(?:from|vendor|seller|billed\s+by)\s*:\s*([^\n]{2,80})\s*$",
                normalized_text,
            )
        if not vendor_match:
            top_window = normalized_text[:1500]
            for line in [ln.strip() for ln in re.split(r"\n+", top_window) if ln.strip()]:
                low = line.lower()
                if any(
                    token in low for token in
                    ["invoice", "bill to", "due date", "amount due", "total", "po ", "ship to"]
                ):
                    continue
                if re.search(r"[A-Za-z]{2,}", line):
                    vendor_match = re.search(r"(.+)", line)
                    break
        if vendor_match:
            result["vendor"] = vendor_match.group(1).strip()

        amount_patterns = [
            r"(?im)\b(?:balance\s+due|amount\s+due|total\s+due)\b\s*[:\-]?\s*([$£€₹]\s?\d[\d,]*(?:\.\d{2})?)",
            r"(?im)\b(?:grand\s+total)\b\s*[:\-]?\s*([$£€₹]\s?\d[\d,]*(?:\.\d{2})?)",
            r"(?im)\b(?:total)\b\s*[:\-]?\s*([$£€₹]\s?\d[\d,]*(?:\.\d{2})?)",
            r"(?im)\b(?:subtotal)\b\s*[:\-]?\s*([$£€₹]\s?\d[\d,]*(?:\.\d{2})?)",
            r"(?im)\b(?:balance\s+due|amount\s+due|total\s+due)\b\s*[:\-]?\s*(\d[\d,]*(?:\.\d{2})?)",
            r"(?im)\b(?:grand\s+total)\b\s*[:\-]?\s*(\d[\d,]*(?:\.\d{2})?)",
            r"(?im)\b(?:total)\b\s*[:\-]?\s*(\d[\d,]*(?:\.\d{2})?)",
            r"(?im)\b(?:subtotal)\b\s*[:\-]?\s*(\d[\d,]*(?:\.\d{2})?)",
        ]
        for pattern in amount_patterns:
            amount_match = re.search(pattern, normalized_text)
            if amount_match:
                result["amount"] = amount_match.group(1).strip()
                break

        if result["amount"] is None:
            # Fallback: choose the largest currency value if no label-driven amount found.
            currency_values = re.findall(r"[$£€₹]\s?\d[\d,]*(?:\.\d{2})?", normalized_text)
            if currency_values:
                def _to_float(value: str) -> float:
                    cleaned = re.sub(r"[^0-9.]", "", value)
                    try:
                        return float(cleaned)
                    except ValueError:
                        return 0.0

                result["amount"] = max(currency_values, key=_to_float).strip()

        due_match = re.search(
            r"(?im)\b(?:due\s*date|payment\s*due)\b\s*[:\-]?\s*([A-Za-z0-9,/\- ]{6,30})",
            normalized_text,
        )
        if due_match:
            result["due_date"] = self._to_iso_date(due_match.group(1))
        result["confidence_score"] = self._compute_dynamic_confidence(result)
        return result

    def _normalize_result(self, parsed: dict | None) -> dict:
        """Normalize model payload to expected schema and types."""
        result = self._null_result()
        if not isinstance(parsed, dict):
            return result

        for key in ("invoice_number", "vendor", "amount", "due_date"):
            value = parsed.get(key)
            if value is not None:
                value = str(value).strip()
                result[key] = value if value else None

        result["due_date"] = self._to_iso_date(result["due_date"])

        confidence = parsed.get("confidence_score")
        if confidence is not None:
            try:
                confidence = float(confidence)
                result["confidence_score"] = min(max(confidence, 0.0), 1.0)
            except (ValueError, TypeError):
                pass

        return result

    def _merge_results(self, llm_result: dict, regex_result: dict) -> dict:
        merged = self._null_result()
        for key in ("invoice_number", "vendor", "amount", "due_date"):
            merged[key] = llm_result.get(key) or regex_result.get(key)
        merged["confidence_score"] = self._compute_dynamic_confidence(merged)
        return merged

    @staticmethod
    def _compute_dynamic_confidence(result: dict) -> float:
        populated = sum(
            1
            for field in ("invoice_number", "vendor", "amount", "due_date")
            if result.get(field) is not None
        )
        return round(populated / 4, 2)

    def extract(self, text: str) -> dict:
        """
        Extract structured financial data from document text.

        Args:
            text: The full document text to process.

        Returns:
            Dictionary with extracted fields: invoice_number, vendor,
            amount, due_date, confidence_score.
        """
        null_result = self._null_result()

        if not text or not text.strip():
            return null_result

        llm_result = null_result
        try:
            prepared_text = self._prepare_text_for_llm(text)
            user_prompt = USER_PROMPT_TEMPLATE.format(text=prepared_text)
            last_error = None
            for model in self.models:
                try:
                    response = model.generate_content(
                        user_prompt,
                        generation_config={
                            "temperature": 0,
                            "response_mime_type": "application/json",
                        },
                    )
                    parsed = self._parse_json_from_response(response.text if response.text else "")
                    llm_result = self._normalize_result(parsed)
                    if any(llm_result.get(k) for k in ("invoice_number", "vendor", "amount", "due_date")):
                        break
                except Exception as model_error:
                    last_error = model_error
                    if "not found" in str(model_error).lower():
                        continue
                    raise
            if last_error and llm_result == null_result:
                raise RuntimeError(str(last_error))
        except Exception as e:
            print(f"[Extractor Error] {str(e)}")

        regex_result = self._regex_fallback_extract(text)
        merged = self._merge_results(llm_result, regex_result)
        return merged
