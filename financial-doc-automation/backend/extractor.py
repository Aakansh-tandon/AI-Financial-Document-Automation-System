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

SYSTEM_PROMPT = (
    "You are an expert financial document parser for invoices. "
    "Return raw JSON only. Never include markdown, comments, or extra text."
)

USER_PROMPT_TEMPLATE = """Extract fields from this financial document.

Rules:
1) Return valid JSON only.
2) Use null for unknown values.
3) Choose ONE final payable amount using this strict priority:
   Balance Due > Amount Due > Total Due > Grand Total > Total > Subtotal.
4) Vendor can be implicit. If no explicit seller label exists, infer the issuing company
   from the top section before "Bill To".
5) Due date should be YYYY-MM-DD when a clear date exists.

Return exactly:
{{
  "invoice_number": "<string or null>",
  "vendor": "<string or null>",
  "amount": "<string with currency symbol or null>",
  "due_date": "<ISO date string YYYY-MM-DD or null>"
}}

Document text:
{text}
"""

EXTRACTION_MODEL_CANDIDATES = (
    os.getenv("EXTRACTION_GEMINI_MODEL", "gemini-2.0-flash"),
    "gemini-1.5-flash-latest",
)

AMOUNT_PRIORITY = {
    "balance_due": 6,
    "amount_due": 5,
    "total_due": 4,
    "grand_total": 3,
    "total": 2,
    "subtotal": 1,
}

AMOUNT_LABEL_PATTERNS = (
    ("balance_due", r"\bbalance\s+due\b"),
    ("amount_due", r"\bamount\s+due\b"),
    ("total_due", r"\btotal\s+due\b"),
    ("grand_total", r"\bgrand\s+total\b"),
    ("total", r"\btotal\b(?!\s*(?:tax|vat|items?|qty|quantity)\b)"),
    ("subtotal", r"\bsub\s*total\b|\bsubtotal\b"),
)

MONEY_PATTERN = r"([$£€₹]?\s?\d[\d,\s]*(?:\.\s*\d{2})?)"


class StructuredExtractor:
    """Extracts structured financial data from document text using Gemini + rules."""

    def __init__(self):
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
    def _normalize_for_rules(text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"([$£€₹])\s+(?=\d)", r"\1", cleaned)
        cleaned = re.sub(r"(?<=\d)\s*,\s*(?=\d{3}\b)", ",", cleaned)
        cleaned = re.sub(r"(?<=\d)\s*\.\s*(?=\d{2}\b)", ".", cleaned)
        cleaned = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _prepare_text_for_llm(text: str, max_chars: int = 7000) -> str:
        """
        Keep deterministic + bounded context. Preserve header/footer by using head+tail.
        """
        clean = StructuredExtractor._normalize_for_rules(text)
        if len(clean) <= max_chars:
            return clean
        half = max_chars // 2
        return f"{clean[:half]}\n\n...[truncated middle]...\n\n{clean[-half:]}"

    @staticmethod
    def _parse_json_from_response(raw: str) -> dict | None:
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
        if value is None:
            return None
        candidate = str(value).strip()
        if not candidate:
            return None

        formats = (
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%b %d, %Y",
            "%B %d, %Y",
        )
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
        return candidate if re.match(r"^\d{4}-\d{2}-\d{2}$", candidate) else None

    @staticmethod
    def _normalize_amount(value: str | None) -> str | None:
        if value is None:
            return None
        amount = str(value).strip()
        if not amount:
            return None
        amount = re.sub(r"\s+", "", amount)
        amount = re.sub(r"(?<=\d),(?=\d{3}\b)", ",", amount)
        amount = re.sub(r"(?<=\d)\.(?=\d{2}\b)", ".", amount)
        amount = re.sub(r"(?<![$£€₹])(?=\d)", "", amount)
        return amount

    @staticmethod
    def _amount_to_float(value: str | None) -> float:
        if value is None:
            return 0.0
        cleaned = re.sub(r"[^0-9.]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _extract_invoice_number(self, text: str) -> str | None:
        match = re.search(
            r"(?im)\b(?:invoice\s*(?:number|no|#)?|inv[#\s-]*|reference)\s*[:#-]?\s*([A-Z0-9-]{3,})\b",
            text,
        )
        return match.group(1).strip() if match else None

    def _extract_due_date(self, text: str) -> str | None:
        match = re.search(
            r"(?im)\b(?:due\s*date|payment\s*due|pay\s*by)\b\s*[:\-]?\s*([A-Za-z0-9,/\- ]{6,30})",
            text,
        )
        if not match:
            return None
        return self._to_iso_date(match.group(1))

    def _extract_vendor(self, text: str) -> str | None:
        patterns = (
            r"(?im)^\s*(?:from|vendor|seller|billed\s+by|issued\s+by)\s*:\s*([^\n]{2,100})$",
            r"(?is)\b(?:from|vendor|seller|billed\s+by|issued\s+by)\b\s*:\s*\n?\s*([^\n,]{2,100})",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip(" -,:")
                if candidate:
                    return candidate

        # Positional heuristic: first meaningful entity before "Bill To"
        bill_to_match = re.search(r"(?im)\bbill\s*to\b", text)
        top_region = text[: bill_to_match.start()] if bill_to_match else text[:1500]
        lines = [ln.strip(" -,:") for ln in top_region.splitlines() if ln.strip()]

        for line in lines:
            low = line.lower()
            if any(token in low for token in ("invoice", "date", "due", "amount", "total", "ship to", "bill to")):
                continue
            if re.search(r"\b(?:inc|llc|ltd|limited|corp|corporation|company|co\.)\b", line, re.I):
                return line

        invoice_header_idx = next((i for i, ln in enumerate(lines) if re.search(r"\binvoice\b", ln, re.I)), None)
        if invoice_header_idx is not None:
            for line in lines[invoice_header_idx + 1 :]:
                low = line.lower()
                if any(token in low for token in ("bill to", "invoice #", "invoice number", "date", "due")):
                    continue
                if re.search(r"[A-Z][A-Za-z&.,'\- ]{2,}", line):
                    return line
        return None

    def _extract_amount_candidates(self, text: str) -> list[dict]:
        candidates: list[dict] = []
        normalized = self._normalize_for_rules(text)
        for label, label_pattern in AMOUNT_LABEL_PATTERNS:
            pattern = re.compile(
                rf"(?im){label_pattern}\s*[:\-]?\s*{MONEY_PATTERN}",
            )
            matches = list(pattern.finditer(normalized))
            if not matches:
                continue
            for match in matches:
                amount = self._normalize_amount(match.group(1))
                if amount:
                    candidates.append(
                        {
                            "label": label,
                            "amount": amount,
                            "priority": AMOUNT_PRIORITY[label],
                            "position": match.start(),
                        }
                    )
        return candidates

    def _choose_final_amount(self, text: str) -> tuple[str | None, str | None]:
        candidates = self._extract_amount_candidates(text)
        if candidates:
            # Highest priority first; for same label prefer the last occurrence.
            best = sorted(candidates, key=lambda c: (c["priority"], c["position"]), reverse=True)[0]
            return best["amount"], best["label"]

        # Fallback: choose max currency value when no labels are available.
        normalized = self._normalize_for_rules(text)
        values = re.findall(r"[$£€₹]\s?\d[\d,]*(?:\.\d{2})?", normalized)
        if values:
            amounts = [self._normalize_amount(v) for v in values]
            amounts = [a for a in amounts if a]
            if amounts:
                best_amount = max(amounts, key=self._amount_to_float)
                return best_amount, "max_currency_fallback"

        return None, None

    def _normalize_result(self, parsed: dict | None) -> dict:
        result = self._null_result()
        if not isinstance(parsed, dict):
            return result

        for key in ("invoice_number", "vendor", "amount", "due_date"):
            value = parsed.get(key)
            if value is not None:
                value = str(value).strip()
                result[key] = value if value else None

        result["amount"] = self._normalize_amount(result["amount"])
        result["due_date"] = self._to_iso_date(result["due_date"])
        return result

    def _regex_fallback_extract(self, text: str) -> tuple[dict, str | None]:
        normalized = self._normalize_for_rules(text)
        result = self._null_result()
        result["invoice_number"] = self._extract_invoice_number(normalized)
        result["vendor"] = self._extract_vendor(normalized)
        result["amount"], amount_label = self._choose_final_amount(normalized)
        result["due_date"] = self._extract_due_date(normalized)
        return result, amount_label

    def _merge_results(self, llm_result: dict, rule_result: dict, text: str, rule_amount_label: str | None) -> tuple[dict, str | None]:
        merged = self._null_result()
        for key in ("invoice_number", "vendor", "due_date"):
            merged[key] = llm_result.get(key) or rule_result.get(key)

        llm_amount = llm_result.get("amount")
        rule_amount = rule_result.get("amount")
        final_amount = llm_amount or rule_amount
        final_label = None

        # Critical guard: if a higher-priority deterministic amount exists, prefer it.
        if rule_amount:
            if not llm_amount:
                final_amount = rule_amount
                final_label = rule_amount_label
            else:
                llm_label = self._infer_amount_label_from_text(text, llm_amount)
                llm_priority = AMOUNT_PRIORITY.get(llm_label, 0)
                rule_priority = AMOUNT_PRIORITY.get(rule_amount_label, 0)
                if rule_priority >= llm_priority:
                    final_amount = rule_amount
                    final_label = rule_amount_label
                else:
                    final_label = llm_label
        else:
            final_label = self._infer_amount_label_from_text(text, final_amount)

        merged["amount"] = final_amount
        merged["confidence_score"] = self._compute_confidence(merged, text, final_label)
        return merged, final_label

    def _infer_amount_label_from_text(self, text: str, amount: str | None) -> str | None:
        if not amount:
            return None
        normalized = self._normalize_for_rules(text)
        escaped_amount = re.escape(amount).replace(r"\,", r",")
        for label, label_pattern in AMOUNT_LABEL_PATTERNS:
            if re.search(rf"(?im){label_pattern}\s*[:\-]?\s*{escaped_amount}", normalized):
                return label
        return None

    def _compute_confidence(self, result: dict, text: str, amount_label: str | None) -> float:
        populated = sum(
            1
            for field in ("invoice_number", "vendor", "amount", "due_date")
            if result.get(field) is not None
        )
        score = populated / 4

        # Penalize if higher-priority "Balance Due" exists but selected amount is lower priority.
        normalized = self._normalize_for_rules(text)
        has_balance_due = bool(re.search(r"(?im)\bbalance\s+due\b", normalized))
        if has_balance_due and amount_label and amount_label != "balance_due":
            score -= 0.20
        elif has_balance_due and result.get("amount") is None:
            score -= 0.25

        return round(max(0.0, min(1.0, score)), 2)

    def extract(self, text: str) -> dict:
        """
        Extract structured financial data from document text.
        """
        if not text or not text.strip():
            return self._null_result()

        llm_result = self._null_result()
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

            if last_error and llm_result == self._null_result():
                raise RuntimeError(str(last_error))
        except Exception as e:
            print(f"[Extractor Error] {str(e)}")

        rule_result, rule_amount_label = self._regex_fallback_extract(text)
        merged, _ = self._merge_results(llm_result, rule_result, text, rule_amount_label)
        return merged
