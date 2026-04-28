"""
Automation Engine Module
Rule-based alert logic for financial document processing.
Evaluates extracted data and generates alerts with severity levels.
"""

import re
from datetime import datetime, date


class AutomationEngine:
    """Evaluates extracted financial data and generates rule-based alerts."""

    HIGH_VALUE_THRESHOLD = 10000  # USD

    def evaluate(self, extracted_data: dict) -> dict:
        """
        Evaluate extracted financial data and produce alerts.

        Rules applied:
          1. Due date proximity check
          2. High value invoice check
          3. Missing fields check
          4. Low confidence check

        Args:
            extracted_data: Dictionary with keys invoice_number, vendor,
                           amount, due_date, confidence_score.

        Returns:
            Dictionary with 'alerts' (list of strings) and 'severity'
            (one of LOW, MEDIUM, HIGH, CRITICAL).
        """
        alerts: list[str] = []
        severity_levels: list[str] = []

        # ── Rule 1: Due date check ──────────────────────────────────
        due_date_str = extracted_data.get("due_date")
        if due_date_str is not None:
            try:
                due_date = datetime.strptime(str(due_date_str), "%Y-%m-%d").date()
                today = date.today()
                days_until_due = (due_date - today).days

                if days_until_due < 0:
                    alerts.append(
                        f"OVERDUE: Payment was due {abs(days_until_due)} days ago"
                    )
                    severity_levels.append("CRITICAL")
                elif 0 <= days_until_due <= 3:
                    alerts.append(
                        f"URGENT: Payment due in {days_until_due} days"
                    )
                    severity_levels.append("HIGH")
                elif 4 <= days_until_due <= 7:
                    alerts.append(
                        f"WARNING: Payment due in {days_until_due} days"
                    )
                    severity_levels.append("MEDIUM")
                else:
                    alerts.append(
                        f"INFO: Payment due in {days_until_due} days"
                    )
                    severity_levels.append("LOW")
            except (ValueError, TypeError):
                # Cannot parse due_date — skip this rule
                pass

        # ── Rule 2: High value check ────────────────────────────────
        amount_str = extracted_data.get("amount")
        if amount_str is not None:
            try:
                # Strip currency symbols and commas
                cleaned = re.sub(r"[$ £€₹,\s]", "", str(amount_str))
                amount_value = float(cleaned)
                if amount_value > self.HIGH_VALUE_THRESHOLD:
                    alerts.append(
                        f"HIGH VALUE: Invoice amount {amount_str} exceeds "
                        f"threshold of ${self.HIGH_VALUE_THRESHOLD:,.2f}"
                    )
                    severity_levels.append("HIGH")
            except (ValueError, TypeError):
                # Cannot parse amount — skip this rule
                pass

        # ── Rule 3: Missing fields check ────────────────────────────
        required_fields = ["invoice_number", "vendor", "amount", "due_date"]
        for field in required_fields:
            value = extracted_data.get(field)
            if value is None:
                alerts.append(f"MISSING FIELD: {field} not found in document")

        # ── Rule 4: Low confidence check ────────────────────────────
        confidence = extracted_data.get("confidence_score")
        if confidence is not None:
            try:
                confidence = float(confidence)
                if confidence < 0.6:
                    alerts.append(
                        f"LOW CONFIDENCE: Extraction confidence is {confidence:.2f}. "
                        "Manual review recommended."
                    )
            except (ValueError, TypeError):
                pass

        # ── Determine overall severity ──────────────────────────────
        hierarchy = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        if severity_levels:
            max_severity = max(severity_levels, key=lambda s: hierarchy.get(s, 0))
        else:
            max_severity = "LOW"

        return {
            "alerts": alerts,
            "severity": max_severity,
        }
