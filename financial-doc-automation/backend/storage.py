"""
Storage Module
Handles reading and writing extraction results to a local JSON file.
Append-only storage — never overwrites previous records.
"""

import json
import os
from datetime import datetime


class StorageManager:
    """Manages persistent storage of extraction results in a JSON file."""

    STORAGE_PATH = os.path.join("data", "extracted", "results.json")

    def __init__(self):
        """Initialize the storage manager. Creates directories and file if needed."""
        storage_dir = os.path.dirname(self.STORAGE_PATH)
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)

        self.data: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load existing data from the JSON file."""
        try:
            if os.path.exists(self.STORAGE_PATH):
                with open(self.STORAGE_PATH, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.data = json.loads(content)
                    else:
                        self.data = []
            else:
                self.data = []
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Storage Warning] Could not load existing data: {e}")
            self.data = []

    def save(self, filename: str, extracted: dict, alerts: dict) -> None:
        """
        Save an extraction result record (append-only).

        Args:
            filename: Name of the processed PDF file.
            extracted: Dictionary of extracted financial fields.
            alerts: Dictionary of alerts and severity.
        """
        record = {
            "filename": filename,
            "extracted": extracted,
            "alerts": alerts,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.data.append(record)

        try:
            with open(self.STORAGE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=str)
        except IOError as e:
            print(f"[Storage Error] Failed to write data: {e}")

    def get_all(self) -> list[dict]:
        """
        Get all stored extraction records.

        Returns:
            List of all record dictionaries.
        """
        self._load()
        return self.data

    def get_latest(self) -> dict | None:
        """
        Get the most recent extraction record.

        Returns:
            The latest record dictionary, or None if no records exist.
        """
        self._load()
        if self.data:
            return self.data[-1]
        return None
