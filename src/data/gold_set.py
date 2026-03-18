"""Gold test set loading and validation."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"question", "gold_answer", "domain", "difficulty", "is_answerable"}
VALID_DOMAINS = {"criminal", "civil", "commercial", "administrative", "constitutional"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_gold_set(path: Path) -> list[dict[str, Any]]:
    """Load gold test set from JSON.

    Args:
        path: Path to JSON file.

    Returns:
        List of gold test examples.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]

    validate_gold_set(data)
    logger.info(f"Loaded gold set: {len(data)} questions from {path}")
    return data


def validate_gold_set(data: list[dict[str, Any]]) -> None:
    """Validate gold test set schema.

    Args:
        data: List of gold test examples.

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(data, list):
        raise ValueError(f"Gold set must be a list, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Gold set is empty")

    errors = []
    for i, item in enumerate(data):
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            errors.append(f"Question {i}: missing fields {missing}")

        domain = item.get("domain", "").lower()
        if domain and domain not in VALID_DOMAINS:
            errors.append(f"Question {i}: invalid domain '{domain}', must be one of {VALID_DOMAINS}")

        difficulty = item.get("difficulty", "").lower()
        if difficulty and difficulty not in VALID_DIFFICULTIES:
            errors.append(f"Question {i}: invalid difficulty '{difficulty}', must be one of {VALID_DIFFICULTIES}")

        if not isinstance(item.get("is_answerable"), bool):
            errors.append(f"Question {i}: 'is_answerable' must be boolean")

    if errors:
        error_msg = "\n".join(errors[:10])
        raise ValueError(f"Gold set validation failed:\n{error_msg}")

    logger.info(f"Gold set validated: {len(data)} questions OK")


def gold_set_stats(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics for a gold test set.

    Args:
        data: List of gold test examples.

    Returns:
        Dict with counts by domain, difficulty, answerability.
    """
    from collections import Counter

    domains = Counter(item.get("domain", "unknown").lower() for item in data)
    difficulties = Counter(item.get("difficulty", "unknown").lower() for item in data)
    answerable = sum(1 for item in data if item.get("is_answerable", True))

    return {
        "total": len(data),
        "by_domain": dict(domains),
        "by_difficulty": dict(difficulties),
        "answerable": answerable,
        "unanswerable": len(data) - answerable,
    }
