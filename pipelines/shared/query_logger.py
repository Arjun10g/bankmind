"""Query logger — append-only JSONL of every query that flows through the system.

One line per call to `app.query_pipeline.run_query`. Captures everything
needed to reproduce a query (config), attribute its cost (timings), debug
its output (top chunks + answer), and audit its safety (guardrail report).

Output: logs/query_log.jsonl   (gitignored)

Phase 10 minimum-viable observability per CLAUDE.md.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "query_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


_LOCK = threading.Lock()


def log_query(
    *,
    query: str,
    config: dict,
    timings: dict,
    transformed_queries: list[str],
    top_chunks: list[dict],
    answer: str | None,
    guardrail_report: dict | None,
) -> str:
    """Append a single record. Returns the assigned query_id."""
    record = {
        "query_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "config": config,
        "transformed_queries": transformed_queries,
        "timings": timings,
        "top_chunks": top_chunks,                   # pre-shaped for storage
        "answer": answer,
        "guardrail_report": guardrail_report,
    }
    line = json.dumps(record, default=str) + "\n"
    with _LOCK:
        with LOG_PATH.open("a") as f:
            f.write(line)
    return record["query_id"]


def chunk_for_log(chunk) -> dict:
    """Compact chunk representation for the log: payload trimmed + score."""
    payload = chunk.payload or {}
    return {
        "chunk_id": chunk.chunk_id,
        "score": float(chunk.score),
        "doc_id": payload.get("doc_id"),
        "section_title": payload.get("section_title"),
        "section_number": payload.get("section_number"),
        "char_start": payload.get("char_start"),
        "char_end": payload.get("char_end"),
        "content_preview": (payload.get("content") or "")[:300],
    }


def read_log(*, limit: int = 50) -> list[dict]:
    """Tail the log — useful for the UI's history view if/when added."""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().splitlines()
    out: list[dict] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
