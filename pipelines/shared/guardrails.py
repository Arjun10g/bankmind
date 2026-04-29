"""Guardrails — runtime safety checks that wrap every RAG response.

All rule-based, no LLM calls. Run after retrieval + (optional) generation
to flag the highest-risk failure modes per CLAUDE.md § 9.

Public surface:
  - check_compliance(answer, chunks, query) -> GuardrailReport
  - check_credit(answer, chunks, query)     -> GuardrailReport

Both return the same shape so the UI can render them uniformly. Flags are
non-blocking — the user still sees the answer, just with warnings attached.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from .retriever import ScoredChunk


# === Shared shapes ==========================================================

@dataclass
class GuardrailWarning:
    severity: str         # "info" | "warning" | "high"
    code: str             # short machine-readable code
    message: str          # human-readable explanation


@dataclass
class GuardrailReport:
    confidence: float                    # [0, 1] — derived from top-1 retrieval score
    confidence_label: str                # "low" | "medium" | "high"
    warnings: list[GuardrailWarning] = field(default_factory=list)
    citation_coverage: Optional[float] = None       # % of answer sentences supported by a chunk
    unsupported_sentences: list[str] = field(default_factory=list)
    grounded_numbers: int = 0
    ungrounded_numbers: list[str] = field(default_factory=list)

    def has_high_severity(self) -> bool:
        return any(w.severity == "high" for w in self.warnings)

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 3),
            "confidence_label": self.confidence_label,
            "citation_coverage": (
                round(self.citation_coverage, 3) if self.citation_coverage is not None else None
            ),
            "unsupported_sentences": self.unsupported_sentences,
            "grounded_numbers": self.grounded_numbers,
            "ungrounded_numbers": self.ungrounded_numbers,
            "warnings": [
                {"severity": w.severity, "code": w.code, "message": w.message}
                for w in self.warnings
            ],
        }


# === Confidence score =======================================================

def _confidence_from_chunks(chunks: list[ScoredChunk]) -> tuple[float, str]:
    """Top-1 retrieval score → [0,1] confidence. Calibrated for hybrid+RRF
    where scores typically land in [0.3, 1.5]; clip + rescale.

    For raw BM25 (sparse), scores can be 5-25; we normalize differently.
    Heuristic but consistent.
    """
    if not chunks:
        return 0.0, "low"
    top = chunks[0].score
    if top <= 0:
        conf = 0.0
    elif top <= 1.5:
        conf = max(0.0, min(1.0, top / 1.5))                # hybrid/RRF range
    else:
        # BM25-style: anything ≥ 15 is highly confident
        conf = max(0.0, min(1.0, top / 15.0))
    if conf < 0.4:
        label = "low"
    elif conf < 0.7:
        label = "medium"
    else:
        label = "high"
    return conf, label


# === Citation enforcement ==================================================

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "as", "by", "from", "at", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "may", "might",
    "must", "it", "its", "their", "they", "them", "any", "such", "not",
    "if", "then", "than", "so", "no", "all", "some", "more", "most", "less",
}


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"\b[a-z][a-z0-9-]+\b", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _sentence_supported(sentence: str, chunks: list[ScoredChunk], *, min_overlap: int = 3) -> bool:
    """A sentence is 'supported' if ≥`min_overlap` content words appear in any
    retrieved chunk. Heuristic, but catches the obvious "from thin air" cases."""
    s_words = _content_words(sentence)
    if len(s_words) < min_overlap:
        # Sentence too short to evaluate; treat as supported (no false alarms)
        return True
    for c in chunks:
        c_words = _content_words(c.content)
        if len(s_words & c_words) >= min_overlap:
            return True
    return False


def _enforce_citations(answer: str, chunks: list[ScoredChunk]) -> tuple[float, list[str]]:
    """Return (coverage_ratio, unsupported_sentences)."""
    if not answer or not chunks:
        return 0.0, []
    sentences = [s.strip() for s in _SENT_END.split(answer) if s.strip()]
    if not sentences:
        return 1.0, []
    unsupported = [s for s in sentences if not _sentence_supported(s, chunks)]
    return 1.0 - len(unsupported) / len(sentences), unsupported


# === Number grounding (credit-critical) ====================================

# Match $ amounts, percentages, multi-digit numbers, fiscal periods.
_NUMBER_PATTERNS = re.compile(
    r"|".join([
        r"\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|m|bn|k))?",
        r"\d+(?:\.\d+)?\s*%",
        r"\b\d{4}\b",                        # fiscal years
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",  # large grouped numbers
        r"\b\d+\.\d+\b",                     # decimals
    ]),
    re.IGNORECASE,
)


def _normalize_number_phrase(s: str) -> str:
    """Normalize for forgiving substring match: strip whitespace, lowercase,
    drop $ signs and commas."""
    s = s.lower().strip()
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    s = re.sub(r"million|billion|trillion|bn|m\b|k\b", "", s)
    return s


def _ground_numbers(answer: str, chunks: list[ScoredChunk]) -> tuple[int, list[str]]:
    """For each number in the answer, check whether it appears (modulo
    formatting) in any retrieved chunk's content. Returns (grounded_count,
    list_of_ungrounded_phrases)."""
    if not answer:
        return 0, []
    raw_numbers = _NUMBER_PATTERNS.findall(answer)
    if not raw_numbers:
        return 0, []
    # Fast: build a single normalized string per chunk
    normalized_corpus = " ".join(_normalize_number_phrase(c.content) for c in chunks)
    grounded = 0
    ungrounded: list[str] = []
    for n in raw_numbers:
        norm = _normalize_number_phrase(n)
        if not norm or norm in {"", "0"}:
            continue
        if norm in normalized_corpus:
            grounded += 1
        else:
            ungrounded.append(n.strip())
    return grounded, ungrounded


# === Version / temporal warnings ===========================================

_TEMPORAL_QUERY_PATTERNS = re.compile(
    r"\b(current|currently|today|this year|now|latest|recent|present|"
    r"upcoming|next year|future)\b",
    re.IGNORECASE,
)


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _version_warnings(chunks: list[ScoredChunk], *, today: Optional[date] = None) -> list[GuardrailWarning]:
    """If any chunk's effective_date or filing_date is older than 2 years,
    emit a warning. Helps users notice they're reading stale regulation."""
    today = today or date.today()
    out: list[GuardrailWarning] = []
    seen_warnings: set[str] = set()
    for c in chunks[:5]:                         # only check top-5
        for field_name in ("effective_date", "filing_date"):
            d = _parse_date(c.payload.get(field_name))
            if d is None:
                continue
            years = (today - d).days / 365.25
            if years > 2:
                doc = c.payload.get("doc_id", "?")
                key = f"old:{doc}:{field_name}"
                if key in seen_warnings:
                    continue
                seen_warnings.add(key)
                out.append(GuardrailWarning(
                    severity="warning",
                    code="stale_source",
                    message=(
                        f"Top-retrieved chunk from `{doc}` has {field_name}={d.isoformat()} "
                        f"({years:.1f} years old). More recent guidance/filings may exist."
                    ),
                ))
    return out


def _temporal_query_warning(query: str, chunks: list[ScoredChunk]) -> Optional[GuardrailWarning]:
    """If the query asks about the current state but every retrieved chunk is
    historical (≥2 yrs), warn."""
    if not _TEMPORAL_QUERY_PATTERNS.search(query):
        return None
    today = date.today()
    historical_count = 0
    for c in chunks[:5]:
        for field_name in ("effective_date", "filing_date"):
            d = _parse_date(c.payload.get(field_name))
            if d is not None and (today - d).days / 365.25 > 2:
                historical_count += 1
                break
    if historical_count >= 3:                   # majority of top-5 stale
        return GuardrailWarning(
            severity="warning",
            code="temporal_mismatch",
            message=(
                "Query implies current/recent state, but ≥3 of top-5 retrieved chunks "
                "are 2+ years old. Answer may not reflect the current regulatory or "
                "financial state."
            ),
        )
    return None


# === Per-module entry points ===============================================

def check_compliance(
    answer: Optional[str],
    chunks: list[ScoredChunk],
    query: str,
) -> GuardrailReport:
    """Compliance guardrails: citation enforcement + version + temporal."""
    confidence, conf_label = _confidence_from_chunks(chunks)
    warnings: list[GuardrailWarning] = []

    citation_coverage = None
    unsupported = []
    if answer:
        citation_coverage, unsupported = _enforce_citations(answer, chunks)
        if citation_coverage < 0.7 and unsupported:
            warnings.append(GuardrailWarning(
                severity="high",
                code="low_citation_coverage",
                message=(
                    f"{len(unsupported)} of {len(unsupported) + int(citation_coverage * 10)} "
                    f"sentences in the answer don't have a clearly supporting "
                    f"chunk. Possible hallucination."
                ),
            ))

    if conf_label == "low":
        warnings.append(GuardrailWarning(
            severity="warning",
            code="low_retrieval_confidence",
            message=f"Top retrieval score = {chunks[0].score:.3f} if chunks else 'no results'. Answer quality may be unreliable.",
        ))

    warnings.extend(_version_warnings(chunks))
    tw = _temporal_query_warning(query, chunks)
    if tw:
        warnings.append(tw)

    return GuardrailReport(
        confidence=confidence,
        confidence_label=conf_label,
        warnings=warnings,
        citation_coverage=citation_coverage,
        unsupported_sentences=unsupported,
        grounded_numbers=0,
        ungrounded_numbers=[],
    )


def check_credit(
    answer: Optional[str],
    chunks: list[ScoredChunk],
    query: str,
) -> GuardrailReport:
    """Credit guardrails: citation enforcement + NUMBER GROUNDING + version + temporal.

    Number grounding is the highest-priority check for credit — hallucinated
    financial figures are the worst failure mode for this module.
    """
    confidence, conf_label = _confidence_from_chunks(chunks)
    warnings: list[GuardrailWarning] = []

    citation_coverage = None
    unsupported: list[str] = []
    grounded = 0
    ungrounded: list[str] = []

    if answer:
        citation_coverage, unsupported = _enforce_citations(answer, chunks)
        grounded, ungrounded = _ground_numbers(answer, chunks)

        if citation_coverage < 0.7 and unsupported:
            warnings.append(GuardrailWarning(
                severity="high",
                code="low_citation_coverage",
                message=(
                    f"{len(unsupported)} answer sentences not supported by any "
                    f"retrieved chunk. Possible hallucination."
                ),
            ))
        if ungrounded:
            warnings.append(GuardrailWarning(
                severity="high",
                code="ungrounded_numbers",
                message=(
                    f"The answer cites {len(ungrounded)} numeric value(s) that don't "
                    f"appear in the retrieved passages: {', '.join(ungrounded[:5])}"
                    f"{' …' if len(ungrounded) > 5 else ''}. "
                    f"Hallucinated financial figures are the highest-risk failure mode."
                ),
            ))

    if conf_label == "low":
        warnings.append(GuardrailWarning(
            severity="warning",
            code="low_retrieval_confidence",
            message=f"Top retrieval score = {chunks[0].score:.3f if chunks else 0:.3f}. Answer quality may be unreliable.",
        ))

    warnings.extend(_version_warnings(chunks))
    tw = _temporal_query_warning(query, chunks)
    if tw:
        warnings.append(tw)

    return GuardrailReport(
        confidence=confidence,
        confidence_label=conf_label,
        warnings=warnings,
        citation_coverage=citation_coverage,
        unsupported_sentences=unsupported,
        grounded_numbers=grounded,
        ungrounded_numbers=ungrounded,
    )


def check(module: str, answer, chunks, query) -> GuardrailReport:
    """Module dispatcher."""
    if module == "compliance":
        return check_compliance(answer, chunks, query)
    return check_credit(answer, chunks, query)
