"""Dual-track QA generator.

Same questions feed both tracks (per CLAUDE.md § 7.1):

  Track A — retrieval evaluation
    Question + passage_id + (char_start, char_end) + source_doc_id + key_concepts.
    No reference answer needed; relevance scored by char-overlap of retrieved
    chunks with the source passage range. CHUNKING-AGNOSTIC.

  Track B — answer-quality evaluation (cross-evaluation)
    Same question + reference_answer (Claude's answer reading ONLY the raw
    passage, before any retrieval). The RAG answer is later scored against
    this reference via semantic-sim, BERTScore-F1, and concept-coverage.
    INDEPENDENT OF RETRIEVAL — measures generation quality given perfect context.

Output schema (per CLAUDE.md § 7.1, both tracks interleaved by qa_id):

    {
      "qa_id": "uuid",
      "track": "A" | "B",
      "module": "compliance" | "credit",
      "question": "...",                    # IDENTICAL across A and B for the same qa_id
      "question_type": "factual" | "interpretive" | "comparative",
      "difficulty": "easy" | "medium" | "hard",
      "key_concepts": ["term1", "term2"],
      "source_passage_id": "...",
      "source_passage_text": "...",
      # Track A only:
      "char_start": int, "char_end": int, "source_doc_id": "...",
      # Track B only:
      "reference_answer": "...",
      "reference_answer_model": "claude-sonnet-4-6",
    }
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from pipelines.shared.llm import claude_json, claude_text


_TRACK_A_PROMPT = """You are generating evaluation questions for a financial RAG system.

Given the passage below, generate {n} distinct questions such that:
1. Each question is answerable ONLY using information explicitly stated in this passage.
2. Questions do NOT lift verbatim phrases from the passage — use natural paraphrasing a {role} would use.
3. Distribute across complexity: at least one factual (specific facts/numbers/clauses), one interpretive (meaning, scope, application).
4. A correct answer REQUIRES information from this passage (not retrievable from general knowledge).

Passage:
\"\"\"
{passage_text}
\"\"\"

Return ONLY a JSON array, exactly {n} items:
[
  {{
    "question": "...",
    "question_type": "factual" | "interpretive" | "comparative",
    "difficulty": "easy" | "medium" | "hard",
    "key_concepts": ["term1", "term2", "term3"]
  }},
  ...
]

`key_concepts` are 2-5 terms that MUST appear in any correct answer. Choose terms that are central to the passage's content."""


_TRACK_B_REFERENCE_PROMPT = """You are a senior {role} at a major bank. You have been given a passage from a {doc_type} document and a question about it.

Your task: write the best possible answer to the question using ONLY the information in this passage. Do NOT use outside knowledge. If the passage does not fully answer the question, state what it does cover and explicitly note what's missing.

Be specific and precise. Cite figures, dates, or clause references that appear in the passage when relevant. Write in the professional voice of a {role}.

Passage:
\"\"\"
{passage_text}
\"\"\"

Question: {question}

Write your answer now. No preamble — just the answer, 2-5 sentences:"""


_ROLE = {"compliance": "compliance officer", "credit": "credit analyst"}


@dataclass
class QAPair:
    qa_id: str
    track: str                            # "A" or "B"
    module: str
    question: str
    question_type: str
    difficulty: str
    key_concepts: list[str]
    source_passage_id: str
    source_passage_text: str
    # Track A
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    source_doc_id: Optional[str] = None
    # Track B
    reference_answer: Optional[str] = None
    reference_answer_model: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None or k in {"char_start", "char_end"}}


def _qa_id_for(passage_id: str, idx: int) -> str:
    """Stable UUIDv5 so reruns produce the same IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{passage_id}|q{idx}"))


def generate_questions_for_passage(
    passage: dict,
    *,
    n_questions: int,
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """Track A: generate `n_questions` questions for one passage."""
    role = _ROLE[passage["module"]]
    raw = claude_json(
        _TRACK_A_PROMPT.format(
            n=n_questions, role=role, passage_text=passage["passage_text"],
        ),
        system=f"You are {role} writing eval questions.",
        model=model,
        max_tokens=800,
    )
    if isinstance(raw, dict) and "questions" in raw:
        items = raw["questions"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    out: list[dict] = []
    for q in items[:n_questions]:
        if not isinstance(q, dict):
            continue
        out.append({
            "question": str(q.get("question", "")).strip(),
            "question_type": str(q.get("question_type", "factual")).lower().strip(),
            "difficulty": str(q.get("difficulty", "medium")).lower().strip(),
            "key_concepts": [str(k).strip() for k in q.get("key_concepts", []) if str(k).strip()],
        })
    return [q for q in out if q["question"]]


def generate_reference_answer(
    passage: dict,
    question: str,
    *,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Track B: have Claude write the gold-standard answer from the raw passage only."""
    role = _ROLE[passage["module"]]
    return claude_text(
        _TRACK_B_REFERENCE_PROMPT.format(
            role=role,
            doc_type=passage["source_doc_type"],
            passage_text=passage["passage_text"],
            question=question,
        ),
        max_tokens=400,
        model=model,
    )


def generate_qa_pairs(
    passages: list[dict],
    *,
    n_per_passage: int = 2,
    model: str = "claude-sonnet-4-6",
) -> list[QAPair]:
    """Top-level: produce both tracks for every passage."""
    out: list[QAPair] = []
    for passage in tqdm(passages, desc=f"  qa-gen ({passages[0]['module']})"):
        try:
            questions = generate_questions_for_passage(
                passage, n_questions=n_per_passage, model=model,
            )
        except Exception as e:
            print(f"  ! question gen failed for {passage['passage_id']}: {e}")
            continue

        for idx, q in enumerate(questions):
            qa_id = _qa_id_for(passage["passage_id"], idx)
            base = dict(
                qa_id=qa_id,
                module=passage["module"],
                question=q["question"],
                question_type=q["question_type"],
                difficulty=q["difficulty"],
                key_concepts=q["key_concepts"],
                source_passage_id=passage["passage_id"],
                source_passage_text=passage["passage_text"],
            )
            # Track A
            out.append(QAPair(
                track="A",
                char_start=passage["char_start"],
                char_end=passage["char_end"],
                source_doc_id=passage["source_doc_id"],
                **base,
            ))
            # Track B — reference answer
            try:
                ref = generate_reference_answer(passage, q["question"], model=model)
            except Exception as e:
                print(f"  ! ref-ans gen failed for {qa_id}: {e}")
                ref = ""
            out.append(QAPair(
                track="B",
                reference_answer=ref.strip(),
                reference_answer_model=model,
                **base,
            ))
    return out
