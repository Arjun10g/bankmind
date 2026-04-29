"""Run all six chunkers (3 compliance + 3 credit) on every parsed document.

Output:
  data/processed/compliance/chunks_regulatory_boundary.jsonl
  data/processed/compliance/chunks_semantic.jsonl
  data/processed/compliance/chunks_hierarchical.jsonl
  data/processed/credit/chunks_financial_statement.jsonl
  data/processed/credit/chunks_semantic.jsonl
  data/processed/credit/chunks_narrative_section.jsonl
  data/processed/_chunking_summary.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.compliance.chunker import CHUNKERS as COMPLIANCE_CHUNKERS  # noqa: E402
from pipelines.credit.chunker import CHUNKERS as CREDIT_CHUNKERS          # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"


def load_parsed_docs(module: str) -> list[dict]:
    parsed_dir = PROCESSED_DIR / module / "parsed"
    return [json.loads(p.read_text()) for p in sorted(parsed_dir.glob("*.json"))]


def run_module(module: str, chunkers: dict) -> dict:
    docs = load_parsed_docs(module)
    print(f"\n=== {module}: {len(docs)} parsed docs ===")

    summary: dict = {"module": module, "n_docs": len(docs), "strategies": {}}

    for strategy, fn in chunkers.items():
        out_path = PROCESSED_DIR / module / f"chunks_{strategy}.jsonl"
        all_chunks = []
        per_doc_counts = []
        token_counts: list[int] = []
        section_type_counts: Counter = Counter()
        contains_table_count = 0
        char_offset_errors = 0

        t0 = time.perf_counter()
        for doc in tqdm(docs, desc=f"  {strategy}", leave=False):
            try:
                chunks = fn(doc)
            except Exception as e:
                print(f"\n  ! {strategy} failed on {doc['doc_id']}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue

            per_doc_counts.append(len(chunks))
            full_text = doc["full_text"]
            for c in chunks:
                # Sanity: every chunk's content should appear (modulo whitespace) at its
                # claimed offsets. We don't enforce strictly because semantic chunker
                # joins sentences with single spaces — the reconstructed text may differ
                # from the slice. But char offsets must be within bounds.
                if c.char_start < 0 or c.char_end > len(full_text) or c.char_start > c.char_end:
                    char_offset_errors += 1
                token_counts.append(c.content_tokens)
                if c.contains_table:
                    contains_table_count += 1
                if c.section_type:
                    section_type_counts[c.section_type] += 1
                all_chunks.append(c)

        elapsed = time.perf_counter() - t0

        with out_path.open("w") as f:
            for c in all_chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

        summary["strategies"][strategy] = {
            "output_file": str(out_path.relative_to(ROOT)),
            "n_chunks": len(all_chunks),
            "n_chunks_per_doc_mean": round(sum(per_doc_counts) / max(len(per_doc_counts), 1), 1),
            "n_chunks_per_doc_min": min(per_doc_counts) if per_doc_counts else 0,
            "n_chunks_per_doc_max": max(per_doc_counts) if per_doc_counts else 0,
            "tokens_min": min(token_counts) if token_counts else 0,
            "tokens_max": max(token_counts) if token_counts else 0,
            "tokens_mean": round(sum(token_counts) / max(len(token_counts), 1), 1),
            "n_with_table": contains_table_count,
            "section_type_distribution": dict(section_type_counts.most_common()),
            "char_offset_errors": char_offset_errors,
            "elapsed_seconds": round(elapsed, 2),
        }
        print(f"  {strategy:25s}  {len(all_chunks):>6d} chunks  "
              f"avg {summary['strategies'][strategy]['tokens_mean']} tok  "
              f"{round(elapsed, 1)}s")

    return summary


def main() -> int:
    summaries = {
        "compliance": run_module("compliance", COMPLIANCE_CHUNKERS),
        "credit":     run_module("credit",     CREDIT_CHUNKERS),
    }

    summary_path = PROCESSED_DIR / "_chunking_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote summary → {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
