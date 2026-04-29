"""Extract source passages from parsed docs for both modules.

Output:
  data/eval/source_passages/compliance_passages.json    (25 passages)
  data/eval/source_passages/credit_passages.json        (25 passages)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.passage_extractor import extract_passages

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "eval" / "source_passages"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PASSAGES_PER_MODULE = 25


def main() -> int:
    summary: dict = {}
    for module in ("compliance", "credit"):
        parsed_dir = PROCESSED / module / "parsed"
        docs = [json.loads(p.read_text()) for p in sorted(parsed_dir.glob("*.json"))]
        # Strip oversized full_text from logs but pass through to extractor
        print(f"\n[{module}] {len(docs)} parsed docs")

        passages = extract_passages(docs, n_passages=N_PASSAGES_PER_MODULE)
        print(f"  → extracted {len(passages)} passages")

        # Distribution check
        by_doc: dict[str, int] = {}
        by_doc_type: dict[str, int] = {}
        for p in passages:
            by_doc[p.source_doc_id] = by_doc.get(p.source_doc_id, 0) + 1
            by_doc_type[p.source_doc_type] = by_doc_type.get(p.source_doc_type, 0) + 1
        print(f"  by doc_type: {dict(sorted(by_doc_type.items(), key=lambda x: -x[1]))}")
        print(f"  unique source docs: {len(by_doc)} (max per doc: {max(by_doc.values()) if by_doc else 0})")

        # Save
        out_path = OUT_DIR / f"{module}_passages.json"
        out_path.write_text(json.dumps([p.to_dict() for p in passages], indent=2))
        print(f"  → {out_path.relative_to(ROOT)}")

        summary[module] = {
            "n_passages": len(passages),
            "by_doc_type": by_doc_type,
            "n_unique_docs": len(by_doc),
        }

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
