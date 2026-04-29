"""Driver: read source passages → generate dual-track QA pairs → save.

Output:
  data/eval/compliance_qa.json   (50 QA pairs: 25 Track A + 25 Track B if 1 q/passage,
                                   or 100 if 2 q/passage)
  data/eval/credit_qa.json
  data/eval/_qa_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.qa_generator import generate_qa_pairs

EVAL_DIR = ROOT / "data" / "eval"

N_PER_PASSAGE = 2


def main() -> int:
    summary: dict = {}
    for module in ("compliance", "credit"):
        passages_path = EVAL_DIR / "source_passages" / f"{module}_passages.json"
        if not passages_path.exists():
            print(f"  ! missing {passages_path}; run scripts/extract_source_passages.py first")
            return 1
        passages = json.loads(passages_path.read_text())
        print(f"\n[{module}] {len(passages)} passages → generating {N_PER_PASSAGE}× questions each")

        out_path = EVAL_DIR / f"{module}_qa.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            print(f"  ⏭  {out_path.relative_to(ROOT)} already exists with {len(existing)} entries; skipping")
            summary[module] = {"n_qa_pairs": len(existing), "skipped": True}
            continue

        pairs = generate_qa_pairs(passages, n_per_passage=N_PER_PASSAGE)
        out_path.write_text(json.dumps([p.to_dict() for p in pairs], indent=2))

        n_track_a = sum(1 for p in pairs if p.track == "A")
        n_track_b = sum(1 for p in pairs if p.track == "B")
        n_with_ref = sum(1 for p in pairs if p.track == "B" and (p.reference_answer or "").strip())
        # Question type / difficulty distributions (from track A only)
        qtypes: dict[str, int] = {}
        diffs: dict[str, int] = {}
        for p in pairs:
            if p.track == "A":
                qtypes[p.question_type] = qtypes.get(p.question_type, 0) + 1
                diffs[p.difficulty] = diffs.get(p.difficulty, 0) + 1
        print(f"  → {out_path.relative_to(ROOT)}")
        print(f"     {n_track_a} Track-A + {n_track_b} Track-B (with refs: {n_with_ref})")
        print(f"     question_types: {qtypes}")
        print(f"     difficulties:   {diffs}")
        summary[module] = {
            "n_qa_pairs": len(pairs),
            "n_track_a": n_track_a,
            "n_track_b": n_track_b,
            "n_with_reference_answer": n_with_ref,
            "question_types": qtypes,
            "difficulties": diffs,
        }

    (EVAL_DIR / "_qa_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
