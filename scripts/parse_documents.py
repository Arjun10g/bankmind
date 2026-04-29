"""Parse all downloaded documents into uniform ParsedDoc JSON files.

For each file in data/raw/{compliance,credit}/ that has a paired .meta.json,
parse it and write to data/processed/{compliance,credit}/parsed/<doc_id>.json.

Idempotent: skips parsed files that already exist (delete to re-parse).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.shared.document_parser import parse_document  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def gather_documents() -> list[tuple[str, Path, dict]]:
    """Return [(module, raw_path, metadata_dict)] for every parseable doc."""
    out: list[tuple[str, Path, dict]] = []
    for module in ("compliance", "credit"):
        raw_module_dir = RAW_DIR / module
        if not raw_module_dir.exists():
            continue
        for meta_file in sorted(raw_module_dir.glob("*.meta.json")):
            doc_id = meta_file.stem.replace(".meta", "")
            metadata = json.loads(meta_file.read_text())
            # The raw file shares the doc_id stem; figure out its extension.
            candidates = list(raw_module_dir.glob(f"{doc_id}.*"))
            candidates = [c for c in candidates if not c.name.endswith(".meta.json")]
            if not candidates:
                print(f"  ! no raw file found for {doc_id}", file=sys.stderr)
                continue
            raw_path = candidates[0]
            out.append((module, raw_path, metadata))
    return out


def main() -> int:
    docs = gather_documents()
    print(f"Found {len(docs)} documents to parse")

    n_ok = 0
    n_skipped = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for module, raw_path, metadata in tqdm(docs, desc="parsing"):
        out_dir = PROCESSED_DIR / module / "parsed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{metadata['doc_id']}.json"

        if out_path.exists() and out_path.stat().st_size > 0:
            n_skipped += 1
            continue

        try:
            doc = parse_document(raw_path, metadata, module)
            out_path.write_text(json.dumps(doc.to_dict(), indent=None))
            n_ok += 1
        except Exception as e:
            n_failed += 1
            failures.append((metadata["doc_id"], f"{type(e).__name__}: {e}"))
            traceback.print_exc(file=sys.stderr)

    summary = {
        "n_total": len(docs),
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "failures": [{"doc_id": d, "error": e} for d, e in failures],
    }
    summary_path = PROCESSED_DIR / "_parse_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Parse summary ===")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skipped}  (already parsed)")
    print(f"  failed:  {n_failed}")
    if failures:
        print("\n--- failures ---")
        for d, e in failures:
            print(f"  {d}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
