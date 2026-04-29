"""Download regulatory & compliance source documents.

Sources are pinned to known-stable URLs (validated 2026-04-26). Each entry
ships with metadata that downstream parsing/chunking will need (doc_type,
regulatory_body, jurisdiction, etc.) so the parser doesn't have to guess.

Idempotent: skips files already present on disk.

Output:
  data/raw/compliance/<doc_id>.<ext>          # raw downloaded file
  data/raw/compliance/<doc_id>.meta.json      # metadata sidecar
  data/raw/compliance/_manifest.json          # download log (success/failure per doc)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "compliance"
RAW_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "BankMind Research arjunghumman1995@gmail.com"
TIMEOUT = httpx.Timeout(60.0, connect=20.0)


@dataclass
class Source:
    doc_id: str
    doc_title: str
    doc_type: str           # 'osfi' | 'fintrac' | 'basel' | 'bank_act' | 'gdpr' | 'fed'
    regulatory_body: str
    jurisdiction: str       # 'canada' | 'eu' | 'usa' | 'international'
    url: str
    file_ext: str           # 'pdf' | 'html'
    effective_date: str | None = None     # YYYY-MM-DD if known
    doc_version: str | None = None
    notes: str = ""


SOURCES: list[Source] = [
    # === OSFI (Canadian banking regulator) ===
    Source(
        doc_id="osfi_b20",
        doc_title="Residential Mortgage Underwriting Practices and Procedures (B-20)",
        doc_type="osfi",
        regulatory_body="OSFI",
        jurisdiction="canada",
        url="https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/residential-mortgage-underwriting-practices-procedures",
        file_ext="html",
    ),
    Source(
        doc_id="osfi_e23",
        doc_title="Enterprise-Wide Model Risk Management (E-23)",
        doc_type="osfi",
        regulatory_body="OSFI",
        jurisdiction="canada",
        url="https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/enterprise-wide-model-risk-management-deposit-taking-institutions",
        file_ext="html",
    ),
    Source(
        doc_id="osfi_b10",
        doc_title="Third Party Risk Management Guideline (B-10)",
        doc_type="osfi",
        regulatory_body="OSFI",
        jurisdiction="canada",
        url="https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/third-party-risk-management-guideline",
        file_ext="html",
    ),
    Source(
        doc_id="osfi_integrity_security",
        doc_title="Integrity and Security Guideline",
        doc_type="osfi",
        regulatory_body="OSFI",
        jurisdiction="canada",
        url="https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/integrity-security-guideline",
        file_ext="html",
    ),

    # === FINTRAC (Canadian AML regulator) ===
    Source(
        doc_id="fintrac_guide11_client_id",
        doc_title="FINTRAC Guide 11 — Methods to verify the identity of persons and entities",
        doc_type="fintrac",
        regulatory_body="FINTRAC",
        jurisdiction="canada",
        url="https://www.fintrac-canafe.gc.ca/guidance-directives/client-clientele/Guide11/11-eng",
        file_ext="html",
    ),

    # === Basel Committee on Banking Supervision (BCBS / BIS) ===
    Source(
        doc_id="basel_iii_framework_2011",
        doc_title="Basel III: A global regulatory framework for more resilient banks and banking systems (2011, revised)",
        doc_type="basel",
        regulatory_body="BCBS",
        jurisdiction="international",
        url="https://www.bis.org/publ/bcbs189.pdf",
        file_ext="pdf",
        effective_date="2011-06-01",
        doc_version="bcbs189",
    ),
    Source(
        doc_id="basel_iii_finalising_2017",
        doc_title="Basel III: Finalising post-crisis reforms (December 2017)",
        doc_type="basel",
        regulatory_body="BCBS",
        jurisdiction="international",
        url="https://www.bis.org/bcbs/publ/d424.pdf",
        file_ext="pdf",
        effective_date="2017-12-01",
        doc_version="d424",
    ),
    Source(
        doc_id="basel_d440",
        doc_title="BCBS Publication d440",
        doc_type="basel",
        regulatory_body="BCBS",
        jurisdiction="international",
        url="https://www.bis.org/bcbs/publ/d440.pdf",
        file_ext="pdf",
        doc_version="d440",
    ),
    Source(
        doc_id="basel_d457",
        doc_title="BCBS Publication d457",
        doc_type="basel",
        regulatory_body="BCBS",
        jurisdiction="international",
        url="https://www.bis.org/bcbs/publ/d457.pdf",
        file_ext="pdf",
        doc_version="d457",
    ),
    Source(
        doc_id="basel_d544",
        doc_title="BCBS Publication d544",
        doc_type="basel",
        regulatory_body="BCBS",
        jurisdiction="international",
        url="https://www.bis.org/bcbs/publ/d544.pdf",
        file_ext="pdf",
        doc_version="d544",
    ),

    # === Bank Act of Canada ===
    Source(
        doc_id="bank_act_canada",
        doc_title="Bank Act (S.C. 1991, c. 46) — full consolidated text",
        doc_type="bank_act",
        regulatory_body="Department of Justice Canada",
        jurisdiction="canada",
        url="https://laws-lois.justice.gc.ca/PDF/B-1.01.pdf",
        file_ext="pdf",
    ),

    # === GDPR ===
    Source(
        doc_id="gdpr_consolidated",
        doc_title="General Data Protection Regulation (Regulation (EU) 2016/679) — consolidated text",
        doc_type="gdpr",
        regulatory_body="European Parliament & Council",
        jurisdiction="eu",
        url="https://gdpr-info.eu/",
        file_ext="html",
        effective_date="2018-05-25",
        doc_version="EU 2016/679",
    ),

    # === Federal Reserve Regulation W (transactions with affiliates) ===
    Source(
        doc_id="fed_reg_w",
        doc_title="Federal Reserve Regulation W — 12 CFR Part 223 (Transactions Between Member Banks and Their Affiliates)",
        doc_type="fed",
        regulatory_body="Federal Reserve Board",
        jurisdiction="usa",
        # govinfo's pkg-pdf URLs return a JS landing page; the /link/ shortcut
        # redirects to the actual PDF blob.
        url="https://www.govinfo.gov/link/cfr/12/223",
        file_ext="pdf",
        doc_version="CFR-current",
    ),
]


@dataclass
class DownloadResult:
    doc_id: str
    url: str
    success: bool
    bytes_written: int = 0
    skipped: bool = False
    error: str = ""


def download_one(client: httpx.Client, src: Source) -> DownloadResult:
    target = RAW_DIR / f"{src.doc_id}.{src.file_ext}"
    meta_target = RAW_DIR / f"{src.doc_id}.meta.json"

    # Always (re)write metadata sidecar — cheap and ensures it stays in sync.
    meta_target.write_text(json.dumps(asdict(src), indent=2))

    if target.exists() and target.stat().st_size > 0:
        return DownloadResult(
            doc_id=src.doc_id, url=src.url, success=True,
            bytes_written=target.stat().st_size, skipped=True,
        )

    try:
        r = client.get(src.url, follow_redirects=True)
        r.raise_for_status()
        target.write_bytes(r.content)
        return DownloadResult(
            doc_id=src.doc_id, url=src.url, success=True,
            bytes_written=len(r.content),
        )
    except Exception as e:
        return DownloadResult(
            doc_id=src.doc_id, url=src.url, success=False, error=f"{type(e).__name__}: {e}",
        )


def main() -> int:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    results: list[DownloadResult] = []

    with httpx.Client(headers=headers, timeout=TIMEOUT) as client:
        for src in tqdm(SOURCES, desc="compliance docs"):
            results.append(download_one(client, src))

    manifest = {
        "downloaded_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "n_sources": len(SOURCES),
        "n_success": sum(1 for r in results if r.success),
        "n_skipped": sum(1 for r in results if r.skipped),
        "n_failed": sum(1 for r in results if not r.success),
        "total_bytes": sum(r.bytes_written for r in results if r.success),
        "results": [asdict(r) for r in results],
    }
    (RAW_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== Compliance download summary ===")
    print(f"  ok:      {manifest['n_success']}/{manifest['n_sources']}")
    print(f"  skipped: {manifest['n_skipped']}  (already on disk)")
    print(f"  failed:  {manifest['n_failed']}")
    print(f"  bytes:   {manifest['total_bytes']:,}")

    failures = [r for r in results if not r.success]
    if failures:
        print("\n--- failures ---")
        for r in failures:
            print(f"  {r.doc_id}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
