"""Download SEC EDGAR filings for the 5 target banks.

US filers (JPM, BAC, GS): 10-K x2, 10-Q x4, 8-K x1 (earnings — item 2.02)
Canadian filers (TD, RY): 40-F x2 (annual), 6-K x4 (interim)

Rationale for 40-F/6-K substitution: TD and Royal Bank of Canada are foreign
private issuers under SEC rules — they don't file 10-K/10-Q. 40-F is the
annual analogue (similar disclosures), 6-K is used for material interim
events (often containing quarterly earnings).

Idempotent. Polite to EDGAR (~6 req/sec, well under the 10 req/sec limit).

Output:
  data/raw/credit/<ticker>_<form>_<date>_<accession>.<ext>
  data/raw/credit/<ticker>_<form>_<date>_<accession>.meta.json
  data/raw/credit/_manifest.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "credit"
RAW_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "BankMind Research arjunghumman1995@gmail.com"
TIMEOUT = httpx.Timeout(60.0, connect=20.0)
EDGAR_DELAY = 0.15  # seconds — keeps us well under SEC's 10 req/sec cap


@dataclass
class Company:
    ticker: str
    cik: int
    name: str
    jurisdiction: str            # 'usa' | 'canada'
    industry_sector: str
    naics_code: str
    forms: dict                  # {form_type: count_to_fetch}


COMPANIES = [
    Company(
        ticker="JPM", cik=19617, name="JPMorgan Chase & Co",
        jurisdiction="usa", industry_sector="banking", naics_code="522110",
        forms={"10-K": 2, "10-Q": 4, "8-K": 1},
    ),
    Company(
        ticker="BAC", cik=70858, name="Bank of America Corporation",
        jurisdiction="usa", industry_sector="banking", naics_code="522110",
        forms={"10-K": 2, "10-Q": 4, "8-K": 1},
    ),
    Company(
        ticker="GS", cik=886982, name="The Goldman Sachs Group Inc",
        jurisdiction="usa", industry_sector="investment_banking", naics_code="523120",
        forms={"10-K": 2, "10-Q": 4, "8-K": 1},
    ),
    Company(
        ticker="TD", cik=947263, name="Toronto-Dominion Bank",
        jurisdiction="canada", industry_sector="banking", naics_code="522110",
        forms={"40-F": 2, "6-K": 4},
    ),
    Company(
        ticker="RY", cik=1000275, name="Royal Bank of Canada",
        jurisdiction="canada", industry_sector="banking", naics_code="522110",
        forms={"40-F": 2, "6-K": 4},
    ),
]


def fetch_submissions(client: httpx.Client, cik: int) -> dict:
    pad = f"{cik:010d}"
    r = client.get(f"https://data.sec.gov/submissions/CIK{pad}.json")
    r.raise_for_status()
    return r.json()


def select_filings(submissions: dict, company: Company) -> list[dict]:
    """Pick most-recent N per form type. Filter 8-K to earnings items (2.02)."""
    recent = submissions["filings"]["recent"]
    n = len(recent["form"])
    candidates = []
    for i in range(n):
        candidates.append({
            "form": recent["form"][i],
            "accession": recent["accessionNumber"][i],
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i] if i < len(recent.get("reportDate", [])) else "",
            "primary_doc": recent["primaryDocument"][i],
            "primary_doc_desc": recent.get("primaryDocDescription", [""] * n)[i] or "",
            "items": recent.get("items", [""] * n)[i] or "",
        })

    selected: list[dict] = []
    for form, count in company.forms.items():
        matching = [c for c in candidates if c["form"] == form]
        if form == "8-K":
            # Filter to earnings releases (Item 2.02 = Results of Operations and Financial Condition)
            matching = [c for c in matching if "2.02" in c["items"]]
        # Most recent first
        matching.sort(key=lambda c: c["filing_date"], reverse=True)
        selected.extend(matching[:count])
    return selected


def build_filing_url(cik: int, accession: str, primary_doc: str) -> str:
    acc_clean = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{primary_doc}"


@dataclass
class DownloadResult:
    ticker: str
    form: str
    filing_date: str
    accession: str
    url: str
    success: bool
    bytes_written: int = 0
    skipped: bool = False
    error: str = ""


def download_one(client: httpx.Client, company: Company, filing: dict) -> DownloadResult:
    url = build_filing_url(company.cik, filing["accession"], filing["primary_doc"])
    # Most EDGAR primary docs are .htm; preserve the actual extension from filename.
    ext = Path(filing["primary_doc"]).suffix.lstrip(".").lower() or "htm"
    safe_acc = filing["accession"].replace("-", "")
    base = f"{company.ticker}_{filing['form'].replace('/', '_')}_{filing['filing_date']}_{safe_acc}"
    target = RAW_DIR / f"{base}.{ext}"
    meta_target = RAW_DIR / f"{base}.meta.json"

    fiscal_year = None
    fiscal_quarter = None
    rd = filing.get("report_date", "")
    if rd:
        try:
            y, m, _ = rd.split("-")
            fiscal_year = int(y)
            fiscal_quarter = (int(m) - 1) // 3 + 1
        except ValueError:
            pass

    meta = {
        "doc_id": base,
        "doc_title": f"{company.name} — {filing['form']} ({filing['filing_date']})",
        "doc_type": filing["form"].lower().replace("-", ""),  # '10k', '10q', '8k', '40f', '6k'
        "company_ticker": company.ticker,
        "company_name": company.name,
        "company_cik": company.cik,
        "jurisdiction": company.jurisdiction,
        "industry_sector": company.industry_sector,
        "naics_code": company.naics_code,
        "form": filing["form"],
        "accession": filing["accession"],
        "filing_date": filing["filing_date"],
        "report_date": filing["report_date"],
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "primary_doc": filing["primary_doc"],
        "primary_doc_desc": filing["primary_doc_desc"],
        "source_url": url,
    }
    meta_target.write_text(json.dumps(meta, indent=2))

    if target.exists() and target.stat().st_size > 0:
        return DownloadResult(
            ticker=company.ticker, form=filing["form"], filing_date=filing["filing_date"],
            accession=filing["accession"], url=url,
            success=True, bytes_written=target.stat().st_size, skipped=True,
        )

    try:
        r = client.get(url, follow_redirects=True)
        r.raise_for_status()
        target.write_bytes(r.content)
        return DownloadResult(
            ticker=company.ticker, form=filing["form"], filing_date=filing["filing_date"],
            accession=filing["accession"], url=url,
            success=True, bytes_written=len(r.content),
        )
    except Exception as e:
        return DownloadResult(
            ticker=company.ticker, form=filing["form"], filing_date=filing["filing_date"],
            accession=filing["accession"], url=url,
            success=False, error=f"{type(e).__name__}: {e}",
        )


def main() -> int:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    results: list[DownloadResult] = []

    with httpx.Client(headers=headers, timeout=TIMEOUT) as client:
        for company in COMPANIES:
            print(f"\n{company.ticker} (CIK {company.cik}) — {company.name}")
            try:
                subs = fetch_submissions(client, company.cik)
            except Exception as e:
                print(f"  ! failed to fetch submissions: {e}")
                continue
            time.sleep(EDGAR_DELAY)

            filings = select_filings(subs, company)
            print(f"  selected {len(filings)} filings: " +
                  ", ".join(f"{f['form']}@{f['filing_date']}" for f in filings))

            for filing in tqdm(filings, desc=f"  {company.ticker}", leave=False):
                results.append(download_one(client, company, filing))
                time.sleep(EDGAR_DELAY)

    manifest = {
        "downloaded_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "n_filings": len(results),
        "n_success": sum(1 for r in results if r.success),
        "n_skipped": sum(1 for r in results if r.skipped),
        "n_failed": sum(1 for r in results if not r.success),
        "total_bytes": sum(r.bytes_written for r in results if r.success),
        "results": [asdict(r) for r in results],
    }
    (RAW_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== EDGAR download summary ===")
    print(f"  ok:      {manifest['n_success']}/{manifest['n_filings']}")
    print(f"  skipped: {manifest['n_skipped']}")
    print(f"  failed:  {manifest['n_failed']}")
    print(f"  bytes:   {manifest['total_bytes']:,}")
    if manifest["n_failed"]:
        print("\n--- failures ---")
        for r in results:
            if not r.success:
                print(f"  {r.ticker} {r.form} {r.filing_date}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
