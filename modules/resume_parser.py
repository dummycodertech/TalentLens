"""
modules/resume_parser.py — Google Drive PDF download + text extraction.

C8: COLUMN_ALIASES is in db.py, not here. This module only handles PDF extraction.
"""

import io
import re
import requests

# ─── Drive URL conversion ──────────────────────────────────────────────────────

DRIVE_PATTERNS = [
    # /file/d/{id}/view
    r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    # ?id={id}
    r"[?&]id=([a-zA-Z0-9_-]+)",
    # /open?id={id}
    r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
]


def extract_drive_file_id(share_url: str) -> str | None:
    """Extract file ID from any common Google Drive share URL format."""
    if not share_url or not isinstance(share_url, str):
        return None
    share_url = share_url.strip()
    for pattern in DRIVE_PATTERNS:
        m = re.search(pattern, share_url)
        if m:
            return m.group(1)
    return None


def drive_to_direct_url(share_url: str) -> str | None:
    """Convert a Drive sharing link to a direct-download URL."""
    file_id = extract_drive_file_id(share_url)
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ─── PDF download ──────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

DOWNLOAD_TIMEOUT = 30  # seconds


def download_pdf(url: str) -> bytes | None:
    """
    Download bytes from url, following redirects.
    Google Drive may redirect through a virus-scan confirmation page for large files.
    Handles the confirmation token automatically.
    """
    try:
        resp = _SESSION.get(url, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()

        # Handle Google Drive large-file confirmation page
        if "Content-Disposition" not in resp.headers and b"confirm=" in resp.content:
            token_match = re.search(rb"confirm=([0-9A-Za-z_-]+)", resp.content)
            if token_match:
                confirm_token = token_match.group(1).decode()
                confirmed_url = url + f"&confirm={confirm_token}"
                resp = _SESSION.get(confirmed_url, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
                resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and len(resp.content) < 1000:
            return None  # Likely an HTML error page, not a PDF

        return resp.content
    except Exception:
        return None


# ─── Text extraction ───────────────────────────────────────────────────────────

def extract_text_pdfplumber(pdf_bytes: bytes) -> tuple[str, bool]:
    """
    Primary extractor using pdfplumber.
    Returns (text, scan_warning) where scan_warning=True means little/no text was found.
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            full_text = "\n".join(pages_text).strip()
            scan_warning = len(full_text) < 100
            return full_text, scan_warning
    except Exception as e:
        return "", True


def extract_text_pymupdf(pdf_bytes: bytes) -> tuple[str, bool]:
    """
    Fallback extractor using PyMuPDF (fitz) for scanned/image-based PDFs.
    Returns (text, scan_warning).
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            t = page.get_text()
            if t:
                pages_text.append(t)
        full_text = "\n".join(pages_text).strip()
        scan_warning = len(full_text) < 100
        return full_text, scan_warning
    except Exception:
        return "", True


def extract_text(pdf_bytes: bytes) -> tuple[str, bool]:
    """
    Try pdfplumber first. Fall back to PyMuPDF if result is empty/short.
    """
    text, scan_warning = extract_text_pdfplumber(pdf_bytes)
    if scan_warning:
        text2, sw2 = extract_text_pymupdf(pdf_bytes)
        if len(text2) > len(text):
            return text2, sw2
    return text, scan_warning


# ─── Main parse entry point ────────────────────────────────────────────────────

def parse_resume(resume_url: str) -> dict:
    """
    Full pipeline: Drive URL → bytes → text.
    Returns:
        {
            "text": str,
            "scan_warning": bool,
            "error": str | None
        }
    """
    if not resume_url or not isinstance(resume_url, str) or not resume_url.strip():
        return {"text": "", "scan_warning": False, "error": "no_resume_url"}

    direct_url = drive_to_direct_url(resume_url.strip())
    if not direct_url:
        # Try using the URL directly (non-Drive links)
        direct_url = resume_url.strip()

    pdf_bytes = download_pdf(direct_url)
    if not pdf_bytes:
        return {"text": "", "scan_warning": False, "error": "download_failed"}

    text, scan_warning = extract_text(pdf_bytes)
    if not text:
        return {"text": "", "scan_warning": True, "error": "extraction_failed"}

    return {"text": text, "scan_warning": scan_warning, "error": None}


def parse_all(db, status_callback=None) -> dict[int, dict]:
    """
    Parse resumes for all candidates with status='uploaded'.
    Updates DB with resume_text and advances status.

    C9: Iterates over actual DB rows, never range(1, n+1).

    Args:
        db: the db module (passed in to avoid circular imports)
        status_callback: optional callable(s_no, status_str) for live UI updates

    Returns: dict of s_no -> parse result
    """
    candidates = db.get_by_status("uploaded")
    results = {}

    for row in candidates:
        sno = row["s_no"]
        if status_callback:
            status_callback(sno, "parsing resume…")

        result = parse_resume(row["resume_url"])
        results[sno] = result

        if result["error"]:
            error_status = (
                "resume_failed_scan" if result["error"] == "extraction_failed"
                else "resume_failed"
            )
            note = result["error"]
            if result.get("scan_warning"):
                note += " (scanned PDF — manual review needed)"
            db.update_status(sno, error_status, error=note)
        else:
            db.update_candidate(sno, resume_text=result["text"])
            if result["scan_warning"]:
                db.update_status(sno, "resume_parsed", error="scan_warning: low text yield, review manually")
            else:
                db.update_status(sno, "resume_parsed")

        if status_callback:
            status_callback(sno, "resume_parsed" if not result["error"] else "resume_failed")

    return results
