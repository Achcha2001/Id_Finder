# app.py
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.ocr_engine import extract_text_from_pdf, ocr_image_file, TESSERACT_PATH

# ----------------- paths & app -----------------
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ID Extractor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")  # allow clicking filenames
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ----------------- routes -----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok", "tesseract_path": TESSERACT_PATH or "NOT FOUND"}

# ----------------- helpers -----------------
def _safe_name(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".", " ")).strip()[:255]

def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}

# ---- ID extraction focused on 9 digits + V (old NIC) with anchors + fuzzy + voting ----

# tolerant raw hit: 9 digits, up to a few non-word chars, then V/v/Y/y
_RX_V_TOL = re.compile(r"\b(\d{9})[\W_]{0,4}([VvYy])\b")

# VERY tolerant (allow OCR letters that look like digits inside the 9 slots)
_RX_V_FUZZY = re.compile(
    r"\b([0-9OQDBIl|ZzSsGgq]{3})[\W_]{0,3}([0-9OQDBIl|ZzSsGgq]{3})[\W_]{0,3}([0-9OQDBIl|ZzSsGgq]{3})[\W_]{0,4}([VvYy])\b"
)

# anchors commonly present on licences/forms (capture NIC field specifically)
_RX_ANCHORS = [
    re.compile(r"(?is)\b4d\b\.?\s*[:\-]?\s*(\d{9})\s*([VvYy])"),           # "4d. 913153782 V"
    re.compile(r"(?is)\bNIC(?:\s*No\.?)?\b.*?(\d{9})\s*([VvYy])"),         # "... NIC No 966074060 V"
    re.compile(r"(?is)\bID(?:\s*No\.?)?\b.*?(\d{9})\s*([VvYy])"),
]

# common OCR confusions -> digits
_OCR_TO_DIGIT = str.maketrans({
    "O":"0","o":"0","Q":"0","D":"0",
    "I":"1","l":"1","|":"1",
    "Z":"2","z":"2",
    "S":"5","s":"5",
    "B":"8",
    "G":"6","g":"6",
    "q":"9"
})

def _canon_v(head: str, tail: str) -> str:
    """Normalize tail Y->V, uppercase."""
    t = tail.upper()
    if t == "Y":
        t = "V"
    return f"{head}{t}"

def _normalize_text_for_counts(text: str) -> str:
    """Make spaced/hyphenated blocks searchable: 123 456 789 v -> 123456789V."""
    if not text:
        return ""
    # join 3-3-3 blocks
    text = re.sub(r"\b(\d{3})[ \t\-_.:]+(\d{3})[ \t\-_.:]+(\d{3})\b", r"\1\2\3", text)
    # merge small gaps before the tail letter and Y->V
    def _fix(m: re.Match) -> str:
        return _canon_v(m.group(1), m.group(2))
    text = re.sub(r"\b(\d{9})[\W_]{0,4}([VvYy])\b", _fix, text)
    return text.upper()

def _extract_v_ids_all_runs(raw_texts: List[str]) -> List[str]:
    """
    From multiple OCR outputs, return the best V-ending IDs:
    1) Anchored matches (4d., NIC, ID No)
    2) Tolerant raw hits (9d + V)
    3) Fuzzy hits (replace O/I/S/B/G/q… with digits)
    Then rank by frequency across all normalized text.
    """
    anchored: List[str] = []
    for t in raw_texts:
        if not t:
            continue
        for rx in _RX_ANCHORS:
            for m in rx.finditer(t):
                anchored.append(_canon_v(m.group(1), m.group(2)))

    tolerant: List[str] = []
    for t in raw_texts:
        if not t:
            continue
        for m in _RX_V_TOL.finditer(t):
            tolerant.append(_canon_v(m.group(1), m.group(2)))

    fuzzy: List[str] = []
    for t in raw_texts:
        if not t:
            continue
        for m in _RX_V_FUZZY.finditer(t):
            a, b, c, tail = m.group(1, 2, 3, 4)
            digits = (a + b + c).translate(_OCR_TO_DIGIT)
            if len(digits) == 9 and digits.isdigit():
                fuzzy.append(_canon_v(digits, tail))

    # frequency voting on a big normalized pool
    big_norm = " \n ".join(_normalize_text_for_counts(t) for t in raw_texts if t)
    counts = Counter()
    for cand in anchored + tolerant + fuzzy:
        c = cand.upper()
        counts[c] += big_norm.count(c)

    def _dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for x in seq:
            u = x.upper()
            if u not in seen:
                seen.add(u); out.append(u)
        return out

    anchored_u = _dedup(anchored)
    tolerant_u = [x for x in _dedup(tolerant) if x not in anchored_u]
    fuzzy_u    = [x for x in _dedup(fuzzy)    if x not in anchored_u and x not in tolerant_u]

    anchored_sorted = sorted(anchored_u, key=lambda x: -counts[x])
    tolerant_sorted = sorted(tolerant_u, key=lambda x: -counts[x])
    fuzzy_sorted    = sorted(fuzzy_u,    key=lambda x: -counts[x])

    return anchored_sorted + tolerant_sorted + fuzzy_sorted

# ----------------- upload API (fast pass + conditional forced OCR) -----------------
@app.post("/api/upload")
async def upload(
    request: Request,
    files: List[UploadFile] = File(...),
):
    if not files:
        return JSONResponse({"error": "No files uploaded"}, status_code=400)

    results: List[Dict[str, Any]] = []
    ts = time.strftime("%Y%m%d-%H%M%S")

    # per-run folder & deterministic naming to prevent mismatches/overwrites
    run_dir = UPLOAD_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for idx, uf in enumerate(files, start=1):
        original = uf.filename or f"file-{idx}"
        safe_orig = _safe_name(original)
        dest = run_dir / f"{idx:03d}__{safe_orig}"

        with dest.open("wb") as f:
            f.write(await uf.read())

        try:
            debug_note = ""
            raw_runs: List[str] = []
            ids_v: List[str] = []

            if _is_pdf(dest):
                # PASS 1 (FAST): use embedded text; OCR only image-only pages internally
                t1, notes1 = extract_text_from_pdf(str(dest), dpi_scale=2.0, force_ocr=False)
                raw_runs.append(t1 or "")
                ids_v = _extract_v_ids_all_runs([t1])
                debug_note = f"PDF pages: {', '.join(notes1)}"

                # Only if still nothing, do a full forced-OCR pass (SLOWER)
                if not ids_v:
                    t2, notes2 = extract_text_from_pdf(str(dest), dpi_scale=2.4, force_ocr=True)
                    raw_runs.append(t2 or "")
                    ids_v = _extract_v_ids_all_runs([t1, t2])
                    if t2:
                        debug_note = f"{debug_note} + FORCED_OCR"

            elif _is_image(dest):
                t = ocr_image_file(str(dest))
                raw_runs.append(t or "")
                ids_v = _extract_v_ids_all_runs([t])
                debug_note = "IMG:OCR"

            else:
                try:
                    t = dest.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    t = ""
                raw_runs.append(t or "")
                ids_v = _extract_v_ids_all_runs([t])
                debug_note = "TXT" if t else "UNSUPPORTED"

            # If nothing found, save combined text for debugging
            if not ids_v:
                big_dbg = "\n\n---RUN---\n\n".join(raw_runs).strip()
                if big_dbg:
                    (OUTPUT_DIR / f"dbg_{dest.stem}.txt").write_text(big_dbg, encoding="utf-8")

            results.append({
                "original": safe_orig,
                "filename": safe_orig,
                "saved_as": dest.name,
                "ids": ids_v,
                "note": debug_note,
            })

        except Exception as e:
            results.append({
                "original": safe_orig,
                "filename": safe_orig,
                "saved_as": dest.name,
                "ids": [],
                "error": str(e),
                "note": debug_note if 'debug_note' in locals() else "",
            })

    # Optional CSV for auditing
    rows = []
    for item in results:
        if item.get("ids"):
            for idv in item["ids"]:
                rows.append({"original": item["original"], "saved_as": item["saved_as"], "id": idv})
        else:
            rows.append({"original": item["original"], "saved_as": item["saved_as"], "id": ""})
    out_csv = OUTPUT_DIR / f"ids_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    return {
        "ok": True,
        "count": len(results),
        "results": results,
        "csv_url": f"/outputs/{out_csv.name}",
        "run_dir": f"/uploads/run_{ts}",
    }
