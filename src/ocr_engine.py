# src/ocr_engine.py
from typing import Tuple, List
import os, io
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import fitz  # PyMuPDF

def configure_tesseract():
    env_path = os.environ.get("TESSERACT_CMD")
    candidates = []
    if env_path: candidates.append(env_path)
    candidates += [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA",""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.getcwd(), "tools", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.getcwd(), "Tesseract-OCR", "tesseract.exe"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return None

TESSERACT_PATH = configure_tesseract()

def _deskew_osd(im: Image.Image) -> Image.Image:
    """Use Tesseract OSD to correct rotation when possible."""
    try:
        osd = pytesseract.image_to_osd(im)
        import re
        m = re.search(r"Rotate: (\d+)", osd)
        if m:
            angle = int(m.group(1)) % 360
            if angle:
                return im.rotate(-angle, expand=True)
    except Exception:
        pass
    return im

def _prep_variants(im: Image.Image) -> List[Image.Image]:
    """Generate several cleaned versions of the image to try with Tesseract."""
    base = im.convert("L")
    variants = []

    def _clean(x: Image.Image) -> Image.Image:
        # Contrast, median denoise, binarize
        x = ImageOps.autocontrast(x)
        x = x.filter(ImageFilter.MedianFilter(size=3))
        x = x.point(lambda p: 255 if p > 160 else 0)
        return x

    for scale in (1.0, 1.3, 1.6, 2.0):
        w, h = base.size
        scaled = base.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        variants.append(_clean(scaled))

    return variants

def _ocr_multi(im: Image.Image) -> str:
    """Try multiple rotations + PSMs + a whitelist and merge text."""
    if not TESSERACT_PATH:
        return ""
    text_runs: List[str] = []
    wh = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZvxVXyY"
    # deskew quickly, but still try other rotations for safety
    im0 = _deskew_osd(im.convert("RGB"))

    for rot in (0, 90, 180, 270):
        rim = im0.rotate(rot, expand=True)
        for var in _prep_variants(rim):
            for psm in (6, 7, 11, 12, 13):
                cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={wh}'
                try:
                    text_runs.append(pytesseract.image_to_string(var, lang="eng", config=cfg))
                except Exception:
                    pass

    return "\n".join([t for t in text_runs if t])

def ocr_image_file(path: str) -> str:
    with Image.open(path) as im:
        return _ocr_multi(im.convert("RGB"))

def _pil_from_pix(pix: fitz.Pixmap) -> Image.Image:
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def extract_text_from_pdf(path: str, dpi_scale: float = 3.0, force_ocr: bool = False) -> Tuple[str, List[str]]:
    """
    If force_ocr=False: use embedded text when present, else OCR.
    If force_ocr=True: OCR every page regardless.
    """
    doc = fitz.open(path)
    texts: List[str] = []
    notes: List[str] = []
    for page in doc:
        raw = (page.get_text("text") or "").strip()
        if raw and not force_ocr:
            texts.append(raw)
            notes.append("TEXT")
            continue

        if not TESSERACT_PATH:
            texts.append("")
            notes.append("NO_TESSERACT")
            continue

        # render high-DPI and OCR with robust pipeline
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        im = _pil_from_pix(pix)
        texts.append(_ocr_multi(im))
        notes.append("OCR")
    return "\n\n".join(texts), notes
