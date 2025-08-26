import re
from typing import Dict, List, Pattern

def compiled_patterns() -> Dict[str, Pattern]:
    pats = {
        "SL_NIC_12": r"\b\d{12}\b",
        # tolerant: 9 digits then up to 3 non-word chars, then V/X (old NIC)
        "SL_NIC_9V_TOL": r"\b\d{9}\W{0,3}[VvXxYy]\b",
        "Passport": r"\b[A-Z]\d{7}\b",
    }
    return {name: re.compile(rx) for name, rx in pats.items()}

def _canon_old_nic(raw: str) -> str:
    """Normalize things like '966074060 v' or '966074060_y' → '966074060V'."""
    s = re.sub(r"\W+", "", raw)   # drop spaces/underscores/dashes
    if len(s) == 10 and s[-1] in "Yy":
        s = s[:-1] + "V"
    return s.upper()

def find_ids(text: str) -> Dict[str, List[str]]:
    text = text or ""
    patterns = compiled_patterns()
    found: Dict[str, List[str]] = {}

    for name, rx in patterns.items():
        hits = []
        for m in rx.finditer(text):
            val = m.group(0)
            # Canonicalize old NIC if matched by tolerant rule
            if name == "SL_NIC_9V_TOL":
                val = _canon_old_nic(val)
            hits.append(val)
        # de-dup, keep order
        hits = list(dict.fromkeys(hits))
        if hits:
            found[name] = hits
    return found

def squash_spaces(text: str) -> str:
    if not text:
        return ""
    # Join spaced/hyphenated 12-digit blocks → 123456789012
    text = re.sub(r"\b(\d{4})[ \t\-](\d{4})[ \t\-](\d{4})\b", r"\1\2\3", text)
    # Normalize old NIC tails with small gaps/underscores; Y→V
    def _fix(m: re.Match) -> str:
        head, tail = m.group(1), m.group(2).upper()
        if tail == "Y":
            tail = "V"
        return f"{head}{tail}"
    text = re.sub(r"\b(\d{9})[\W_]{0,3}([VvXxYy])\b", _fix, text)

    # Collapse leftover spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
