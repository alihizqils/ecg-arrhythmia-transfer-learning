"""Comment-insensitive fingerprint used by verify_hashes.py.

Removes full-line comments (lines whose first non-blank character is '#') and, for Python files, the module
docstring at the top of the file. Everything else, including inline comments and all code, is kept byte for byte.
"""
import hashlib
from pathlib import Path


def code_fingerprint(path: Path) -> str:
    text = path.read_bytes().decode("utf-8")
    if path.suffix == ".py" and text.startswith('"""'):
        text = text[text.index('"""', 3) + 3:]
    kept = [line for line in text.split("\n") if not line.lstrip().startswith("#")]
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()
