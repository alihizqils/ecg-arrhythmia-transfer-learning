"""Verify that the code and data protocol in this repository match the locks recorded by the official runs.

Without importing torch, the script checks:
  * Data protocol: the SHA-256 of every file listed in the protocol lock, and the protocol hash itself.
  * Method (S1/S2 and S3): every file that enters the method hashes is either byte-identical to the file as run
    (SHA-256 in scripts/lock_equivalence.json, whose files reproduce the official method hashes), or differs only
    in full-line comments or the module docstring (identical code fingerprint, scripts/_fingerprint.py).

It also prints the method hashes of the released files; new runs from this repository record those values.

Usage: python scripts/verify_hashes.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _fingerprint import code_fingerprint  # noqa: E402

REPO = HERE.parent
CODE = REPO / "code"
OUT = CODE / "eksperimen_v4_svdb/outputs_colab_r15"
DATA = CODE / "data/processed/v4_svdb"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def protocol_path(key: str) -> Path:
    root, _, rest = key.partition("/")
    if root == "source":
        return DATA / "global" / rest
    if root == "prepared":
        return DATA / "prepared" / rest
    if key == "fase_b/protocol_v4_lock.json":
        return DATA / "protocol_v4_lock.json"
    return CODE / key


def check_protocol() -> bool:
    lock = json.loads((OUT / "protocol_lock.json").read_text())
    ok = True
    for key, want in sorted(lock["files"].items()):
        p = protocol_path(key)
        if not p.exists() or sha(p) != want:
            print(f"FAIL protocol file {key}"); ok = False
    if ok:
        got = hashlib.sha256(json.dumps({k: sha(protocol_path(k)) for k in lock["files"]}, sort_keys=True).encode()).hexdigest()
        ok = got == lock["protocol_hash"]
    print(f"{'OK  ' if ok else 'FAIL'} data protocol: {len(lock['files'])} files byte-identical, "
          f"protocol hash {lock['protocol_hash'][:12]}")
    return ok


def check_method(eq: dict) -> bool:
    ok, same, comment_only = True, 0, []
    for f in eq["method_files"] + eq["s3_files"]:
        p = REPO / f
        if sha(p) == eq["original_sha256"][f]:
            same += 1
        elif f in eq["files"] and code_fingerprint(p) == eq["files"][f]["code_fingerprint"]:
            comment_only.append(f)
        else:
            print(f"FAIL method file {f}: code differs from the run version"); ok = False
    m = json.loads((OUT / "v4_method_lock.json").read_text())["method_hash"]
    s3 = json.loads((OUT / "v4_s3_method_lock.json").read_text())["method_hash"]
    print(f"{'OK  ' if ok else 'FAIL'} method (official hash {m[:12]}, S3 {s3[:12]}): {same} files byte-identical, "
          f"{len(comment_only)} differ only in comments")
    for f in comment_only:
        print(f"       comments only: {f}")
    return ok


def released_method_hashes(eq: dict) -> tuple[str, str]:
    m = hashlib.sha256(b"".join(f[len("code/"):].encode() + (REPO / f).read_bytes() for f in eq["method_files"])).hexdigest()
    s3 = hashlib.sha256(m.encode() + b"".join((REPO / f).read_bytes() for f in eq["s3_files"])).hexdigest()
    return m, s3


def main() -> int:
    eq = json.loads((HERE / "lock_equivalence.json").read_text())
    ok = check_protocol() & check_method(eq)
    m, s3 = released_method_hashes(eq)
    print(f"Method hash of the released files (recorded by new runs): {m[:12]}, S3 {s3[:12]}")
    print("ALL CHECKS PASSED" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
