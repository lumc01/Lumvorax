#!/usr/bin/env python3
"""Fail if tracked binary files are present in quantum_problem_hubbard_hts.

Binary detection is content-based (NUL-byte in first 8 KiB), with allowlist
for known binary-like but acceptable formats if ever needed.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
SCOPE = pathlib.Path("src/advanced_calculations/quantum_problem_hubbard_hts")
ALLOWLIST_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".parquet"
}


def git_ls_files(scope: pathlib.Path) -> list[pathlib.Path]:
    out = subprocess.check_output([
        "git", "ls-files", str(scope)
    ], cwd=ROOT, text=True)
    return [pathlib.Path(line) for line in out.splitlines() if line.strip()]


def is_binary(path: pathlib.Path) -> bool:
    if path.suffix.lower() in ALLOWLIST_SUFFIXES:
        return False
    full = ROOT / path
    try:
        data = full.read_bytes()[:8192]
    except OSError:
        return False
    return b"\x00" in data


def main() -> int:
    offenders = [p for p in git_ls_files(SCOPE) if is_binary(p)]
    if offenders:
        print("[FAIL] Tracked binaries detected in quantum_problem_hubbard_hts:")
        for p in offenders:
            print(f" - {p}")
        return 1

    print("[PASS] No tracked binaries detected in quantum_problem_hubbard_hts scope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
