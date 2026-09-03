#!/usr/bin/env python3
"""
Prepare a UTF-8, NFC-normalized corpus from one text file per language.

Input format:
  one sentence per line

The script deliberately avoids language-specific rewriting so the evaluation
corpus remains auditable.
"""
import argparse
import unicodedata
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            s = unicodedata.normalize("NFC", line.strip())
            if s:
                kept.append(s)

    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for s in kept:
            f.write(s + "\n")

    print(f"Wrote {len(kept)} sentences to {dst}")

if __name__ == "__main__":
    main()
