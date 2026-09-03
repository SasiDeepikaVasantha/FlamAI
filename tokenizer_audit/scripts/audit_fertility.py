#!/usr/bin/env python3
import argparse
import re
import unicodedata
from statistics import mean

def load_tokenizer(spec):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    import tiktoken
    return tiktoken.get_encoding(spec).encode

def read_lines(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(unicodedata.normalize("NFC", line))
    return lines

def original_analyze(lines, encode):
    """Exact logic of the supplied fertility.py, kept for audit comparison."""
    fert = []
    tpc = []
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        words = line.split(" ")
        chars = len(line)
        fert.append(len(tokens) / len(words))
        tpc.append(len(tokens) / chars)
    return mean(fert), mean(tpc)

def corrected_analyze(lines, encode):
    """Corpus-level metrics with whitespace tokenization and multiple denominators."""
    token_total = 0
    word_total = 0
    codepoint_total = 0
    byte_total = 0
    line_count = 0

    for raw in lines:
        line = raw.lower()
        tokens = encode(line)
        words = re.findall(r"\S+", line)

        token_total += len(tokens)
        word_total += len(words)
        codepoint_total += len(line)
        byte_total += len(line.encode("utf-8"))
        line_count += 1

    return {
        "lines": line_count,
        "tokens": token_total,
        "words": word_total,
        "tokens_per_word": token_total / word_total,
        "tokens_per_codepoint": token_total / codepoint_total,
        "tokens_per_utf8_byte": token_total / byte_total,
        "tokens_per_line": token_total / line_count,
    }

def audit_whitespace(lines, encode):
    """Isolate the effect of split(' ') vs whitespace-aware splitting."""
    old = original_analyze(lines, encode)[0]

    token_total = 0
    old_words = 0
    new_words = 0
    for raw in lines:
        line = raw.lower()
        token_total += len(encode(line))
        old_words += len(line.split(" "))
        new_words += len(re.findall(r"\S+", line))

    return {
        "original_per_line_avg_tok_per_word": old,
        "aggregate_with_split_space": token_total / old_words,
        "aggregate_with_regex_whitespace": token_total / new_words,
        "word_count_difference": old_words - new_words,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    encode = load_tokenizer(args.tokenizer)

    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path)

        old_fert, old_tpc = original_analyze(lines, encode)
        new = corrected_analyze(lines, encode)
        ws = audit_whitespace(lines, encode)

        print(f"\n=== {lang} ===")
        print(f"Original tok/word:       {old_fert:.6f}")
        print(f"Original tok/char:       {old_tpc:.6f}")
        print(f"Corrected tok/word:      {new['tokens_per_word']:.6f}")
        print(f"Tokens/codepoint:        {new['tokens_per_codepoint']:.6f}")
        print(f"Tokens/UTF-8 byte:       {new['tokens_per_utf8_byte']:.6f}")
        print(f"Tokens/line:             {new['tokens_per_line']:.6f}")
        print(f"Original-vs-regex words: {ws['word_count_difference']} extra split(' ') fields")

if __name__ == "__main__":
    main()
