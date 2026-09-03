#!/usr/bin/env python3
import argparse
import re
import unicodedata

def load_tokenizer(spec):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    import tiktoken
    return tiktoken.get_encoding(spec).encode

def read_lines(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s:
                out.append(unicodedata.normalize("NFC", s).lower())
    return out

def grapheme_count(s):
    # regex supports Unicode extended grapheme clusters via \X.
    import regex
    return len(regex.findall(r"\X", s))

def analyze(lines, encode):
    tokens = words = codepoints = graphemes = utf8_bytes = 0
    for line in lines:
        tokens += len(encode(line))
        words += len(re.findall(r"\S+", line))
        codepoints += len(line)
        graphemes += grapheme_count(line)
        utf8_bytes += len(line.encode("utf-8"))

    return {
        "tokens_per_word": tokens / words,
        "tokens_per_codepoint": tokens / codepoints,
        "tokens_per_grapheme": tokens / graphemes,
        "tokens_per_utf8_byte": tokens / utf8_bytes,
        "tokens_per_line": tokens / len(lines),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--tokenizer", action="append", required=True)
    args = ap.parse_args()

    corpora = [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in args.corpus]

    for tok_spec in args.tokenizer:
        encode = load_tokenizer(tok_spec)
        print(f"\nTOKENIZER: {tok_spec}")
        print("lang,tok_per_word,tok_per_codepoint,tok_per_grapheme,tok_per_utf8_byte,tok_per_line")

        for lang, path in corpora:
            metrics = analyze(read_lines(path), encode)
            print(
                f"{lang},"
                f"{metrics['tokens_per_word']:.6f},"
                f"{metrics['tokens_per_codepoint']:.6f},"
                f"{metrics['tokens_per_grapheme']:.6f},"
                f"{metrics['tokens_per_utf8_byte']:.6f},"
                f"{metrics['tokens_per_line']:.6f}"
            )

if __name__ == "__main__":
    main()
