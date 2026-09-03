# FlamAI Assignment Submission

## Structure

- `tokenizer_audit/` — Part A
- `serving_analysis/` — Part B
- `model_strategy/` — Part C
- `NOTEBOOK.md` — chronological experiments
- `AI_USAGE.md` — AI assistance disclosure

## Setup

```bash
pip install -r requirements.txt
```

Run tokenizer_audit:

```bash
python tokenizer_audit/scripts/audit_fertility.py \
  --corpus eng=starter_corpus/eng_sample.txt \
  --corpus hin=starter_corpus/hin_sample.txt
```

Run tokenizer comparison:

```bash
python tokenizer_audit/scripts/compare_tokenizers.py \
  --corpus eng=starter_corpus/eng_sample.txt \
  --corpus hin=starter_corpus/hin_sample.txt \
  --tokenizer gpt2 \
  --tokenizer hf:xlm-roberta-base
```

Run serving_analysis:

```bash
python serving_analysis/analyze_bench.py \
  --csv bench/bench_log.csv \
  --model-spec bench/model_spec.md
```

Replace the starter corpus with the larger evaluation corpus required by the assignment before final submission.
"# FlamAI" 
