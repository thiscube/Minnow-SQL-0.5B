# Results

Execution accuracy (EX) on the Spider development set, 1,034 questions. A prediction is correct when it runs and returns the same rows as the reference query. Scoring is done by [`minnow/execution.py`](minnow/execution.py) on the original Spider databases.

| Date | Model | Change | Dev EX (%) | Correct | Files |
| --- | --- | --- | ---: | ---: | --- |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct | Baseline: no training, greedy decoding, CREATE TABLE statements as the schema | 45.36 | 469 / 1034 | [results/baseline](results/baseline) |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-spider | LoRA fine-tuning on Spider train: 6,142 examples of at most 1,024 tokens, 3 epochs | 60.64 | 627 / 1034 | [results/sft-spider](results/sft-spider) |

## Notes

### 2026-11-21, baseline

- 389 predictions did not run. The most common errors: `no such column` (356), `no such table` (15), `misuse of aggregate` (7).
- 176 predictions ran and returned different rows.
- 30 of the 469 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 291 seconds.

### 2026-11-21, sft-spider

- 208 predictions did not run. The most common errors: `no such column` (189), `no such table` (9), `syntax error` (4).
- 199 predictions ran and returned different rows.
- 40 of the 627 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 182 seconds.
