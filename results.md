# Results

Execution accuracy (EX) on the Spider development set, 1,034 questions. A prediction is correct when it runs and returns the same rows as the reference query. Scoring is done by [`minnow/execution.py`](minnow/execution.py) on the original Spider databases.

| Date | Model | Change | Dev EX (%) | Correct | Files |
| --- | --- | --- | ---: | ---: | --- |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct | Baseline: no training, greedy decoding, CREATE TABLE statements as the schema | 45.36 | 469 / 1034 | [results/baseline](results/baseline) |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-spider | LoRA fine-tuning on Spider train: 6,142 examples of at most 1,024 tokens, 3 epochs | 60.64 | 627 / 1034 | [results/sft-spider](results/sft-spider) |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-spider | Same model as sft-spider; five candidate queries per question, keeping the result most candidates agree on | 64.31 | 665 / 1034 | [results/sft-spider-vote5](results/sft-spider-vote5) |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-spider-all | LoRA fine-tuning on Spider train plus train_others: 7,800 examples, 3 epochs, one greedy query | 59.77 | 618 / 1034 | [results/sft-spider-all](results/sft-spider-all) |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-spider-all | Same model as sft-spider-all; vote over five candidate queries | 65.09 | 673 / 1034 | [results/sft-spider-all-vote5](results/sft-spider-all-vote5) |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | LoRA fine-tuning with one example row per table in the schema: 5,259 examples, one greedy query | 61.51 | 636 / 1034 | [results/sft-rows1](results/sft-rows1) |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | Same model as sft-rows1; vote over five candidate queries | 65.76 | 680 / 1034 | [results/sft-rows1-vote5](results/sft-rows1-vote5) |
| 2026-11-21 | vote of sft-rows1-vote5, sft-spider-all-vote5, sft-spider-vote5 | Vote across the three fine-tuned models' voted answers (15 candidate queries per question in total) | 68.67 | 710 / 1034 | [results/vote-of-three](results/vote-of-three) |

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

### 2026-11-21, sft-spider-vote5

- 130 predictions did not run. The most common errors: `no such column` (120), `no such table` (6), `syntax error` (2).
- 239 predictions ran and returned different rows.
- 42 of the 665 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 979 seconds.

### 2026-11-21, sft-spider-all

- 228 predictions did not run. The most common errors: `no such column` (215), `no such table` (9), `misuse of aggregate` (4).
- 188 predictions ran and returned different rows.
- 40 of the 618 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 191 seconds.

### 2026-11-21, sft-spider-all-vote5

- 137 predictions did not run. The most common errors: `no such column` (131), `no such table` (4), `misuse of aggregate` (2).
- 224 predictions ran and returned different rows.
- 43 of the 673 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 991 seconds.

### 2026-11-21, sft-rows1

- 212 predictions did not run. The most common errors: `no such column` (192), `no such table` (13), `misuse of aggregate` (2).
- 186 predictions ran and returned different rows.
- 40 of the 636 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 199 seconds.

### 2026-11-21, sft-rows1-vote5

- 127 predictions did not run. The most common errors: `no such column` (118), `no such table` (9).
- 227 predictions ran and returned different rows.
- 43 of the 680 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 1,017 seconds.

### 2026-11-21, vote-of-three

- 58 predictions did not run. The most common errors: `no such column` (55), `no such table` (3).
- 266 predictions ran and returned different rows.
- 42 of the 710 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
