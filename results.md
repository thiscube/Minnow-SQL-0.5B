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
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | Same model as sft-rows1, voting over 15 candidate queries instead of 5 | 67.02 | 693 / 1034 | [results/sft-rows1-vote15](results/sft-rows1-vote15) |
| 2026-11-24 | Qwen2.5-Coder-0.5B-Instruct + sft-synsql | Fresh LoRA on Spider train plus a SynSQL-2.5M slice (12,215 examples that fit 1,024 tokens, 6,000 synthetic databases), 2 epochs, one greedy query | 61.80 | 639 / 1034 | [results/sft-synsql](results/sft-synsql) |
| 2026-11-24 | Qwen2.5-Coder-0.5B-Instruct + sft-synsql | Same model as sft-synsql; the prompt also lists question words found in the database's text values (value hints), one greedy query | 64.70 | 669 / 1034 | [results/sft-synsql-hints](results/sft-synsql-hints) |
| 2026-11-24 | Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | Same model as sft-rows1 (one example row per table); value hints added to the prompt, one greedy query | 62.28 | 644 / 1034 | [results/sft-rows1-hints](results/sft-rows1-hints) |
| 2026-11-24 | vote of sft-synsql-hints, sft-rows1-vote15, sft-spider-all-vote5 | Several models: vote across three fine-tuned adapters (sft-synsql with value hints, sft-rows1 with 15 candidates, sft-spider-all with 5), same majority rule as before | 71.18 | 736 / 1034 | [results/vote-hints-three](results/vote-hints-three) |
| 2026-11-24 | Qwen2.5-Coder-0.5B-Instruct + grpo-synsql | RL pilot: GRPO from sft-synsql on 600 fresh SynSQL questions (only 252 fit 900 prompt tokens), lr 2e-5, value hints, one greedy query | 63.64 | 658 / 1034 | [results/grpo600-hints](results/grpo600-hints) |
| 2026-11-24 | Qwen2.5-Coder-0.5B-Instruct + grpo-spider | RL pilot: GRPO from sft-synsql on 600 Spider train questions, lr 2e-5, value hints, one greedy query | 64.89 | 671 / 1034 | [results/grpo-spider600-hints](results/grpo-spider600-hints) |
| 2026-11-25 | Qwen2.5-Coder-0.5B-Instruct + grpo-spider-lr5 | GRPO from sft-synsql on 2,400 Spider train questions, lr 5e-6, value hints, one greedy query (65.38% at 800, 65.76% at 1,600) | 66.92 | 692 / 1034 | [results/grpo-lr5-2400-hints](results/grpo-lr5-2400-hints) |
| 2026-11-25 | vote of sft-synsql-hints, sft-rows1-vote15, sft-spider-all-vote5 | Several models: same three adapters as vote-hints-three, vote over each model's pick with empty and all-NULL results last; rule confirmed on 1,230 unseen Spider test questions (71.14% to 72.20%) | 72.63 | 751 / 1034 | [results/vote-hints-three-rows](results/vote-hints-three-rows) |
| 2026-11-25 | Qwen2.5-Coder-0.5B-Instruct + sft-synsql-hints | Trained with value hints in the prompt (Spider + SynSQL slice, 2 epochs), greedy | 62.48 | 646 / 1034 | [results/sft-hintstrained](results/sft-hintstrained) |
| 2026-11-25 | Qwen2.5-Coder-0.5B-Instruct + sft-synsql-hints | Same hints-trained model, vote over 15 candidates | 68.57 | 709 / 1034 | [results/sft-hintstrained-vote15](results/sft-hintstrained-vote15) |
| 2026-11-25 | vote of picks-sft-hintstrained-vote15, sft-rows1-vote15, sft-spider-all-vote5 | Three models vote (hints-trained with 15 candidates, one-row model with 15, all-Spider model with 5), empty results last | 73.31 | 758 / 1034 | [results/vote-hintstrained-three](results/vote-hintstrained-three) |

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

### 2026-11-21, sft-rows1-vote15

- 102 predictions did not run. The most common errors: `no such column` (95), `no such table` (7).
- 239 predictions ran and returned different rows.
- 44 of the 693 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 3,255 seconds.

### 2026-11-24, sft-synsql

- 198 predictions did not run. The most common errors: `no such column` (186), `no such table` (5), `misuse of aggregate` (3).
- 197 predictions ran and returned different rows.
- 38 of the 639 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 191 seconds.

### 2026-11-24, sft-synsql-hints

- 197 predictions did not run. The most common errors: `no such column` (183), `no such table` (8), `misuse of aggregate` (3).
- 168 predictions ran and returned different rows.
- 40 of the 669 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 187 seconds.

### 2026-11-24, sft-rows1-hints

- 220 predictions did not run. The most common errors: `no such column` (198), `no such table` (16), `a JOIN clause is required before ON` (1).
- 170 predictions ran and returned different rows.
- 40 of the 644 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 209 seconds.

### 2026-11-24, vote-hints-three

- 61 predictions did not run. The most common errors: `no such column` (59), `no such table` (2).
- 237 predictions ran and returned different rows.
- 44 of the 736 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.

### 2026-11-24, grpo600-hints

- 187 predictions did not run. The most common errors: `no such column` (174), `no such table` (7), `misuse of aggregate` (3).
- 189 predictions ran and returned different rows.
- 40 of the 658 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 187 seconds.

### 2026-11-24, grpo-spider600-hints

- 143 predictions did not run. The most common errors: `no such column` (125), `no such table` (11), `misuse of aggregate` (3).
- 220 predictions ran and returned different rows.
- 39 of the 671 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 192 seconds.

### 2026-11-25, grpo-lr5-2400-hints

- 143 predictions did not run. The most common errors: `no such column` (131), `misuse of aggregate` (5), `no such table` (4).
- 199 predictions ran and returned different rows.
- 38 of the 692 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 184 seconds.

### 2026-11-25, vote-hints-three-rows

- 61 predictions did not run. The most common errors: `no such column` (59), `no such table` (2).
- 222 predictions ran and returned different rows.
- 40 of the 751 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.

### 2026-11-25, sft-hintstrained

- 186 predictions did not run. The most common errors: `no such column` (175), `no such table` (9), `misuse of aggregate` (2).
- 202 predictions ran and returned different rows.
- 39 of the 646 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 187 seconds.

### 2026-11-25, sft-hintstrained-vote15

- 66 predictions did not run. The most common errors: `no such column` (66).
- 259 predictions ran and returned different rows.
- 40 of the 709 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 2,979 seconds.

### 2026-11-25, vote-hintstrained-three

- 42 predictions did not run. The most common errors: `no such column` (42).
- 234 predictions ran and returned different rows.
- 39 of the 758 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
