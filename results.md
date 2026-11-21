# Results

Execution accuracy (EX) on the Spider development set, 1,034 questions. A prediction is correct when it runs and returns the same rows as the reference query. Scoring is done by [`minnow/execution.py`](minnow/execution.py) on the original Spider databases.

| Date | Model | Change | Dev EX (%) | Correct | Files |
| --- | --- | --- | ---: | ---: | --- |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct | Baseline: no training, greedy decoding, `CREATE TABLE` statements as the schema | 45.36 | 469 / 1034 | [results/baseline](results/baseline) |

## Notes

### 2026-11-21, baseline

- 389 predictions did not run. 356 of them named a column that does not exist, and 15 a table that does not exist.
- 176 predictions ran and returned different rows.
- 30 of the 469 correct predictions matched a reference query that returns no rows. With a single database per question, such a match cannot be told apart from a wrong query that also returns nothing.
- Generating all 1,034 predictions took 291 seconds on one RTX 3060 Ti with batch size 16, including the model download.
- The score depends slightly on the batch size. Redoing every fifth question one at a time changed the answer text for 21 of 200 questions and the verdict for 3 of 200 (88 correct with batch size 16, 87 one at a time). Scored runs therefore always use batch size 16 in fixed blocks, and differences of about one point between runs are within this noise.
