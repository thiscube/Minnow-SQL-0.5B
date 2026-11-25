# Findings

A timeline of what was learned while building Minnow-SQL-0.5B, newest first. An entry belongs here only if it changed a decision or would change one. Scores of individual runs are in [results.md](results.md).

Each entry has a kind:

- **Finding**: something measured about the model, the data or the evaluation.
- **Improvement**: a change that made results better or more trustworthy.
- **Decision**: a choice between options, with the reason.

## Timeline

### 2026-11-25 · Improvement · Reinforcement learning at a lower learning rate lifts the best single model to 66.92%

- **What:** GRPO from the SynSQL model on Spider training questions with value hints, learning rate 5e-6 instead of 2e-5, scored on dev every 800 questions.
- **Evidence:** 65.38% after 800 questions, 65.76% after 1,600 and 66.92% (692 of 1,034) after 2,400, against 64.70% at the start: 52 fixed, 29 broken. Queries that do not run fell from 197 to 143. Training takes 1.7 seconds per question and 4.6 GB.
- **So what:** Three rising checkpoints in a row make this the first single-model gain from reinforcement learning, at no cost beyond local GPU time. The run continues through all 6,994 Spider training questions, scored every 1,200. In the three-model vote this model gives 741 of 1,034 (71.66%) instead of 736, not yet logged.

### 2026-11-24 · Finding · Short reinforcement-learning pilots move many answers but not the score

- **What:** Two GRPO pilots of 600 questions each from the SynSQL model, with value hints: one on fresh SynSQL questions, one on Spider training questions. Learning rate 2e-5.
- **Evidence:** SynSQL pilot 63.64% (21 fixed, 32 broken), Spider pilot 64.89% (53 fixed, 51 broken), against 64.70% at the start. In the SynSQL pilot 348 of 600 questions were skipped because their prompt was over 900 tokens, so it learned from 252 questions in 58 updates. Its rising training reward was noise when split into blocks of 100 (0.47 to 0.63, no trend). A reviewer found no shift in SQL style, only resampled guesses on hard questions.
- **So what:** At this learning rate the model changes a quarter of its answers after about 60 updates without a direction. The next pilot uses a 4 times lower rate (5e-6), runs 2,400 questions and is scored every 800 to see a trend rather than one noisy point.

### 2026-11-24 · Improvement · Three models voting, one with value hints, pass 70%: 71.18%

- **What:** The same majority vote as before over three fine-tuned adapters on one base model: the SynSQL model with value hints (greedy), the example-row model (15 candidates) and the Spider-plus-others model (5 candidates).
- **Evidence:** 736 of 1,034, up from 710 for the earlier three-model vote: 50 fixed, 24 broken. Putting empty and all-NULL results last would give 751 (72.63%), but that rule was found by looking at dev and is not used until it is confirmed on the held-out test half.
- **So what:** The 70% goal is reached by several models voting, not by one model; the best single model is 64.70%. Which three runs to combine was also chosen on dev, so the final claim must be repeated on the untouched test half.

### 2026-11-24 · Improvement · Listing the values a question mentions adds 2.9 points without retraining

- **What:** Before the question, the prompt lists words from the question that appear as text values in the database, with the table and column that hold them, for example `country.Name = 'Angola'`. The model was never trained with this list.
- **Evidence:** The SynSQL model with one greedy query scores 64.70% (669 of 1,034) with the list and 61.80% (639) without: 38 questions fixed, 8 broken. On dev the list contains 283 of the 333 text values the reference queries use. The older model trained with example rows gains less: 62.28% against 61.51%.
- **So what:** This is the largest single gain since fine-tuning, and it costs nothing at answer time. The list stays on, and the reinforcement-learning run trains with it so the model learns to rely on it.

### 2026-11-24 · Finding · Adding 6,000 SynSQL databases to training gives a different model, not a clearly better one

- **What:** A fresh LoRA trained on Spider plus the SynSQL slice (12,215 examples), 2 epochs, 2 hours 2 minutes.
- **Evidence:** 61.80% (639 of 1,034) with one greedy query, against 61.51% for the best Spider-only model. Between the two, 99 questions were fixed and 96 broke. The final training loss was 0.11, against 0.03 for Spider alone, so the model memorises less. Queries naming a column that does not exist rose to 183.
- **So what:** More varied data did not raise the single-model score, but this model answers a different set of questions, which helps when models vote together. The extra "no such column" errors are what the reinforcement-learning reward targets.

### 2026-11-24 · Decision · Next steps chosen by a panel of three critics

- **What:** Three independent reviewers (research evidence, 8 GB feasibility, failure analysis) rated eleven ideas, and a second round attacked the chosen plan.
- **Evidence:** All three ranked better answer selection and reinforcement learning on execution results first. The published 0.5B result of 72.08% came mainly from reinforcement learning: the same paper's model reaches only 65.31% after supervised training on 916k examples. The failure analysis found that in 82 of the 84 questions the three-model vote misses while one model is right, only one model had it right. Chain-of-thought training, longer examples and a larger LoRA were rated lowest.
- **So what:** Every candidate query is now saved for later re-voting. Spider test is split by database into a half for choosing settings and a half scored once at the end. Reinforcement learning (GRPO) runs locally at 1.6 seconds per question and 4.5 GB, with rewards of 1 for the right rows, 0.1 for a query that runs, and advantages not divided by the group's spread.

### 2026-11-21 · Decision · Next training run mixes Spider with 6,000 databases of SynSQL

- **What:** A slice of SynSQL-2.5M is added to Spider's training questions: up to two questions from each of 6,000 synthetic databases, without the ones that need extra knowledge, the hardest level, or a query that does not run. The column comments in SynSQL's table definitions are removed.
- **Evidence:** The slice has 11,990 questions, 8,377 of them with a join. With the comments a schema is 1,580 tokens at the median and only 10% of the examples fit in 1,024 tokens. Without them the median is 1,015 and 52% fit, which gives 12,215 training examples together with Spider. The SLM-SQL paper reports 65.31% on Spider dev for the same base model after training on 916,156 SynSQL examples, with 16-candidate voting.
- **So what:** The model sees 43 times more databases than Spider alone has, and most new questions are joins, the mistake that hurts `car_1` and `world_1`. One update takes about 5.5 seconds instead of 2.5 because the examples are longer, so the run uses 2 epochs.

### 2026-11-21 · Finding · Fifteen candidates beat five by 1.3 points at three times the cost

- **What:** The best single model voting over 15 candidate queries per question instead of 5.
- **Evidence:** 67.02% (693 of 1,034) against 65.76%. 20 questions fixed, 7 broken. Queries that do not run fell from 127 to 102. Generation took 3,255 seconds.
- **So what:** The gain is just above the one-point noise and each extra candidate buys less. Published work uses 16 to 30 candidates, so 15 is kept for final scores only, and 5 for comparing training runs.

### 2026-11-21 · Improvement · Voting across three fine-tuned models reaches 68.67%

- **What:** The voted answers of the three fine-tuned models are combined with the same rule: run them, keep the result most of them agree on. That is 15 candidate queries per question in total.
- **Evidence:** 710 of 1,034, up from 680 for the best single model with voting. 52 questions fixed, 22 broken. Queries that do not run fell from 127 to 58. For 794 questions (76.79%) at least one of the three models has a correct answer.
- **So what:** This is the best score so far and a clear gain of 2.9 points, at three times the cost of answering a question. It is three adapters on one base model, not one model, which has to be said wherever the number is shown. The 76.79% is what a perfect way of choosing between the three could reach, so better selection is worth more than another similar training run.

### 2026-11-21 · Finding · One example row per table helps only slightly, and hurts the weakest database

- **What:** Retraining with one row of real data shown under each table in the prompt.
- **Evidence:** 61.51% with one greedy query (60.64% without rows) and 65.76% with five-candidate voting (64.31% without). `pets_1` rose from 60% to 76% and `wta_1` from 50% to 60%, but `car_1`, the database the change was aimed at, fell from 20% to 14%.
- **So what:** The gain of about one point is at the edge of the noise, and the run trained on 5,259 examples instead of 6,142 because the prompts are longer. Example rows do not fix the model putting columns on the wrong table. That mistake needs training data with more multi-table joins.

### 2026-11-21 · Decision · Next run shows the model one example row per table

- **What:** The schema in the prompt gets a comment after each table with the column names and one row of real data, and the model is retrained with that prompt.
- **Evidence:** The two weakest large databases fail in ways example rows should help. In `car_1` (25%, 92 questions) the model puts columns on the wrong table, such as horsepower on `car_names` when it lives in `cars_data`. In `world_1` (41%, 120 questions) it confuses what a column holds, such as filtering `Code = "Angola"` where the name is in `Name`. All dev schemas are at most 838 tokens, so leaving long examples out of training is not what hurts.
- **So what:** Two rows per table would be better but make training 3.5 times slower on an 8 GB card (8.8 against 2.7 seconds per update). One row keeps the speed, at the cost of training on 5,259 examples instead of 6,142.

### 2026-11-21 · Finding · 1,659 more Spider training examples do not clearly help

- **What:** Training on `train_spider.json` plus `train_others.json` (7,800 examples) instead of `train_spider.json` alone (6,142).
- **Evidence:** With one greedy query 59.77% against 60.64%. With five-candidate voting 65.09% against 64.31%. Both differences are inside the one-point noise. Between the two models 73 questions were fixed and 82 broke with greedy decoding, so about 15% of the questions flip from one training run to the next.
- **So what:** More data of the same kind is not the lever. The large churn between two similar models means single runs have to be compared with care, and that combining models could help later.

### 2026-11-21 · Improvement · Voting over five candidate queries adds 3.7 points without training

- **What:** For each question the fine-tuned model writes one greedy query and four sampled ones (temperature 0.8). Each is run on the database, and the result most candidates agree on is kept. No reference query is used.
- **Evidence:** 64.31% (665 of 1,034), up from 60.64% for the same model with one greedy query. 42 questions fixed, 4 broken. Queries that do not run fell from 208 to 130. Generation took 979 seconds instead of about 300.
- **So what:** The gain is well above the one-point noise, so voting stays on for the best score. It mostly rescues queries that crash. Queries that run but return different rows rose from 199 to 239, so wrong-but-runnable answers are now the largest group and need better training data, not more voting.

### 2026-11-21 · Finding · Fine-tuning on Spider lifts the score from 45.36% to 60.64%

- **What:** LoRA fine-tuning on 6,142 Spider training examples for 3 epochs, 47 minutes on one RTX 3060 Ti with 5.1 GB of GPU memory.
- **Evidence:** 627 of 1,034 correct, up from 469. Against the baseline, 221 questions were fixed and 63 broke. Queries naming a column that does not exist fell from 356 to 189. Queries that run but return different rows rose from 176 to 199. The training loss fell from 0.38 to 0.03.
- **So what:** Training fixes about half of the column mistakes, and 208 answers still do not run at all. The weakest databases are `car_1` (20%), `real_estate_properties` (25%) and `world_1` (39%), and `pets_1` got worse (55% to 45%). A loss of 0.03 on the training set with 60% on dev suggests the model is memorising, so more varied data should help more than more epochs.

### 2026-11-21 · Decision · First training run leaves out the longest examples

- **What:** Fine-tune with LoRA on Spider's training questions that are at most 1,024 tokens long, one example at a time with 16 examples per weight update.
- **Evidence:** Memory grows with the square of an example's length. With examples up to 1,536 tokens the run needed 8.0 GB and took 8.7 seconds per update. With a limit of 1,024 tokens it needs 5.1 GB and 2.5 seconds per update. The limit keeps 6,142 of 6,994 examples.
- **So what:** The first run takes about 50 minutes instead of 3 hours and leaves room on an 8 GB card. The 852 long examples are the ones with the largest schemas, so a later run should test whether leaving them out costs accuracy on large databases.

### 2026-11-21 · Improvement · Training loss is computed on the answer only

- **What:** The loss is calculated from the positions of the SQL answer, not from every position of the prompt.
- **Evidence:** The standard loss scores all positions against the 151,936-token vocabulary and then ignores the prompt, which costs gigabytes on a long schema. Both ways give the same loss on real examples (0.2450 and 0.2450).
- **So what:** Same training signal, far less memory. Without this the run did not fit in 8 GB at all.

### 2026-11-21 · Finding · Six training questions are worded exactly like dev questions

- **What:** The leak guard stopped the first training attempt.
- **Evidence:** Six generic questions, such as "How many employees are there?" and "How many flights do we have?", appear in both Spider's training set and its dev set, on different databases.
- **So what:** The six are left out of training, which leaves 6,994 of 7,000 examples. No training example uses a dev database.

### 2026-11-21 · Decision · First training run uses Spider's training questions only

- **What:** Stage 2 trains on `train_spider.json` and nothing else.
- **Evidence:** The set has 7,000 examples over 140 databases, and none of those databases is in the dev set. Examples are 381 tokens long at the median, 1,104 at the 90th percentile and 2,324 at most.
- **So what:** This gives a clean first comparison with the baseline before any synthetic data is added.

### 2026-11-21 · Finding · One mistake causes a third of all failures

- **What:** 356 of the 1,034 baseline answers do not run because they name a column that does not exist.
- **Evidence:** Baseline run: 469 correct, 176 ran with different rows, 389 did not run. Of the 389, 356 are `no such column` and 15 are `no such table`.
- **So what:** Fixing this one mistake is worth up to 34 points, which would put the ceiling near 80%. It is the first thing training has to fix.

### 2026-11-21 · Finding · Accuracy differs widely between databases

- **What:** The baseline scores between 13% and 87% depending on the database.
- **Evidence:** Per-database accuracy over the 20 dev databases, shown in the dashboard. Best: `employee_hire_evaluation` at 87%. Worst: `course_teach` at 13%.
- **So what:** An average hides weak topics. Later runs are compared per database as well as overall.

### 2026-11-21 · Finding · About one point of score is noise

- **What:** The answers depend slightly on how many questions are generated at once.
- **Evidence:** Redoing every fifth question one at a time changed the answer text for 21 of 200 questions and the verdict for 3 of 200 (88 correct with batch size 16, 87 one at a time).
- **So what:** Every scored run uses batch size 16 in fixed blocks. A difference of about one point between two runs is not counted as an improvement.

### 2026-11-21 · Finding · The baseline figure is slightly optimistic

- **What:** A wrong query can be counted as correct when both it and the reference return no rows.
- **Evidence:** 30 of the 469 correct baseline answers match a reference query that returns nothing. A manual read of those 30 found at least 16 that are plainly wrong queries.
- **So what:** The honest baseline is nearer 44% than 45.36%. The official single-database evaluation has the same weakness, so numbers stay comparable. The count is reported with every run.

### 2026-11-21 · Finding · The untouched base model scores 45.36%

- **What:** Qwen2.5-Coder-0.5B-Instruct, no training, greedy decoding, schema given as `CREATE TABLE` statements.
- **Evidence:** 469 of 1,034 on Spider dev, generated in 291 seconds on one RTX 3060 Ti. Scoring the reference queries against themselves gives 1,034 of 1,034, and an independent re-score gave the same 469.
- **So what:** This is the number every later run has to beat. It is close to the 44.07% the SLM-SQL paper reports for the same model.

### 2026-11-21 · Improvement · Every reported number is generated from the runs' files

- **What:** The results page, the chart and the README table are written by a tool, and a check refuses anything edited by hand.
- **Evidence:** Nine result files were tampered with one at a time (scores, a swapped answer, a flipped verdict, a removed verdict, an edited model name). All nine were refused.
- **So what:** A wrong or outdated score can no longer reach the README by accident.

### 2026-11-21 · Finding · Two reference numbers in the original plan were wrong

- **What:** The best published 0.5B result on Spider dev is 72.08%, not 73.5%, and the base model's published starting point is 44.07%, not 34.5%.
- **Evidence:** In the SLM-SQL paper, 73.50% is the Spider test score and 72.08% the dev score. The 34.5% figure comes from a model card that used the non-instruct base model.
- **So what:** The stretch goal is 72.08% on dev. Numbers quoted from other work are checked at the source before they are used.
