# Runs on Spider test questions

The run files behind the README numbers that are not on Spider dev: for every question the query that was picked, the verdict, and the score.

Spider's test set was cut in two by database with `python -m minnow.splits data/spider`:

- `validation/`: the half used to choose settings such as the voting rule and the line-up (1,230 questions, 21 databases).
- `final/`: the other half (917 questions, 19 databases). Nothing was chosen on it. It was scored once, after everything else was fixed.

Each folder is a run folder like the ones in `results/`. To judge one again:

```bash
python -m minnow.outcomes heldout/final/<run> --spider data/spider --split test_final.json
```

`standing.json` lists which of these runs fill the score table at the top of the main README. `python -m minnow.runlog render` writes that table from the runs' files, and `python -m minnow.runlog check` fails when the table and the files disagree.
