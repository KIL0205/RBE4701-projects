# Your goal #

In this scenario, you must plan the route of your agent from the top-left
corner to the exit. However, your route is obstructed - you need to use the bomb
to create a path to the exit.

## Variant 1: Alone in the world ##

In the first variant of this scenario, the world is deterministic and your agent
is alone in the environment.

## Variant 2: Random monster ##

In the second variant of this scenario, a stupid monster is present. The monster
chooses its next cell uniformly at random among the possible reachable cells.

## Variant 3: Self-preserving monster ##

In the third variant of this scenario, a smarter monster is present:
- The monster goes straight until it has reached an obstacle
- When it reaches an obstacle, it changes direction at random among the cells
  that are walkable and are not an explosion (if an agent, monster or character,
  touches an explosion, it dies)
- If the 8-distance of your agent to the monster is 1, the monster attacks your
  agent immediately and kills it

## Variant 4: Aggressive monster ##

In the fourth variant of this scenario, an aggressive monster is present:
- The monster goes straight until it has reached an obstacle
- When it reaches an obstacle, it changes direction at random among the cells
  that are walkable and are not an explosion (if an agent, monster or character,
  touches an explosion, it dies)
- If the 8-distance of your agent to the monster is 2, the monster moves towards
  your agent and attempts to kill it

## Variant 5: Stupid and Aggressive monsters together ##

In the fifth variant of this scenario, two monsters are present: an aggressive
one and a stupid one.

## Project 2 training and evaluation

### Train
in project 2 run:
```text
python training.py --curriculum drills --workers 20 --guis 1 --trials 20 --survive 10 --eval-trials 20 --drill-refresh-after 100 --drill-refresh-trials 5
```
or 
```text
python training.py -h
```
which will show all of the command line configurables with discribtions

Use `--guis 0` for headless training. Individual worker reports are off by
default; add `--worker-details` when debugging. Drill maps are in
`team01/project2/drills/`, and the default checkpoint is
`team01/project2/q_learning_weights.json`. Training writes a diagnostic
`training_history.jsonl`; this history is not needed to load weights or resume
the curriculum.

### Grade the saved policy

in eval run 
```text
python project2_grading_eval.py --runs 50 --no-display
```

Use `--runs 2 --no-display` for a quick test. The evaluator loads the
included `team01/project2/q_learning_weights.json` checkpoint by default and
writes its report under `team01/eval/results/`, creating that directory when
needed. To show game windows, omit `--no-display` and run.