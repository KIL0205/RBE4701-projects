# Project 2: DQN training and grading
Used AI to Generate this README 

## Architecture

The DQN scores each legal action from an action-conditioned vector of 17
features: QAgent's 16 shared features plus `direct_objective_progress`. A
128-128 ReLU policy network estimates Q(s, a); a separate frozen target network
provides bootstrap values. Parallel CPU workers receive immutable policy
snapshots and return transitions. The parent is the canonical learner: it owns
replay, Adam updates, target synchronization, curriculum decisions, and
checkpoint writes.

## Setup

From the repository root, install runtime and development dependencies:

Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

The runtime dependencies are `pygame`, `colorama`, and PyTorch;

## Train

```powershell
python -m team01.project2.training --agent dqn --curriculum drills --workers 10 --trials 20 --survive 15 --eval-trials 20 --weights team01/project2/dqn_checkpoint.pt --guis 0
```

The DQN checkpoint defaults to `team01/project2/dqn_checkpoint.pt`. If it
exists, omitting `--fresh` loads its policy, target network, Adam state, and
counters; replay starts empty and refills before updates resume. `--fresh`
starts new networks and does not overwrite an existing checkpoint until a
valid batch completes. `--trials` sets episodes per training batch; `--workers`
sets maximum concurrent rollouts. The curriculum's `survive / trials`
threshold is scaled to the newest variant's share of each mixed batch, so
`--survive 15 --trials 20` sets a 75% stage threshold. This batch progression
is distinct from final frozen evaluation, which requires at least 90% wins on
every variant. Stagnation can trigger drill refreshes, but drills do not unlock
variants.

## Resume at a variant

```powershell
python -m team01.project2.training --agent dqn --start-variant 4 --workers 10 --trials 20 --survive 15 --weights team01/project2/dqn_checkpoint.pt
```

`--start-variant 4` makes V1-V4 available and makes V4 the current newest
training stage. V4 has **not** passed its stage threshold, and V5 remains
locked. The checkpoint stores learner state, not curriculum position; use
`--start-variant` to select the variant stage when resuming.

## DQN overrides

Use `--dqn-learning-rate` and `--dqn-epsilon` to override the run's learning
rate and exploration rate. The learning-rate override is applied after loading
the checkpoint because Adam restores its saved parameter-group settings; it
changes the rate without resetting learned network parameters or Adam moments.

```powershell
python -m team01.project2.training --agent dqn --dqn-learning-rate 0.00005 --dqn-epsilon 0.05
```

## Parallel grader

```powershell
python -m team01.eval.project2_grading_eval --agent dqn --weights team01/project2/dqn_checkpoint.pt --runs 20 --workers 10 --no-display
```

`--runs` is the number of episodes **per variant** (100 total for five
variants); `--workers` limits concurrent worker processes and does not change
the episode count. The parent assigns each episode a unique seed before
dispatch, and results are restored to that planned order before scoring.
Workers evaluate frozen policy copies and never train or save checkpoints.
Use `--agent q` to grade the linear QAgent with
`team01/project2/q_learning_weights.json`.

## Checkpoints and tests

DQN checkpoints atomically save policy and target parameters, Adam state,
training counters, and feature-schema metadata. Replay is intentionally not
saved. Checkpoints, `runs/`, training logs, and history files are generated
local state and normally should not be committed.
01/tests/test_project2_grading_parallel.py
```
