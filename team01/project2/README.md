# Project 2 use

The are three agents are `q` (Q-learning),
`qbt` (Behavior Tree with Q-learning action nodes), and `dqn` (Deep Q-Network).
QBT is the default for training, grading, and standalone variants. 

Each agent uses a separate learned-state file by default:

- `q`: `team01/project2/q_learning_weights.json`
- `qbt`: `team01/project2/q_learning_bt_weights.json`
- `dqn`: `team01/project2/dqn_checkpoint.pt`

## Training

```powershell
python training.py --trials 10 --workers 4
python training.py --agent q --trials 10
python training.py --agent dqn --trials 10 --workers 4
```

`--trials` sets episodes per training batch; `--workers` limits concurrent
rollouts. Existing learned state is loaded and continued by default. Use
`--fresh` to start from new state, `--weights PATH` to choose a state file, or
`--start-variant 4` to resume training with V1–V4 unlocked and V4 as the current
stage. DQN-only controls include `--dqn-learning-rate`, `--dqn-epsilon`, and
`--dqn-updates-per-batch`.

## Grading Evaluation

```powershell
python project2_grading_eval.py --runs 10 --workers 4 --no-display
python project2_grading_eval.py --agent dqn --runs 10 --workers 4 --no-display
```

`--agent` accepts `q`, `qbt`, or `dqn` and defaults to `qbt`. `--runs` is the
number of trials **per variant**; `--workers` limits concurrent workers and
does not increase the number of trials. Grading uses a frozen policy. Use
`--weights PATH` to select a learned-state file. 

The program will automatically default to the appropriate weights based on the agent

## Individual variants

Run `variant1` through `variant5`; for example:

```powershell
python variant1.py
python variant4.py --agent qbt --seed 12345
python variant3.py --agent q --seed 12345
python variant4.py --agent dqn --seed 12345
```

Each runner accepts `--agent {q,qbt,dqn}`, `--weights PATH`, `--seed INT`,
`--display`, and `--no-display`. An explicit seed makes the scenario
reproducible and is independent of agent choice. If omitted, a random seed from
0 through 1000 is generated and printed. Standalone variants use frozen agents
and do not save learned state.


## For Grading

We would like you to use the defaults

```powershell
python variant1.py
python variant2.py
python variant3.py
python variant4.py
python variant5.py
```

Both QAgent and QBTAgent are trained past the full grading threshold.
DeepQAgent either still needs more trainging or needs some tweaks for full functionality