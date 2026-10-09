"""Estimate Project 2 grading performance with fresh randomized trials."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import random
import secrets
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from team01.agent.deep_q_learning import DeepQAgent
    from team01.agent.q_learning import QBTAgent

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
else:
    _ROOT = Path(__file__).resolve().parents[2]

_BOMBERMAN = _ROOT / "Bomberman"
if str(_BOMBERMAN) not in sys.path:
    sys.path.insert(0, str(_BOMBERMAN))

from team01.project2.agent_backends import (
    AGENT_CHOICES,
    AGENT_LABELS,
    DEFAULT_AGENT,
    DEFAULT_AGENT_STATE_PATHS,
    DEFAULT_QBT_WEIGHTS_PATH,
    DEFAULT_Q_WEIGHTS_PATH,
    Q_LEARNING_AGENTS,
)

RUNS_PER_VARIANT = 50
DISPLAY = True
SOLVED_THRESHOLD = 0.50
VARIANT_POINTS = {
    1: 10,
    2: 25,
    3: 25,
    4: 25,
    5: 35,
}
DEFAULT_MAP_PATH = _ROOT / "team01" / "project2" / "map.txt"
DEFAULT_WEIGHTS_PATH = DEFAULT_Q_WEIGHTS_PATH
RESULT_PATH = Path(__file__).resolve().parent / "results" / "project2_grading_eval.json"
SPRITE_DIRECTORY = str(_BOMBERMAN / "sprites") + os.sep

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from Bomberman.game import Game
from Bomberman.monsters.selfpreserving_monster import SelfPreservingMonster
from Bomberman.monsters.stupid_monster import StupidMonster
from team01.agent.q_learning import QAgent
from team01.eval.train_q_learning import EpisodeResult, run_episode


@dataclass(frozen=True)
class MonsterSpec:
    kind: str
    name: str
    avatar: str
    x: int
    y: int
    detection_range: Optional[int] = None


@dataclass(frozen=True)
class VariantSpec:
    number: int
    name: str
    monsters: tuple[MonsterSpec, ...]


VARIANT_SPECS = (
    VariantSpec(1, "Alone in the world", ()),
    VariantSpec(2, "Random monster", (MonsterSpec("stupid", "stupid", "S", 3, 9),)),
    VariantSpec(
        3,
        "Self-preserving monster",
        (MonsterSpec("smart", "selfpreserving", "S", 3, 9, 1),),
    ),
    VariantSpec(
        4,
        "Aggressive monster",
        (MonsterSpec("smart", "aggressive", "A", 3, 13, 2),),
    ),
    VariantSpec(
        5,
        "Stupid and aggressive monsters",
        (
            MonsterSpec("stupid", "stupid", "S", 3, 5),
            MonsterSpec("smart", "aggressive", "A", 3, 13, 2),
        ),
    ),
)
VARIANT_BY_NUMBER = {variant.number: variant for variant in VARIANT_SPECS}


@dataclass(frozen=True)
class GradingTrialResult:
    """One independent Project 2 grading trial."""

    variant: int
    seed: int
    success: bool
    reason: str
    ticks: int
    reward: float
    bombs: int
    max_abs_q: float
    td_update_count: int


@dataclass(frozen=True)
class GradingTrialTask:
    """Pickle-safe identity and seed for one parent-planned grading episode."""

    trial_index: int
    variant: int
    run_number: int
    seed: int


class GradingAborted(Exception):
    """Raised when the visible grading window is closed."""


@dataclass(frozen=True)
class GradingBackend:
    """Loaded canonical policy and a factory for clean trial agents."""

    agent_type: str
    checkpoint_path: Path
    weights: Optional[dict[str, float]]
    agent_factory: Callable[[], QAgent | QBTAgent | DeepQAgent]
    canonical_snapshot: Optional[dict[str, Any]] = None
    canonical_agent: Optional[DeepQAgent] = None


_WORKER_AGENT: Optional[QAgent | QBTAgent | DeepQAgent] = None
_WORKER_WEIGHTS: Optional[dict[str, float]] = None
_WORKER_TRIAL_RUNNER: Optional[Callable[..., GradingTrialResult]] = None
_WORKER_DISPLAY = False
_WORKER_AGENT_TYPE: Optional[str] = None


def configure_runtime(display: bool) -> None:
    """Select the existing real or dummy Bomberman display backend."""
    os.environ["BOMBERMAN_DISPLAY"] = "1" if display else "0"
    if display:
        os.environ.pop("SDL_VIDEODRIVER", None)
    else:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


def load_frozen_agent(weights_path: Path | str = DEFAULT_WEIGHTS_PATH) -> QAgent:
    """Load the saved learned policy and verify learning cannot run."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Project 2 learned weights not found: {path}. Train or provide --weights; "
            "grading evaluation will not fall back to untrained weights."
        )
    agent = QAgent("me", "C", 0, 0)
    agent.load_weights(str(path))
    agent.set_learning(no_training=True, epsilon=0.0)
    if agent.training or agent.epsilon != 0.0:
        raise RuntimeError("Project 2 grading agent did not enter frozen-policy mode")
    return agent


def load_frozen_weights(weights_path: Path | str = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
    """Return the exact canonical weights loaded from the learned checkpoint."""
    return dict(load_frozen_agent(weights_path).weights)


def load_frozen_qbt_agent(weights_path: Path | str = DEFAULT_QBT_WEIGHTS_PATH) -> QBTAgent:
    """Load the Q-learning Behavior Tree backend without enabling learning."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"QBT learned weights not found: {path}. Train QBT or provide --weights; "
            "grading evaluation will not fall back to untrained weights."
        )
    from team01.agent.q_learning import QBTAgent

    agent = QBTAgent("me", "C", 0, 0)
    agent.load_weights(str(path))
    agent.set_learning(no_training=True, epsilon=0.0)
    if agent.training or agent.epsilon != 0.0:
        raise RuntimeError("Project 2 QBT agent did not enter frozen-policy mode")
    return agent


def _default_checkpoint_path(agent_type: str) -> Path:
    try:
        return DEFAULT_AGENT_STATE_PATHS[agent_type]
    except KeyError as error:
        raise ValueError(f"unknown Project 2 agent type: {agent_type}") from error


def _assert_frozen_agent(agent: QAgent | QBTAgent | DeepQAgent) -> None:
    if agent.training or agent.epsilon != 0.0:
        raise RuntimeError("Project 2 grading trial did not start with a frozen policy")
    if isinstance(agent, QAgent):
        return
    from team01.agent.deep_q_learning import DeepQAgent

    if not isinstance(agent, DeepQAgent):
        raise TypeError(f"unsupported Project 2 grading agent: {type(agent).__name__}")
    if agent.learning_enabled or agent.collect_experience:
        raise RuntimeError("DQN grading agent must disable learning and experience collection")
    if agent.policy_network.training:
        raise RuntimeError("DQN grading policy network must be in evaluation mode")
    if len(agent.replay_buffer) != 0:
        raise RuntimeError("DQN grading agent must start with an empty replay buffer")


def _frozen_q_agent_factory(
    agent_type: str,
    weights: dict[str, float],
) -> Callable[[], QAgent | QBTAgent]:
    from team01.agent.q_learning import QBTAgent

    if agent_type not in Q_LEARNING_AGENTS:
        raise ValueError(f"unknown Q-learning backend: {agent_type}")
    agent_class = QBTAgent if agent_type == "qbt" else QAgent
    canonical_weights = dict(weights)

    def make_agent() -> QAgent | QBTAgent:
        agent = agent_class("me", "C", 0, 0)
        agent.weights = dict(canonical_weights)
        agent.set_learning(no_training=True, epsilon=0.0)
        return agent

    return make_agent


def _load_grading_agent(agent_type: str, checkpoint_path: Path) -> GradingBackend:
    """Load one canonical policy and provide fresh frozen agents for trials."""
    if agent_type in Q_LEARNING_AGENTS:
        load_agent = load_frozen_qbt_agent if agent_type == "qbt" else load_frozen_agent
        try:
            canonical_agent = load_agent(checkpoint_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                f"{AGENT_LABELS[agent_type]} weights are not valid JSON: "
                f"{checkpoint_path}. "
                "Use --agent dqn for a DQN checkpoint."
            ) from error
        weights = dict(canonical_agent.weights)

        return GradingBackend(
            agent_type=agent_type,
            checkpoint_path=checkpoint_path,
            weights=weights,
            agent_factory=_frozen_q_agent_factory(agent_type, weights),
        )

    if agent_type == "dqn":
        from team01.agent.deep_q_learning import DeepQAgent

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {checkpoint_path}. "
                "Train the DQN first or provide --weights <path>."
            )
        canonical_agent = DeepQAgent("me", "C", 0, 0)
        canonical_agent.load_checkpoint(checkpoint_path)
        canonical_agent.set_learning(no_training=True, epsilon=0.0)
        _assert_frozen_agent(canonical_agent)
        canonical_snapshot = canonical_agent.get_checkpoint_snapshot()

        def make_dqn_agent() -> DeepQAgent:
            agent = copy.deepcopy(canonical_agent)
            agent.reset_episode()
            agent.set_learning(no_training=True, epsilon=0.0)
            return agent

        return GradingBackend(
            agent_type="dqn",
            checkpoint_path=checkpoint_path,
            weights=None,
            agent_factory=make_dqn_agent,
            canonical_snapshot=canonical_snapshot,
            canonical_agent=canonical_agent,
        )

    raise ValueError(f"unknown Project 2 agent type: {agent_type}")


def generate_seed(used_seeds: set[int]) -> int:
    """Generate a fresh valid trial seed not used in this evaluation."""
    while True:
        seed = secrets.randbits(32)
        if seed not in used_seeds:
            used_seeds.add(seed)
            return seed


def is_solved(wins: int, runs: int = RUNS_PER_VARIANT) -> bool:
    """Return whether a variant meets the grading threshold, inclusive."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    return wins / runs >= SOLVED_THRESHOLD


def _build_trial_game(
    variant: int,
    seed: int,
    weights: Optional[dict[str, float]],
    agent_factory: Optional[Callable[[], QAgent | QBTAgent | DeepQAgent]] = None,
) -> tuple[Game, QAgent | QBTAgent | DeepQAgent]:
    """Construct one clean Project 2 game and one clean frozen agent."""
    if variant not in VARIANT_BY_NUMBER:
        raise ValueError("Project 2 grading supports variants 1 through 5")
    spec = VARIANT_BY_NUMBER[variant]
    random.seed(seed)
    game = Game.fromfile(str(DEFAULT_MAP_PATH), sprite_dir=SPRITE_DIRECTORY)
    for monster in spec.monsters:
        if monster.kind == "stupid":
            game.add_monster(
                StupidMonster(monster.name, monster.avatar, monster.x, monster.y)
            )
        elif monster.kind == "smart":
            game.add_monster(
                SelfPreservingMonster(
                    monster.name,
                    monster.avatar,
                    monster.x,
                    monster.y,
                    monster.detection_range,
                )
            )
        else:
            raise ValueError(f"Unknown Project 2 monster kind: {monster.kind}")

    if agent_factory is None:
        if weights is None:
            raise ValueError("trial construction requires weights or an agent factory")
        agent = QAgent("me", "C", 0, 0)
        agent.weights = dict(weights)
        agent.set_learning(no_training=True, epsilon=0.0)
    else:
        agent = agent_factory()
        _assert_frozen_agent(agent)
    game.add_character(agent)
    if not any(agent in characters for characters in game.world.characters.values()):
        raise RuntimeError("Project 2 grading agent was not registered in its world")
    return game, agent


def _display_callback(game: Game, label: str) -> Callable[[int], None]:
    """Render one grading game through the existing Bomberman display."""
    import pygame

    pygame.display.set_caption(f"Project 2 Grading | {label}")
    game.display_gui()

    def on_tick(_tick: int) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise GradingAborted("grading window closed")
        game.display_gui()
        pygame.time.wait(50)

    return on_tick


def run_trial(
    variant: int,
    *,
    seed: int,
    weights: Optional[dict[str, float]] = None,
    display: bool = False,
    game_builder: Optional[
        Callable[..., tuple[Any, QAgent | QBTAgent | DeepQAgent]]
    ] = None,
    episode_runner: Optional[Callable[..., EpisodeResult]] = None,
    agent_factory: Optional[Callable[[], QAgent | QBTAgent | DeepQAgent]] = None,
) -> GradingTrialResult:
    """Run one fresh-seed Project 2 game with a clean frozen agent."""
    if weights is None and agent_factory is None:
        raise ValueError("run_trial requires loaded learned weights")
    canonical_weights = dict(weights) if weights is not None else None
    builder = game_builder or _build_trial_game
    runner = episode_runner or run_episode
    if agent_factory is None:
        game, agent = builder(variant, seed, canonical_weights)
    else:
        game, agent = builder(
            variant,
            seed,
            canonical_weights,
            agent_factory=agent_factory,
        )
    _assert_frozen_agent(agent)
    if isinstance(agent, QAgent):
        before_snapshot = {
            "weights": dict(agent.weights),
            "td_update_count": agent.td_update_count,
        }
        replay_size = None
    else:
        from team01.agent.deep_q_learning import DeepQAgent

        if not isinstance(agent, DeepQAgent):
            raise TypeError(f"unsupported Project 2 grading agent: {type(agent).__name__}")
        before_snapshot = agent.get_checkpoint_snapshot()
        replay_size = len(agent.replay_buffer)
    label = f"Variant {variant} | Seed {seed}"
    on_tick = _display_callback(game, label) if display else None
    try:
        episode = runner(
            game.world,
            agent,
            1,
            progress_label=None,
            heartbeat_interval=0,
            on_tick=on_tick,
        )
    finally:
        try:
            import pygame

            pygame.quit()
        except Exception:
            pass

    if getattr(agent, "td_update_count", 0) != 0 or episode.td_update_count != 0:
        raise RuntimeError("Project 2 grading trial unexpectedly performed TD updates")
    if isinstance(agent, QAgent):
        if (
            agent.weights != before_snapshot["weights"]
            or (weights is not None and weights != canonical_weights)
        ):
            raise RuntimeError("Project 2 grading trial modified learned weights")
    else:
        from team01.agent.deep_q_learning import (
            DeepQAgent,
            checkpoint_snapshots_equal,
        )

        if not isinstance(agent, DeepQAgent):
            raise TypeError(f"unsupported Project 2 grading agent: {type(agent).__name__}")
        if len(agent.replay_buffer) != replay_size:
            raise RuntimeError("Project 2 DQN grading trial modified replay")
        agent.episodes = before_snapshot["episodes"]
        after_snapshot = agent.get_checkpoint_snapshot()
        if not checkpoint_snapshots_equal(before_snapshot, after_snapshot):
            raise RuntimeError("Project 2 DQN grading trial modified checkpoint state")
    return GradingTrialResult(
        variant=variant,
        seed=seed,
        success=episode.outcome == "WON",
        reason=episode.outcome,
        ticks=episode.ticks,
        reward=episode.total_reward,
        bombs=episode.bombs_placed,
        max_abs_q=episode.max_abs_q,
        td_update_count=episode.td_update_count,
    )


def evaluate_trial(
    variant: int,
    run_number: int,
    result: GradingTrialResult,
) -> dict[str, Any]:
    return {
        "run": run_number,
        "variant": variant,
        "seed": result.seed,
        "success": result.success,
        "reason": result.reason,
        "ticks": result.ticks,
        "reward": result.reward,
        "bombs": result.bombs,
        "max_abs_q": result.max_abs_q,
        "td_update_count": result.td_update_count,
    }


def evaluate_variant(
    variant: int,
    runs: int,
    used_seeds: set[int],
    weights: Optional[dict[str, float]],
    display: bool,
    trial_runner: Callable[..., GradingTrialResult] = run_trial,
    agent_factory: Optional[Callable[[], QAgent | QBTAgent | DeepQAgent]] = None,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for run_number in range(1, runs + 1):
        seed = generate_seed(used_seeds)
        result = trial_runner(
            variant,
            seed=seed,
            weights=weights,
            display=display,
            agent_factory=agent_factory,
        )
        trial = evaluate_trial(variant, run_number, result)
        trials.append(trial)
        print(
            f"Run {run_number:2d}/{runs} | V{variant} | seed={seed} | "
            f"{result.reason} | ticks={result.ticks}"
        )

    wins = sum(trial["success"] for trial in trials)
    rate = wins / runs
    failures = Counter(trial["reason"] for trial in trials if not trial["success"])
    solved = is_solved(wins, runs)
    points_available = VARIANT_POINTS[variant]
    summary: dict[str, Any] = {
        "points_available": points_available,
        "wins": wins,
        "runs": runs,
        "success_rate": rate,
        "solved": solved,
        "points_earned": points_available if solved else 0,
        "failure_reasons": dict(sorted(failures.items())),
        "trials": trials,
    }
    failures_only = [trial for trial in trials if not trial["success"]]
    if failures_only:
        first_failure = failures_only[0]
        summary["first_failure"] = {
            "run": first_failure["run"],
            "seed": first_failure["seed"],
            "reason": first_failure["reason"],
            "ticks": first_failure["ticks"],
        }
    return summary


def build_trial_plan(
    runs: int,
    used_seeds: set[int],
    *,
    seed_generator: Callable[[set[int]], int] = generate_seed,
) -> list[GradingTrialTask]:
    """Assign every grading trial its identity and seed in the parent process."""
    if runs < 1:
        raise ValueError("runs per variant must be at least 1")
    tasks = []
    for variant in VARIANT_POINTS:
        for run_number in range(1, runs + 1):
            tasks.append(
                GradingTrialTask(
                    trial_index=len(tasks),
                    variant=variant,
                    run_number=run_number,
                    seed=seed_generator(used_seeds),
                )
            )
    return tasks


def order_trial_results(
    tasks: list[GradingTrialTask],
    results: dict[int, GradingTrialResult],
) -> list[GradingTrialResult]:
    """Restore parent trial order after workers complete in arbitrary order."""
    expected = {task.trial_index for task in tasks}
    if set(results) != expected:
        raise ValueError("grading results do not match the planned trial indices")
    return [results[task.trial_index] for task in tasks]


def _initialize_grading_worker(
    agent_type: str,
    weights: Optional[dict[str, float]],
    dqn_policy_state: Optional[dict[str, Any]],
    display: bool,
    trial_runner: Callable[..., GradingTrialResult],
) -> None:
    """Install one immutable per-process policy snapshot under spawn."""
    global _WORKER_AGENT, _WORKER_WEIGHTS, _WORKER_TRIAL_RUNNER, _WORKER_DISPLAY
    global _WORKER_AGENT_TYPE
    configure_runtime(display)
    _WORKER_TRIAL_RUNNER = trial_runner
    _WORKER_DISPLAY = display
    _WORKER_AGENT_TYPE = agent_type
    if agent_type in Q_LEARNING_AGENTS:
        if weights is None:
            raise ValueError(
                f"parallel {AGENT_LABELS[agent_type]} grading requires loaded weights"
            )
        # Spawn workers reconstruct QBT from the immutable canonical weight vector.
        _WORKER_WEIGHTS = dict(weights)
        _WORKER_AGENT = None
        return
    if agent_type != "dqn" or dqn_policy_state is None:
        raise ValueError("parallel DQN grading requires a validated policy snapshot")

    import torch
    from team01.agent.deep_q_learning import DeepQAgent

    torch.set_num_threads(1)
    agent = DeepQAgent("me", "C", 0, 0, device="cpu")
    # Loading the snapshot creates worker-local networks with no shared learner state.
    agent.policy_network.load_state_dict(dqn_policy_state)
    agent.target_network.load_state_dict(dqn_policy_state)
    agent.policy_network.eval()
    agent.set_learning(no_training=True, epsilon=0.0)
    _assert_frozen_agent(agent)
    _WORKER_AGENT = agent
    _WORKER_WEIGHTS = None


def _make_worker_trial_agent() -> QAgent | QBTAgent | DeepQAgent:
    """Return a new isolated frozen agent from the process-local snapshot."""
    if _WORKER_AGENT_TYPE in Q_LEARNING_AGENTS:
        from team01.agent.q_learning import QAgent, QBTAgent

        if _WORKER_WEIGHTS is None:
            raise RuntimeError(
                f"parallel {_WORKER_AGENT_TYPE} worker has no initialized weight snapshot"
            )
        agent_class = QBTAgent if _WORKER_AGENT_TYPE == "qbt" else QAgent
        agent = agent_class("me", "C", 0, 0)
        agent.weights = dict(_WORKER_WEIGHTS)
        agent.set_learning(no_training=True, epsilon=0.0)
        _assert_frozen_agent(agent)
        return agent
    if _WORKER_AGENT is None:
        raise RuntimeError("parallel DQN worker has no initialized policy")
    agent = copy.deepcopy(_WORKER_AGENT)
    agent.reset_episode()
    agent.set_learning(no_training=True, epsilon=0.0)
    _assert_frozen_agent(agent)
    return agent


def _run_grading_task(task: GradingTrialTask) -> tuple[int, GradingTrialResult]:
    """Run one parent-planned trial in a worker process."""
    runner = _WORKER_TRIAL_RUNNER
    if runner is None:
        raise RuntimeError("grading worker was not initialized")
    kwargs: dict[str, Any] = {
        "seed": task.seed,
        "weights": _WORKER_WEIGHTS,
        "display": _WORKER_DISPLAY,
    }
    if _WORKER_AGENT is not None or _WORKER_AGENT_TYPE in Q_LEARNING_AGENTS:
        kwargs["agent_factory"] = _make_worker_trial_agent
    import torch

    with torch.no_grad():
        result = runner(task.variant, **kwargs)
    if not isinstance(result, GradingTrialResult):
        raise TypeError(
            f"worker returned malformed result for trial {task.trial_index}: "
            f"{type(result).__name__}"
        )
    if result.variant != task.variant or result.seed != task.seed:
        raise ValueError(
            f"worker returned mismatched identity for trial {task.trial_index}: "
            f"V{result.variant} seed={result.seed}"
        )
    return task.trial_index, result


def _summarize_trial_results(
    variant: int,
    runs: int,
    results: list[GradingTrialResult],
) -> dict[str, Any]:
    trials = [
        evaluate_trial(variant, run_number, result)
        for run_number, result in enumerate(results, start=1)
    ]
    wins = sum(trial["success"] for trial in trials)
    rate = wins / runs
    failures = Counter(trial["reason"] for trial in trials if not trial["success"])
    summary: dict[str, Any] = {
        "points_available": VARIANT_POINTS[variant],
        "wins": wins,
        "runs": runs,
        "success_rate": rate,
        "solved": is_solved(wins, runs),
        "points_earned": VARIANT_POINTS[variant] if is_solved(wins, runs) else 0,
        "failure_reasons": dict(sorted(failures.items())),
        "trials": trials,
    }
    failures_only = [trial for trial in trials if not trial["success"]]
    if failures_only:
        first_failure = failures_only[0]
        summary["first_failure"] = {
            "run": first_failure["run"],
            "seed": first_failure["seed"],
            "reason": first_failure["reason"],
            "ticks": first_failure["ticks"],
        }
    return summary


def print_variant_summary(variant: int, summary: dict[str, Any]) -> None:
    rate = summary["success_rate"]
    status = "FULL POINTS" if summary["solved"] else "BELOW THRESHOLD"
    print(f"\nVariant {variant} Results")
    print("-----------------")
    print(f"Wins:         {summary['wins']} / {summary['runs']}")
    print(f"Success Rate: {rate:.1%}")
    print(f"Threshold:    {SOLVED_THRESHOLD:.1%}")
    print(f"Status:       {status}")
    print(f"Points:       {summary['points_earned']} / {summary['points_available']}")
    print("Failures:")
    if summary["failure_reasons"]:
        for reason, count in summary["failure_reasons"].items():
            print(f"    {reason}: {count}")
    else:
        print("    none")


def save_results(result: dict[str, Any], output_path: Path | str = RESULT_PATH) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_evaluation(
    weights: Optional[dict[str, float]],
    *,
    runs: int = RUNS_PER_VARIANT,
    display: bool = DISPLAY,
    weights_path: Optional[Path | str] = None,
    output_path: Path | str = RESULT_PATH,
    trial_runner: Callable[..., GradingTrialResult] = run_trial,
    agent_factory: Optional[Callable[[], QAgent | QBTAgent | DeepQAgent]] = None,
    agent_type: str = DEFAULT_AGENT,
    workers: int = 1,
    canonical_snapshot: Optional[dict[str, Any]] = None,
    seed_generator: Callable[[set[int]], int] = generate_seed,
) -> dict[str, Any]:
    """Grade ``runs`` episodes per variant with an optional bounded worker pool.

    The parent creates every task and seed before dispatch; worker completion
    order must not affect result ordering or scoring.
    """
    if runs < 1:
        raise ValueError("runs per variant must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if agent_type not in AGENT_CHOICES:
        raise ValueError(f"unknown Project 2 agent type: {agent_type}")
    if weights_path is None:
        weights_path = _default_checkpoint_path(agent_type)
    canonical_weights = dict(weights) if weights is not None else None
    if agent_factory is None:
        if agent_type in Q_LEARNING_AGENTS:
            if canonical_weights is None:
                backend = _load_grading_agent(agent_type, Path(weights_path))
                canonical_weights = backend.weights
                agent_factory = backend.agent_factory
            else:
                agent_factory = _frozen_q_agent_factory(
                    agent_type,
                    canonical_weights,
                )
        else:
            backend = _load_grading_agent(agent_type, Path(weights_path))
            agent_factory = backend.agent_factory
            if canonical_snapshot is None:
                canonical_snapshot = backend.canonical_snapshot
    used_seeds: set[int] = set()
    tasks = build_trial_plan(runs, used_seeds, seed_generator=seed_generator)
    planned_results: dict[int, GradingTrialResult] = {}

    if workers == 1:
        for task in tasks:
            result_for_trial = trial_runner(
                task.variant,
                seed=task.seed,
                weights=canonical_weights,
                display=display,
                agent_factory=agent_factory,
            )
            if not isinstance(result_for_trial, GradingTrialResult):
                raise TypeError(
                    f"trial runner returned malformed result for trial "
                    f"{task.trial_index}: {type(result_for_trial).__name__}"
                )
            if (
                result_for_trial.variant != task.variant
                or result_for_trial.seed != task.seed
            ):
                raise ValueError(
                    f"trial runner returned mismatched identity for trial "
                    f"{task.trial_index}"
                )
            planned_results[task.trial_index] = result_for_trial
    else:
        if agent_type == "dqn" and (
            canonical_snapshot is None
        ):
            raise ValueError(
                "parallel DQN grading requires the validated checkpoint snapshot"
            )
        pool_size = min(workers, len(tasks))
        # Spawn imports module-level workers afresh; pass snapshots, not live agents.
        with ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_grading_worker,
            initargs=(
                agent_type,
                canonical_weights,
                (
                    None
                    if canonical_snapshot is None
                    else canonical_snapshot["policy_state_dict"]
                ),
                display,
                trial_runner,
            ),
        ) as executor:
            future_tasks = {
                executor.submit(_run_grading_task, task): task for task in tasks
            }
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    trial_index, trial_result = future.result()
                    if trial_index != task.trial_index:
                        raise ValueError(
                            f"worker returned trial index {trial_index}, "
                            f"expected {task.trial_index}"
                        )
                    if not isinstance(trial_result, GradingTrialResult):
                        raise TypeError(
                            f"worker returned malformed result for trial "
                            f"{task.trial_index}"
                        )
                    if (
                        trial_result.variant != task.variant
                        or trial_result.seed != task.seed
                    ):
                        raise ValueError(
                            f"worker returned mismatched identity for trial "
                            f"{task.trial_index}"
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"grading worker failed for trial {task.trial_index} "
                        f"(V{task.variant}, run {task.run_number}, seed {task.seed}): "
                        f"{error}"
                    ) from error
                planned_results[trial_index] = trial_result
        print(
            f"Parallel grading: {len(planned_results)}/{len(tasks)} valid episodes "
            f"| Worker pool: {pool_size} | Errors: 0"
        )

    ordered_results = order_trial_results(tasks, planned_results)
    ordered_by_index = {
        task.trial_index: result for task, result in zip(tasks, ordered_results)
    }
    variants: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_POINTS:
        print(f"\n=== Variant {variant} ===")
        variant_tasks = [task for task in tasks if task.variant == variant]
        variant_results = [
            ordered_by_index[task.trial_index] for task in variant_tasks
        ]
        for task, trial_result in zip(variant_tasks, variant_results):
            print(
                f"Run {task.run_number:2d}/{runs} | V{variant} | "
                f"seed={task.seed} | {trial_result.reason} | "
                f"ticks={trial_result.ticks}"
            )
        summary = _summarize_trial_results(variant, runs, variant_results)
        variants[str(variant)] = summary
        print_variant_summary(variant, summary)

    total_points = sum(summary["points_earned"] for summary in variants.values())
    maximum_points = sum(VARIANT_POINTS.values())
    total_wins = sum(summary["wins"] for summary in variants.values())
    solved_count = sum(summary["solved"] for summary in variants.values())
    result = {
        "agent_type": agent_type,
        "weights_path": str(Path(weights_path).resolve()),
        "runs_per_variant": runs,
        "threshold": SOLVED_THRESHOLD,
        "learning_enabled": False,
        "epsilon": 0.0,
        "variants": variants,
        "variants_solved": solved_count,
        "total_variants": len(VARIANT_POINTS),
        "overall_wins": total_wins,
        "overall_trials": runs * len(VARIANT_POINTS),
        "total_points": total_points,
        "maximum_points": maximum_points,
    }
    if weights is not None and weights != canonical_weights:
        raise RuntimeError("Project 2 grading evaluation changed canonical weights")
    if agent_type == "dqn":
        result["checkpoint"] = str(Path(weights_path).resolve())
    save_results(result, output_path)
    return result


def print_final_summary(
    result: dict[str, Any],
    *,
    output_path: Path | str = RESULT_PATH,
) -> None:
    print("\n" + "=" * 60)
    print("PROJECT 2 GRADING ESTIMATE")
    print("=" * 60)
    print("\nVariant   Wins   Rate     Solved   Points")
    print("------------------------------------------------------------")
    for variant, summary in result["variants"].items():
        solved = "YES" if summary["solved"] else "NO"
        print(
            f"{variant:<9}{summary['wins']:>2}/{summary['runs']:<4} "
            f"{summary['success_rate']:>6.1%}   {solved:<7} "
            f"{summary['points_earned']:>2}/{summary['points_available']}"
        )
    print("------------------------------------------------------------")
    print(f"TOTAL                              {result['total_points']}/{result['maximum_points']}")
    print(f"Variants solved: {result['variants_solved']} / {result['total_variants']}")
    print(f"Overall wins:    {result['overall_wins']} / {result['overall_trials']}")
    print(f"Weights:         {result['weights_path']}")
    print(f"Results saved:   {Path(output_path)}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  python -m team01.eval.project2_grading_eval --agent qbt --workers 10\n"
            "  python -m team01.eval.project2_grading_eval --agent dqn --workers 10\n"
            "  python -m team01.eval.project2_grading_eval --agent dqn "
            "--weights team01/project2/dqn_checkpoint.pt --workers 10"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default=DEFAULT_AGENT,
        help=(
            "Controller: q=basic Q-learning, dqn=Deep Q-Network, "
            "qbt=Behavior Tree with Q-learning action nodes (default: qbt)."
        ),
    )
    parser.add_argument("--runs", type=int, default=RUNS_PER_VARIANT)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Maximum concurrent worker processes (default: 1; episode count "
            "is controlled by --runs)."
        ),
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--variant", type=int, choices=tuple(sorted(VARIANT_POINTS)))
    parser.add_argument("--seed", type=int)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="display", action="store_true")
    display_group.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=DISPLAY)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if (args.variant is None) != (args.seed is None):
        parser.error("single-trial replay requires both --variant and --seed")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    return args


def main(argv: Optional[list[str]] = None) -> dict[str, Any] | GradingTrialResult:
    args = parse_args(argv)
    weights_path = Path(args.weights) if args.weights else _default_checkpoint_path(args.agent)
    print("PROJECT 2 GRADING EVALUATION")
    print(f"\nWeights: {weights_path.resolve()}")
    print(f"Agent: {AGENT_LABELS[args.agent]}")
    if args.agent == "dqn":
        print(f"Checkpoint: {weights_path.resolve()}")
    print("Mode: Frozen learned policy")
    print("Training: False")
    print("Epsilon: 0.0")
    print("Learning: DISABLED")
    print("Exploration: DISABLED")

    if not weights_path.is_file():
        if args.agent == "dqn":
            raise FileNotFoundError(
                f"DQN checkpoint not found: {weights_path}. "
                "Train the DQN first or provide --weights <path>."
            )
        if args.agent == "qbt":
            raise FileNotFoundError(
                f"QBT learned weights not found: {weights_path}. "
                "Train QBT first or provide --weights <path>."
            )
        raise FileNotFoundError(
            f"Project 2 learned weights not found: {weights_path}. Train or provide "
            "--weights; grading evaluation will not fall back to untrained weights."
        )
    checkpoint_before = weights_path.read_bytes()
    backend = _load_grading_agent(args.agent, weights_path)
    configure_runtime(args.display)

    try:
        if args.variant is not None:
            result = run_trial(
                args.variant,
                seed=args.seed,
                weights=backend.weights,
                display=args.display,
                agent_factory=backend.agent_factory,
            )
            print(
                f"Replay result | V{result.variant} | seed={result.seed} | "
                f"{result.reason} | ticks={result.ticks} | success={result.success}"
            )
        else:
            result = run_evaluation(
                backend.weights,
                runs=args.runs,
                display=args.display,
                weights_path=weights_path,
                output_path=args.output,
                agent_factory=backend.agent_factory,
                agent_type=backend.agent_type,
                workers=args.workers,
                canonical_snapshot=backend.canonical_snapshot,
            )
            print_final_summary(result, output_path=args.output)
    finally:
        if weights_path.read_bytes() != checkpoint_before:
            raise RuntimeError("Project 2 grading evaluation modified the learned checkpoint")
        if (
            backend.agent_type == "dqn"
            and backend.canonical_snapshot is not None
            and backend.canonical_agent is not None
        ):
            from team01.agent.deep_q_learning import checkpoint_snapshots_equal

            canonical_after = backend.canonical_agent.get_checkpoint_snapshot()
            if not checkpoint_snapshots_equal(
                backend.canonical_snapshot,
                canonical_after,
            ):
                raise RuntimeError("Project 2 DQN grading modified its canonical policy")
    return result


if __name__ == "__main__":
    main()
