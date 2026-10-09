"""Progressively train Project 2 Q-learning or DQN agents."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Configuration and curriculum definitions

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BOMBERMAN_DIR = _ROOT / "Bomberman"
if str(_BOMBERMAN_DIR) not in sys.path:
    sys.path.insert(0, str(_BOMBERMAN_DIR))

DEFAULT_MAP_PATH = Path(__file__).resolve().with_name("map.txt")
DRILL_MAP_DIRECTORY = Path(__file__).resolve().with_name("drills")
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().with_name("q_learning_weights.json")
DEFAULT_DQN_CHECKPOINT_PATH = Path(__file__).resolve().with_name("dqn_checkpoint.pt")
DEFAULT_HISTORY_PATH = Path(__file__).resolve().with_name("training_history.jsonl")
DEFAULT_TRIALS = 10
DEFAULT_SURVIVE = 5
DEFAULT_EVAL_TRIALS = 10
DQN_UPDATES_PER_BATCH = 100
FINAL_WIN_RATE_THRESHOLD = 0.90
STALE_WINDOW = 10
STALE_RELATIVE_DELTA_THRESHOLD = 1e-5
STALE_MEAN_TD_THRESHOLD = 0.05
STALE_EVAL_IMPROVEMENT_THRESHOLD = 0.01
UPDATE_CANCELLATION_RATIO = 0.1
GENETIC_RANK_POWER = 2.0
GENETIC_CROSSOVER_OFFSET = 0x6A09E667
RELATIVE_WEIGHT_NORM_THRESHOLD = 1e-9
DEFAULT_EVAL_TIMEOUT_SECONDS = 60.0
EVALUATION_HEARTBEAT_SECONDS = 5.0
DEFAULT_FOCUS_BATCHES = 3
DEFAULT_DRILL_REFRESH_AFTER = 100
DEFAULT_DRILL_REFRESH_TRIALS = 5
DRILL_REFRESH_SEED_OFFSET = 0xD21F0000
DRILL_SEED_OFFSET = 0xD4110000
DRILL_VARIANT_SEED_OFFSET = 0xA2110000
DEFAULT_DISPLAY = True
HEARTBEAT_INTERVAL = 100
PARALLEL_HEARTBEAT_SECONDS = 5
DEFAULT_SEED = random.randint(0, 2**32 - 1)


@dataclass(frozen=True)
class MonsterConfig:
    kind: str
    name: str
    avatar: str
    x: int
    y: int
    detection_range: Optional[int] = None


@dataclass(frozen=True)
class Challenge:
    number: int
    name: str
    map_path: Path
    monsters: tuple[MonsterConfig, ...]


@dataclass(frozen=True)
class Drill:
    number: int
    name: str
    map_path: Path
    description: str
    monsters: tuple[MonsterConfig, ...] = ()
    breach_target: Optional[tuple[int, int]] = None


@dataclass
class TrainingSummary:
    completed_trials: int = 0
    completed_batches: int = 0
    stages_passed: int = 0
    current_stage: Optional[Challenge | Drill] = None
    current_batch: int = 0
    completed: bool = False
    interrupted: bool = False
    stopped_reason: Optional[str] = None
    weights_saved: bool = False
    current_round: int = 0
    decision_trace: list = field(default_factory=list)


@dataclass(frozen=True)
class VariantResumeState:
    """Variant progress that a drill refresh must leave unchanged."""

    unlocked_variants: tuple[int, ...]
    variant_batch: int
    variant_seed: int
    focus_variant: Optional[int]
    batches_since_evaluation: int
    evaluation_win_rates: dict[int, float]
    current_stage: Optional[Challenge | Drill]
    completed_batches: int


@dataclass(frozen=True)
class TrainingOptions:
    """Validated run configuration shared by both execution paths."""

    trials: int
    survive: int
    seed: int
    weights_path: Optional[Path]
    display: bool
    fresh: bool
    max_batches: Optional[int]
    q_contributions_diagnostic: bool
    no_training: bool
    workers: int
    guis: Optional[int]
    eval_trials: int
    history_path: Optional[Path]
    merge_strategy: str
    eval_timeout_seconds: float
    curriculum: str
    stop_after_drills: bool
    focus_batches: int
    drill_refresh_after: int
    drill_refresh_trials: int
    worker_details: bool
    evaluation_runner: Optional[Callable]
    agent_factory: Optional[Callable]
    game_factory: Optional[Callable]
    episode_runner: Optional[Callable]
    agent_type: str = "q"
    dqn_updates_per_batch: int = DQN_UPDATES_PER_BATCH
    start_variant: Optional[int] = None
    dqn_learning_rate: Optional[float] = None
    dqn_epsilon: Optional[float] = None


@dataclass(frozen=True)
class TrainingBatchPlan:
    """Episode assignments and scaled unlock threshold for one batch."""

    episode_items: tuple[int, ...]
    distribution: dict[int, int]
    newest_item: int
    required_newest_wins: int


class TrainingAborted(Exception):
    """Raised when the user closes the active game window."""


class FrozenEvaluationTimeout(Exception):
    """Raised when a frozen evaluation episode exceeds its worker deadline."""


@dataclass(frozen=True)
class ParallelTrainingTask:
    worker_id: int
    trial_index: int
    trial_number: int
    trial_count: int
    batch_number: int
    round_number: int
    seed: int
    challenge: Challenge | Drill
    feature_version: int
    feature_names: tuple[str, ...]
    alpha: float
    gamma: float
    epsilon: float
    starting_weights: dict[str, float]
    displayed: bool
    q_contributions_diagnostic: bool
    no_training: bool = False
    evaluation_timeout_seconds: Optional[float] = None
    phase: str = "variant"
    worker_details: bool = False


@dataclass
class ParallelTrainingResult:
    worker_id: int
    trial_index: int
    outcome: str
    weights: dict[str, float]
    feature_version: Optional[int]
    feature_names: tuple[str, ...]
    trial_number: int = 0
    trial_count: int = 0
    round_number: int = 0
    stage_number: int = 0
    seed: int = 0
    ticks: int = 0
    total_reward: float = 0.0
    bombs_placed: int = 0
    update_count: int = 0
    mean_abs_td_error: float = 0.0
    max_abs_q: float = 0.0
    non_finite_q_values: int = 0
    non_finite_q_fallbacks: int = 0
    displayed: bool = False
    diagnostic_output: str = ""
    error: Optional[str] = None
    phase: str = "variant"


@dataclass(frozen=True)
class DQNRolloutTask:
    """Pickle-safe episode input carrying one frozen CPU policy snapshot."""

    worker_id: int
    trial_index: int
    trial_number: int
    batch_number: int
    seed: int
    challenge: Challenge | Drill
    policy_state_dict: dict[str, object]
    feature_version: int
    feature_names: tuple[str, ...]
    epsilon: float
    displayed: bool
    worker_details: bool = False
    time_limit: Optional[int] = None


@dataclass(frozen=True)
class DQNRolloutResult:
    """One rollout's outcome and transitions for the parent learner."""

    worker_id: int
    trial_index: int
    trial_number: int
    batch_number: int
    seed: int
    challenge_number: int
    outcome: str
    transitions: tuple[object, ...] = ()
    ticks: int = 0
    total_reward: float = 0.0
    bombs_placed: int = 0
    executed_actions: int = 0
    max_abs_q: float = 0.0
    non_finite_q_values: int = 0
    non_finite_q_fallbacks: int = 0
    optimizer_steps: int = 0
    target_sync_count: int = 0
    policy_unchanged: bool = True
    gradients_absent: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Curriculum maps and game setup

def progression(map_path: Path = DEFAULT_MAP_PATH) -> tuple[Challenge, ...]:
    """Return the existing Project 2 variants in their documented order."""
    return (
        Challenge(1, "Variant 1: Alone in the world", map_path, ()),
        Challenge(2, "Variant 2: Random monster", map_path, (MonsterConfig("stupid", "stupid", "S", 3, 9),),),
        Challenge(3, "Variant 3: Self-preserving monster", map_path, (MonsterConfig("smart", "selfpreserving", "S", 3, 9, 1),),),
        Challenge(4, "Variant 4: Aggressive monster", map_path, (MonsterConfig("smart", "aggressive", "A", 3, 13, 2),),),
        Challenge(5, "Variant 5: Stupid and aggressive monsters", map_path, (
                MonsterConfig("stupid", "stupid", "S", 3, 5),
                MonsterConfig("smart", "aggressive", "A", 3, 13, 2),
            ),
        ),
    )

def drill_progression() -> tuple[Drill, ...]:
    return (
        Drill(
            1,
            "Open Exit Navigation",
            DRILL_MAP_DIRECTORY / "d1_open_exit.txt",
            "Navigate an open map and reach its distant exit.",
        ),
        Drill(
            2,
            "Safe Navigation with Monster",
            DRILL_MAP_DIRECTORY / "d2_safe_navigation_monster.txt",
            "Reach the exit while routing around one moving monster.",
            (MonsterConfig("stupid", "drill-monster", "S", 4, 4),),
        ),
        Drill(
            3,
            "Basic Wall Breach",
            DRILL_MAP_DIRECTORY / "d3_basic_wall_breach.txt",
            "Escape a bomb and destroy the only wall crossing to the exit.",
            breach_target=(4, 4),
        ),
        Drill(
            4,
            "Breach and Avoid Stupid Monster",
            DRILL_MAP_DIRECTORY / "d4_breach_and_avoid.txt",
            "Breach the wall, escape the blast, avoid a stupid monster, and reach the exit.",
            (MonsterConfig("stupid", "drill-monster", "S", 7, 6),),
            breach_target=(4, 4),
        ),
        Drill(
            5,
            "Breach and Avoid Aggressive Monster",
            DRILL_MAP_DIRECTORY / "d4_breach_and_avoid.txt",
            "Breach the wall, escape the blast, avoid an aggressive monster, and reach the exit.",
            (MonsterConfig("smart", "drill-monster", "A", 7, 6, 2),),
            breach_target=(4, 4),
        ),
    )


def _map_reaches_exit(world, opened_wall: tuple[int, int] | None = None) -> bool:
    start = (0, 0)
    if world.exitcell is None or world.wall_at(*start):
        return False
    reachable = {start}
    frontier = [start]
    while frontier:
        x, y = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                target = (x + dx, y + dy)
                tx, ty = target
                if not (0 <= tx < world.width() and 0 <= ty < world.height()):
                    continue
                if (
                    (world.wall_at(tx, ty) and target != opened_wall)
                    or target in reachable
                ):
                    continue
                reachable.add(target)
                frontier.append(target)
    return world.exitcell in reachable


def validate_drill_maps() -> None:
    from team01.eval.run_trials import _map_game

    for drill in drill_progression():
        game = _map_game(drill.map_path)
        world = game.world
        if world.exitcell is None or world.wall_at(0, 0):
            raise ValueError(f"drill {drill.number} must have an open start and one exit")
        for monster in drill.monsters:
            if not (0 <= monster.x < world.width() and 0 <= monster.y < world.height()):
                raise ValueError(f"drill {drill.number} monster spawn is outside the map")
            if world.wall_at(monster.x, monster.y) or (monster.x, monster.y) == (0, 0):
                raise ValueError(f"drill {drill.number} monster spawn is not a valid cell")

        wall_count = sum(
            world.wall_at(x, y)
            for x in range(world.width())
            for y in range(world.height())
        )
        if drill.breach_target is None:
            if wall_count or not _map_reaches_exit(world):
                raise ValueError(
                    f"drill {drill.number} must have a clear, reachable exit without bombing"
                )
        else:
            target_x, target_y = drill.breach_target
            if not world.wall_at(target_x, target_y):
                raise ValueError(f"drill {drill.number} breach target is not destructible")
            if _map_reaches_exit(world):
                raise ValueError(f"drill {drill.number} exit is reachable before the breach")
            if not _map_reaches_exit(world, opened_wall=drill.breach_target):
                raise ValueError(f"drill {drill.number} remains blocked after the breach")
            bomb_site = (target_x, target_y - 1)
            if world.wall_at(*bomb_site) or bomb_site == (0, 0):
                raise ValueError(f"drill {drill.number} has no open breach bomb site")


# ---------------------------------------------------------------------------
# Episode and worker setup

def _configure_runtime(display: bool) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ["BOMBERMAN_DISPLAY"] = "1" if display else "0"
    if display:
        os.environ.pop("SDL_VIDEODRIVER", None)
    else:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


def _qagent_class():
    from team01.agent.q_learning import QAgent

    return QAgent


def _feature_names() -> tuple[str, ...]:
    from team01.agent.evaluation import q_feature_names

    return tuple(sorted(q_feature_names()))


def _new_agent():
    agent = _create_qagent()
    agent.weights = {name: 0.0 for name in _feature_names()}
    return agent


def _create_qagent(**learning_parameters):
    return _qagent_class()("me", "C", 0, 0, **learning_parameters)


def _create_dqn_agent():
    from team01.agent.deep_q_learning import DeepQAgent

    return DeepQAgent("me", "C", 0, 0)


def _default_learning_parameters() -> tuple[float, float, float]:
    agent = _create_qagent()
    return agent.alpha, agent.gamma, agent.epsilon


def _weights_are_finite(weights: dict[str, float]) -> bool:
    return all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in weights.values()
    )


def _build_variant_game(
    challenge: Challenge | Drill,
    seed: int,
    weights: Optional[dict[str, float]],
    agent_factory: Callable,
):
    from Bomberman.monsters.selfpreserving_monster import SelfPreservingMonster
    from Bomberman.monsters.stupid_monster import StupidMonster
    from team01.eval.run_trials import _map_game

    random.seed(seed)
    game = _map_game(challenge.map_path)
    for monster in challenge.monsters:
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
            raise ValueError(f"Unknown Project 2 monster type: {monster.kind}")

    agent = agent_factory()
    if weights is not None:
        agent.weights = weights
    game.add_character(agent)
    if not any(
        agent in characters for characters in game.world.characters.values()
    ):
        raise RuntimeError("training agent was not registered in its world")
    return game, agent


def _dqn_policy_snapshot(agent) -> dict[str, object]:
    """Copy only the policy parameters needed by inference-only workers."""
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in agent.policy_network.state_dict().items()
    }


def _make_dqn_rollout_tasks(
    agent,
    assignments: tuple[tuple[Challenge | Drill, int], ...],
    workers: int,
    guis: int,
    batch_number: int,
    epsilon: float,
    worker_details: bool = False,
    time_limit: Optional[int] = None,
) -> tuple[DQNRolloutTask, ...]:
    """Plan episodes that all use the same immutable policy snapshot."""
    from team01.agent.dqn_features import DQN_FEATURE_VERSION, dqn_feature_names

    # CPU tensor copies keep CUDA state and the live parent agent out of worker payloads.
    policy_snapshot = _dqn_policy_snapshot(agent)
    return tuple(
        DQNRolloutTask(
            worker_id=(index - 1) % workers + 1,
            trial_index=index,
            trial_number=index,
            batch_number=batch_number,
            seed=seed,
            challenge=challenge,
            policy_state_dict=policy_snapshot,
            feature_version=DQN_FEATURE_VERSION,
            feature_names=dqn_feature_names(),
            epsilon=epsilon,
            displayed=((index - 1) % min(workers, len(assignments))) < guis,
            worker_details=worker_details,
            time_limit=time_limit,
        )
        for index, (challenge, seed) in enumerate(assignments, start=1)
    )


def _dqn_rollout_worker(task: DQNRolloutTask) -> DQNRolloutResult:
    """Run one frozen-policy episode and return experience, never train or save.

    Kept at module scope with a pickle-safe task/result so Windows spawn can
    import the worker in a fresh process.
    """
    import torch

    diagnostic_buffer = io.StringIO()
    torch.set_num_threads(1)
    agent = None
    outcome = "WORKER_ERROR"
    error_message = None
    episode_result = None
    try:
        with redirect_stdout(diagnostic_buffer), redirect_stderr(diagnostic_buffer):
            _configure_runtime(task.displayed)
            from team01.agent.deep_q_learning import DeepQAgent
            from team01.agent.dqn_features import (
                DQN_FEATURE_VERSION,
                dqn_feature_names,
            )

            if (
                task.feature_version != DQN_FEATURE_VERSION
                or task.feature_names != dqn_feature_names()
            ):
                raise ValueError("DQN rollout task uses a different feature schema")

            agent = DeepQAgent(
                "me",
                "C",
                0,
                0,
                epsilon=task.epsilon,
                device="cpu",
            )
            agent.policy_network.load_state_dict(task.policy_state_dict, strict=True)
            agent.set_rollout(epsilon=task.epsilon, collect_experience=True)
            policy_before = _dqn_policy_snapshot(agent)
            game, agent = _build_variant_game(
                task.challenge,
                task.seed,
                None,
                lambda: agent,
            )
            if task.time_limit is not None:
                game.world.time = task.time_limit
            label = (
                f"Worker {task.worker_id} | Trial {task.trial_index} "
                f"(batch {task.batch_number}) | {task.challenge.number} | "
                f"Seed {task.seed}"
            )
            episode_result = _run_episode(
                game,
                agent,
                task.trial_index,
                label,
                task.displayed,
                worker_details=False,
            )
            policy_after = _dqn_policy_snapshot(agent)
            policy_unchanged = all(
                torch.equal(policy_before[name], policy_after[name])
                for name in policy_before
            )
            gradients_absent = all(
                parameter.grad is None
                for parameter in agent.policy_network.parameters()
            )
            if not policy_unchanged or not gradients_absent:
                raise RuntimeError(
                    "rollout changed policy parameters or accumulated gradients"
                )
            return DQNRolloutResult(
                worker_id=task.worker_id,
                trial_index=task.trial_index,
                trial_number=task.trial_number,
                batch_number=task.batch_number,
                seed=task.seed,
                challenge_number=task.challenge.number,
                outcome=episode_result.outcome,
                transitions=tuple(episode_result.transitions),
                ticks=episode_result.ticks,
                total_reward=episode_result.total_reward,
                bombs_placed=episode_result.bombs_placed,
                executed_actions=episode_result.executed_actions,
                max_abs_q=episode_result.max_abs_q,
                non_finite_q_values=episode_result.non_finite_q_values,
                non_finite_q_fallbacks=episode_result.non_finite_q_fallbacks,
                optimizer_steps=agent.optimizer_steps,
                target_sync_count=agent.target_sync_count,
                policy_unchanged=policy_unchanged,
                gradients_absent=gradients_absent,
            )
    except TrainingAborted as error:
        outcome = "TRAINING_ABORTED"
        error_message = str(error)
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"

    return DQNRolloutResult(
        worker_id=task.worker_id,
        trial_index=task.trial_index,
        trial_number=task.trial_number,
        batch_number=task.batch_number,
        seed=task.seed,
        challenge_number=task.challenge.number,
        outcome=outcome,
        ticks=getattr(episode_result, "ticks", 0),
        total_reward=getattr(episode_result, "total_reward", 0.0),
        bombs_placed=getattr(episode_result, "bombs_placed", 0),
        executed_actions=getattr(episode_result, "executed_actions", 0),
        max_abs_q=getattr(episode_result, "max_abs_q", 0.0),
        non_finite_q_values=getattr(episode_result, "non_finite_q_values", 0),
        non_finite_q_fallbacks=getattr(
            episode_result,
            "non_finite_q_fallbacks",
            0,
        ),
        error=error_message,
    )


def _run_dqn_rollout_batch(
    *,
    agent,
    assignments: tuple[tuple[Challenge | Drill, int], ...],
    workers: int,
    guis: int,
    batch_number: int,
    epsilon: float,
    worker_details: bool = False,
    time_limit: Optional[int] = None,
) -> tuple[DQNRolloutResult, ...]:
    """Run all scheduled DQN episodes from one CPU policy snapshot."""
    if not assignments:
        return ()
    tasks = _make_dqn_rollout_tasks(
        agent,
        assignments,
        workers,
        guis,
        batch_number,
        epsilon,
        worker_details=worker_details,
        time_limit=time_limit,
    )
    executor = ProcessPoolExecutor(max_workers=min(workers, len(tasks)))
    results = []
    try:
        future_tasks = {
            executor.submit(_dqn_rollout_worker, task): task
            for task in tasks
        }
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                result = future.result()
            except Exception as error:
                result = DQNRolloutResult(
                    worker_id=task.worker_id,
                    trial_index=task.trial_index,
                    trial_number=task.trial_number,
                    batch_number=task.batch_number,
                    seed=task.seed,
                    challenge_number=task.challenge.number,
                    outcome="WORKER_ERROR",
                    error=f"{type(error).__name__}: {error}",
                )
            expected_identity = (
                task.worker_id,
                task.trial_index,
                task.trial_number,
                task.batch_number,
                task.seed,
                task.challenge.number,
            )
            actual_identity = (
                result.worker_id,
                result.trial_index,
                result.trial_number,
                result.batch_number,
                result.seed,
                result.challenge_number,
            )
            if actual_identity != expected_identity:
                result = replace(
                    result,
                    outcome="WORKER_ERROR",
                    error="worker result identity does not match its scheduled task",
                )
            if result.outcome == "TRAINING_ABORTED":
                raise TrainingAborted(result.error or "DQN rollout was aborted")
            results.append(result)
    except KeyboardInterrupt:
        _terminate_executor_processes(executor)
        raise
    except TrainingAborted:
        _terminate_executor_processes(executor)
        raise
    except Exception:
        _terminate_executor_processes(executor)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=True)

    # Completion order depends on episode length; restore plan order before replay ingestion.
    return tuple(sorted(results, key=lambda item: item.trial_index))


def _display_callback(game, progress_label: str):
    import pygame

    pygame.display.set_caption(f"Project 2 Q-Learning | {progress_label}")
    game.display_gui()

    def on_tick(tick: int) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise TrainingAborted("game window closed")
        game.display_gui()
        pygame.time.wait(50)

    return on_tick


def _run_episode(
    game,
    agent,
    episode: int,
    progress_label: str,
    display: bool,
    *,
    worker_details: bool = False,
):
    from team01.eval.train_q_learning import run_episode

    on_tick = _display_callback(game, progress_label or "Project 2 Training") if display else None
    return run_episode(
        game.world,
        agent,
        episode,
        progress_label=progress_label if worker_details else None,
        heartbeat_interval=HEARTBEAT_INTERVAL if worker_details else 0,
        on_tick=on_tick,
    )


def _run_parallel_episode(game, agent, episode: int, progress_label: str, display: bool):
    from team01.eval.train_q_learning import run_episode

    on_tick = _display_callback(game, progress_label) if display else None
    return run_episode(
        game.world,
        agent,
        episode,
        progress_label=None,
        heartbeat_interval=0,
        on_tick=on_tick,
    )


def _run_frozen_evaluation_episode(
    game,
    agent,
    episode: int,
    timeout_seconds: float,
):
    from team01.eval.train_q_learning import run_episode

    deadline = time.monotonic() + timeout_seconds

    def check_deadline(_tick: int) -> None:
        if time.monotonic() >= deadline:
            raise FrozenEvaluationTimeout("frozen evaluation worker deadline exceeded")

    return run_episode(
        game.world,
        agent,
        episode,
        progress_label=None,
        heartbeat_interval=0,
        on_tick=check_deadline,
    )


def _make_parallel_result(
    task: ParallelTrainingTask,
    agent,
    outcome: str,
    episode_result=None,
    diagnostic_output: str = "",
    error: Optional[str] = None,
) -> ParallelTrainingResult:
    from team01.agent.evaluation import Q_FEATURE_VERSION, q_feature_names

    weights = dict(agent.weights) if agent is not None else {}
    update_count = getattr(agent, "td_update_count", 0)
    mean_abs_td_error = (
        agent.total_abs_td_error / update_count if agent is not None and update_count else 0.0
    )
    return ParallelTrainingResult(
        worker_id=task.worker_id,
        trial_index=task.trial_index,
        outcome=outcome,
        weights=weights,
        feature_version=Q_FEATURE_VERSION,
        feature_names=tuple(q_feature_names()),
        trial_number=task.trial_number,
        trial_count=task.trial_count,
        round_number=task.round_number,
        stage_number=task.challenge.number,
        seed=task.seed,
        ticks=getattr(episode_result, "ticks", 0),
        total_reward=getattr(
            episode_result,
            "total_reward",
            getattr(agent, "episode_reward", 0.0),
        ),
        bombs_placed=getattr(episode_result, "bombs_placed", getattr(agent, "bombs_placed", 0)),
        update_count=update_count,
        mean_abs_td_error=mean_abs_td_error,
        max_abs_q=getattr(episode_result, "max_abs_q", getattr(agent, "max_abs_q", 0.0)),
        non_finite_q_values=getattr(agent, "non_finite_q_values", 0),
        non_finite_q_fallbacks=getattr(agent, "non_finite_q_fallbacks", 0),
        displayed=task.displayed,
        diagnostic_output=diagnostic_output,
        error=error,
        phase=task.phase,
    )


def _parallel_training_worker(task: ParallelTrainingTask) -> ParallelTrainingResult:
    """Construct and run one independent episode inside a child process."""
    diagnostic_buffer = io.StringIO()
    agent = None
    episode_result = None
    outcome = "WORKER_ERROR"
    error = None

    try:
        with redirect_stdout(diagnostic_buffer), redirect_stderr(diagnostic_buffer):
            try:
                _configure_runtime(task.displayed)
                from team01.agent.evaluation import Q_FEATURE_VERSION, q_feature_names

                if (
                    task.feature_version != Q_FEATURE_VERSION
                    or task.feature_names != tuple(q_feature_names())
                ):
                    raise ValueError("Worker feature schema differs from round snapshot")
                random.seed(task.seed)
                game, agent = _build_variant_game(
                    task.challenge,
                    task.seed,
                    dict(task.starting_weights),
                    partial(
                        _create_qagent,
                        alpha=task.alpha,
                        gamma=task.gamma,
                        epsilon=task.epsilon,
                    ),
                )
                agent.set_learning(task.no_training, epsilon=task.epsilon)
                agent.q_contributions_diagnostic = (
                    task.q_contributions_diagnostic and task.displayed
                )
                label = (
                    f"Worker {task.worker_id} | Trial {task.trial_index} "
                    f"(batch {task.trial_number}/{task.trial_count}) | "
                    f"{task.phase.title()} {task.challenge.number} | "
                    f"Seed {task.seed}"
                )
                if task.evaluation_timeout_seconds is None:
                    episode_result = _run_parallel_episode(
                        game,
                        agent,
                        task.trial_index,
                        label,
                        task.displayed,
                    )
                else:
                    episode_result = _run_frozen_evaluation_episode(
                        game,
                        agent,
                        task.trial_index,
                        task.evaluation_timeout_seconds,
                    )
                outcome = episode_result.outcome
            except FrozenEvaluationTimeout as caught:
                outcome = "EVALUATION_TIMEOUT"
                error = str(caught)
            except TrainingAborted as caught:
                outcome = "CANCELLED"
                error = str(caught)
            except Exception as caught:
                outcome = "WORKER_ERROR"
                error = f"{type(caught).__name__}: {caught}"
    finally:
        try:
            import pygame

            pygame.quit()
        except Exception:
            pass

    return _make_parallel_result(
        task,
        agent,
        outcome,
        episode_result,
        diagnostic_buffer.getvalue(),
        error,
    )


def _save_weights(agent, weights: dict[str, float], weights_path: Path) -> None:
    writer = agent if agent is not None else _new_agent()
    writer.weights = weights
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    writer.save_weights(str(weights_path))


def _load_initial_weights(
    weights_path: Path,
    fresh: bool,
    agent_factory: Callable,
) -> tuple[dict[str, float], object, bool]:
    file_exists = weights_path.exists()
    if fresh:
        agent = agent_factory()
        agent.weights = {name: 0.0 for name in _feature_names()}
        if file_exists:
            print(
                "Fresh start requested. Existing weights will remain untouched "
                "until a valid training checkpoint is produced.",
                flush=True,
            )
            can_save = False
        else:
            print("Fresh start requested. Starting with zero weights.", flush=True)
            can_save = False
        return agent.weights, agent, can_save

    if file_exists:
        agent = agent_factory()
        try:
            agent.load_weights(str(weights_path))
        except Exception as error:
            print(
                "Existing weights use an incompatible feature version. Fresh "
                f"training is required; pass --fresh. Original file left untouched "
                f"at {weights_path}. Details: {error}",
                flush=True,
            )
            raise
        print(f"Loaded existing weights from {weights_path}", flush=True)
        return agent.weights, agent, True

    agent = agent_factory()
    agent.weights = {name: 0.0 for name in _feature_names()}
    print("No saved weights found. Starting training from scratch.", flush=True)
    return agent.weights, agent, False


def _result_is_numerically_stable(result, weights: dict[str, float]) -> bool:
    return (
        result.outcome != "NON_FINITE"
        and result.non_finite_q_values == 0
        and result.non_finite_q_fallbacks == 0
        and _weights_are_finite(weights)
    )


def _apply_dqn_overrides(agent, learning_rate, epsilon) -> None:
    """Apply run-level LR/epsilon overrides after any checkpoint load."""
    if learning_rate is not None:
        agent.set_learning_rate(learning_rate)
    if epsilon is not None:
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("--dqn-epsilon must be between 0 and 1")
        agent.epsilon = epsilon
    if learning_rate is not None or epsilon is not None:
        print(
            "DQN overrides applied: "
            f"lr={[group['lr'] for group in agent.optimizer.param_groups]} "
            f"epsilon={agent.epsilon} gamma={agent.gamma} "
            f"optimizer_steps={agent.optimizer_steps} episodes={agent.episodes}",
            flush=True,
        )


def _load_dqn_training_state(
    checkpoint_path: Path,
    fresh: bool,
    agent_factory: Callable,
    learning_rate: Optional[float] = None,
    epsilon: Optional[float] = None,
):
    """Create or restore the canonical DQN learner without touching its checkpoint."""
    checkpoint_exists = checkpoint_path.is_file()
    agent = agent_factory()
    if fresh:
        if checkpoint_exists:
            print(
                "Fresh start requested. Existing DQN checkpoint will remain untouched "
                "until a valid training batch completes.",
                flush=True,
            )
        else:
            print("Fresh start requested. Starting with a new DQN.", flush=True)
        _apply_dqn_overrides(agent, learning_rate, epsilon)
        return agent, False

    if checkpoint_exists:
        agent.load_checkpoint(checkpoint_path)
        print(f"Loaded existing DQN checkpoint from {checkpoint_path}", flush=True)
        # Loading Adam restores its saved LR; apply the requested run override afterward.
        _apply_dqn_overrides(agent, learning_rate, epsilon)
        return agent, False

    print("No DQN checkpoint found. Starting with a new DQN.", flush=True)
    _apply_dqn_overrides(agent, learning_rate, epsilon)
    return agent, False


def _consume_dqn_episode_results(
    agent,
    episode_results: list[object],
    updates_per_batch: int,
) -> dict:
    """Ingest completed rollout transitions, then perform central replay updates."""
    # Only this parent-side consumer mutates replay, policy parameters, and checkpoint state.
    transitions_collected = 0
    for episode_result in episode_results:
        for transition in getattr(episode_result, "transitions", ()):
            agent.store_transition(transition)
            transitions_collected += 1

    updates = []
    for _ in range(updates_per_batch):
        agent._central_update_in_progress = True
        update = agent.train_batch()
        agent._central_update_in_progress = False
        if update is None:
            break
        updates.append(update)

    return {
        "transitions_collected": transitions_collected,
        "replay_size": len(agent.replay_buffer),
        "replay_capacity": agent.replay_buffer.capacity,
        "replay_warmup": agent.replay_warmup,
        "updates_this_batch": len(updates),
        "optimizer_steps": agent.optimizer_steps,
        "mean_loss": (
            math.fsum(update.loss for update in updates) / len(updates)
            if updates
            else None
        ),
        "mean_abs_td_error": (
            math.fsum(update.mean_abs_td_error for update in updates) / len(updates)
            if updates
            else None
        ),
        "max_abs_td_error": max(
            (update.max_abs_td_error for update in updates),
            default=None,
        ),
        "mean_q": (
            math.fsum(update.mean_q for update in updates) / len(updates)
            if updates
            else None
        ),
        "max_abs_q": max((update.max_abs_q for update in updates), default=None),
        "mean_gradient_norm": (
            math.fsum(update.gradient_norm for update in updates) / len(updates)
            if updates
            else None
        ),
        "max_gradient_norm": max(
            (update.gradient_norm for update in updates),
            default=None,
        ),
        "target_synced": any(update.target_synced for update in updates),
    }


def _print_dqn_batch_summary(metrics: dict) -> None:
    """Print aggregate central DQN learning diagnostics for one batch."""
    if metrics.get("worker_limit", 1) > 1:
        print(
            f"Parallel DQN rollout: {metrics['episodes_valid']}/"
            f"{metrics['episodes_assigned']} episodes valid | "
            f"Worker limit: {metrics['worker_limit']} | "
            f"Episode errors: {metrics['episode_errors']}",
            flush=True,
        )
    print("DQN learning:", flush=True)
    print(
        f"Transitions collected: {metrics['transitions_collected']} | "
        f"Replay size: {metrics['replay_size']} / {metrics['replay_capacity']} | "
        f"Warmup: {metrics['replay_size']} / {metrics['replay_warmup']}",
        flush=True,
    )
    print(
        f"Optimizer steps this batch: {metrics['updates_this_batch']} | "
        f"Total optimizer steps: {metrics['optimizer_steps']}",
        flush=True,
    )
    if metrics["updates_this_batch"]:
        print(
            f"Mean loss: {metrics['mean_loss']:.3f} | "
            f"Mean |TD|: {metrics['mean_abs_td_error']:.3f} | "
            f"Max |TD|: {metrics['max_abs_td_error']:.3f} | "
            f"Mean Q: {metrics['mean_q']:.3f} | "
            f"Max |Q|: {metrics['max_abs_q']:.3f} | "
            f"Mean gradient norm: {metrics['mean_gradient_norm']:.3f} | "
            f"Max gradient norm: {metrics['max_gradient_norm']:.3f} | "
            f"Target synced: {'yes' if metrics['target_synced'] else 'no'}",
            flush=True,
        )
    else:
        print(
            (
                "No optimizer update: replay warmup has not been reached "
                f"({metrics['replay_size']}/{metrics['replay_warmup']} transitions)."
            )
            if metrics["replay_size"] < metrics["replay_warmup"]
            else "No optimizer update was performed.",
            flush=True,
        )


def _dqn_result_is_numerically_stable(result) -> bool:
    return (
        getattr(result, "error", None) is None
        and result.outcome not in {"WORKER_ERROR", "TRAINING_ABORTED"}
        and getattr(result, "policy_unchanged", True)
        and getattr(result, "gradients_absent", True)
        and getattr(result, "optimizer_steps", 0) == 0
        and getattr(result, "target_sync_count", 0) == 0
        and result.outcome != "NON_FINITE"
        and result.non_finite_q_values == 0
        and result.non_finite_q_fallbacks == 0
        and math.isfinite(result.total_reward)
        and math.isfinite(result.max_abs_q)
    )


def _dqn_rollout_coverage(
    episode_items: tuple[int, ...],
    results: tuple[DQNRolloutResult, ...],
) -> tuple[dict[int, int], tuple[int, ...]]:
    valid_counts = {
        item_number: sum(
            result.challenge_number == item_number
            and _dqn_result_is_numerically_stable(result)
            for result in results
        )
        for item_number in sorted(set(episode_items))
    }
    missing_items = tuple(
        item_number
        for item_number, valid_count in valid_counts.items()
        if valid_count == 0
    )
    return valid_counts, missing_items


def _trial_rounds(trials: int, workers: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(start, min(start + workers, trials + 1)))
        for start in range(1, trials + 1, workers)
    )


def _mixed_training_plan(
    trials: int,
    unlocked_variants: tuple[int, ...] | list[int],
    seed: int,
    batch_number: int,
    *,
    focus_variant: Optional[int] = None,
) -> tuple[int, ...]:
    """Plan a batch around the newest or focused variant and replay older ones."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not unlocked_variants:
        raise ValueError("at least one variant must be unlocked")

    unlocked = tuple(sorted(set(unlocked_variants)))
    if focus_variant is not None and focus_variant not in unlocked:
        raise ValueError("focus variant must be unlocked")
    newest = focus_variant if focus_variant is not None else unlocked[-1]
    if len(unlocked) == 1:
        return (newest,) * trials

    newest_count = (trials + 1) // 2
    replay_count = trials - newest_count
    plan = [newest] * newest_count
    replay_variants = tuple(variant for variant in unlocked if variant != newest)
    replay_start = (seed + batch_number - 1) % len(replay_variants)
    if focus_variant is None:
        plan.extend(
            replay_variants[(replay_start + index) % len(replay_variants)]
            for index in range(replay_count)
        )
    else:
        base_replay_count, extra_replay_count = divmod(
            replay_count, len(replay_variants)
        )
        replay_counts = [base_replay_count] * len(replay_variants)
        for index in range(extra_replay_count):
            replay_counts[(replay_start + index) % len(replay_variants)] += 1
        for variant, count in zip(replay_variants, replay_counts):
            plan.extend([variant] * count)

    shuffle_seed = (
        seed
        ^ (batch_number * 0x9E3779B1)
        ^ (trials * 0x85EBCA77)
        ^ sum(variant * 0xC2B2AE3D for variant in unlocked)
        ^ ((focus_variant or 0) * 0x27D4EB2F)
    )
    random.Random(shuffle_seed).shuffle(plan)
    return tuple(plan)


def _select_weakest_variant(
    evaluation_win_rates: dict[int, float],
    variants: tuple[int, ...] | list[int],
) -> int:
    if not variants:
        raise ValueError("at least one variant is required for focus selection")
    return min(
        variants,
        key=lambda variant: (evaluation_win_rates.get(variant, 0.0), variant),
    )


def _evaluation_win_counts(
    evaluation_win_rates: dict[int, float],
    eval_trials: int,
    variants: tuple[int, ...] | list[int],
) -> dict[int, int]:
    return {
        variant: max(
            0,
            min(
                eval_trials,
                round(evaluation_win_rates.get(variant, 0.0) * eval_trials),
            ),
        )
        for variant in variants
    }


def _frozen_evaluation_improved(
    previous_rates: dict[int, float],
    current_rates: dict[int, float],
    eval_trials: int,
    focus_variant: Optional[int],
    variants: tuple[int, ...] | list[int],
) -> bool:
    if not previous_rates:
        return True
    previous = _evaluation_win_counts(previous_rates, eval_trials, variants)
    current = _evaluation_win_counts(current_rates, eval_trials, variants)
    required_wins = math.ceil(FINAL_WIN_RATE_THRESHOLD * eval_trials)
    previous_passing = sum(count >= required_wins for count in previous.values())
    current_passing = sum(count >= required_wins for count in current.values())
    if current_passing > previous_passing:
        return True
    if min(current.values()) > min(previous.values()):
        return True
    return (
        focus_variant is not None
        and current[focus_variant] > previous[focus_variant]
    )


def _print_focus_selection(
    evaluation_win_rates: dict[int, float],
    focus_variant: int,
    focus_batches: int,
) -> None:
    print("FOCUS SELECTION", flush=True)
    print(
        f"Next focus: V{focus_variant} "
        f"{progression()[focus_variant - 1].name}"
        f"\nFocused batches before reevaluation: {focus_batches}",
        flush=True,
    )


def _print_focus_variant_results(
    focus_variant: int,
    results: list[tuple[int, object]],
) -> None:
    focus_results = [
        result for variant, result in results if variant == focus_variant
    ]
    wins = sum(result.outcome == "WON" for result in focus_results)
    mean_reward = (
        math.fsum(result.total_reward for result in focus_results) / len(focus_results)
        if focus_results
        else 0.0
    )
    mean_ticks = (
        math.fsum(result.ticks for result in focus_results) / len(focus_results)
        if focus_results
        else 0.0
    )
    print(
        "FOCUS VARIANT RESULTS\n"
        f"V{focus_variant}: Episodes: {len(focus_results)} | Wins: {wins} | "
        f"Win rate: {wins / len(focus_results) * 100 if focus_results else 0.0:.1f}% | "
        f"Mean reward: {mean_reward:.1f} | Mean ticks: {mean_ticks:.1f}",
        flush=True,
    )


def _training_distribution(episode_variants: tuple[int, ...]) -> dict[int, int]:
    return {
        variant: episode_variants.count(variant)
        for variant in sorted(set(episode_variants))
    }


def _phase_history_fields(
    phase: str,
    unlocked_items: list[int] | tuple[int, ...],
    episode_items: tuple[int, ...],
) -> dict:
    fields = {
        "phase": phase,
        "unlocked_items": list(unlocked_items),
        "episode_items": list(episode_items),
    }
    if phase == "drill":
        fields["unlocked_drills"] = list(unlocked_items)
        fields["episode_drills"] = list(episode_items)
    else:
        fields["unlocked_variants"] = list(unlocked_items)
        fields["episode_variants"] = list(episode_items)
    return fields


def _required_newest_wins(survive: int, trials: int, newest_episodes: int) -> int:
    """Scale the configured survive/trials threshold to the newest item's share."""
    return math.ceil((survive / trials) * newest_episodes)


def _plan_training_batch(
    trials: int,
    survive: int,
    unlocked_items: tuple[int, ...] | list[int],
    seed: int,
    batch_number: int,
    focus_variant: Optional[int] = None,
) -> TrainingBatchPlan:
    """Choose this batch's episodes and the win threshold for its newest item."""
    episode_items = _mixed_training_plan(
        trials,
        unlocked_items,
        seed,
        batch_number,
        focus_variant=focus_variant,
    )
    distribution = _training_distribution(episode_items)
    newest_item = focus_variant if focus_variant is not None else max(unlocked_items)
    required_wins = _required_newest_wins(
        survive,
        trials,
        distribution.get(newest_item, 0) or 1,
    )
    return TrainingBatchPlan(
        episode_items=episode_items,
        distribution=distribution,
        newest_item=newest_item,
        required_newest_wins=required_wins,
    )


def _count_newest_item_wins(
    results: list[tuple[int, object]],
    newest_item: int,
    *,
    require_error_free: bool = False,
) -> int:
    """Count wins for the newest curriculum item in a completed batch."""
    return sum(
        item_number == newest_item
        and result.outcome == "WON"
        and (not require_error_free or getattr(result, "error", None) is None)
        for item_number, result in results
    )


def _drill_refresh_is_due(
    *,
    curriculum: str,
    no_training: bool,
    phase: str,
    stagnation_trials: int,
    refresh_after: int,
    final_complete: bool,
    all_variants_unlocked: bool,
    evaluation_performed: bool,
) -> bool:
    """Check whether stalled variant training should replay the drills."""
    return (
        curriculum == "drills"
        and not no_training
        and phase == "variant"
        and refresh_after > 0
        and stagnation_trials >= refresh_after
        and not final_complete
        and (not all_variants_unlocked or evaluation_performed)
    )


def _frozen_evaluation_is_due(
    *,
    phase: str,
    all_variants_unlocked: bool,
    no_training: bool,
    focus_variant: Optional[int],
    batches_since_evaluation: int,
    focus_batches: int,
) -> bool:
    """Evaluate after unlock, before focusing, or when the focus cycle ends."""
    return (
        phase == "variant"
        and all_variants_unlocked
        and (
            no_training
            or focus_variant is None
            or batches_since_evaluation >= focus_batches
        )
    )


def _curriculum_items(phase: str):
    items = drill_progression() if phase == "drill" else progression()
    return items, {item.number: item for item in items}


@dataclass(frozen=True)
class BatchDecision:
    """Curriculum outcome of one completed training batch."""

    newest_item: int
    newest_wins: int
    required_newest_wins: int
    stage_unlocked: bool
    phase_transitioned: bool
    evaluation_due: bool
    progress_detected: bool = False


@dataclass(frozen=True)
class EvaluationDecision:
    """Curriculum outcome of one all-variant frozen evaluation."""

    complete: bool
    improved: Optional[bool]
    focus_variant: Optional[int]
    previous_focus: Optional[int]
    progress_detected: bool = False


class CurriculumEngine:
    """Backend-agnostic curriculum decisions (unlock, focus, stagnation, refresh).

    Pure state machine: no I/O, no agent. Callers run episodes and feed results
    back through ``plan_batch()`` and ``record_batch()``. Unlocking uses the
    configured ``survive / trials`` threshold, scaled to the newest item's share
    of the batch. Once every variant is unlocked, separate frozen evaluations
    apply ``FINAL_WIN_RATE_THRESHOLD`` and select a focus variant. Stagnation
    counts completed training episodes; drill refreshes reset that counter but
    never unlock variants. Every decision is recorded as deterministic,
    JSON-serializable trace data.
    """

    def __init__(
        self,
        *,
        trials: int,
        survive: int,
        seed: int,
        eval_trials: int,
        curriculum: str = "drills",
        focus_batches: int = 1,
        drill_refresh_after: int = 0,
        no_training: bool = False,
        start_variant: Optional[int] = None,
    ) -> None:
        self.trials = trials
        self.survive = survive
        self.seed = seed
        self.eval_trials = eval_trials
        self.curriculum = curriculum
        self.focus_batches = focus_batches
        self.drill_refresh_after = drill_refresh_after
        self.no_training = no_training
        self.drill_count = len(drill_progression())
        self.variant_count = len(progression())
        self.phase = "drill" if curriculum == "drills" else "variant"
        self.unlocked_items = [1]
        self.phase_batch = 0
        self.phase_seed = seed ^ DRILL_SEED_OFFSET if self.phase == "drill" else seed
        if start_variant is not None:
            if not 1 <= start_variant <= self.variant_count:
                raise ValueError(
                    f"start_variant must be between 1 and {self.variant_count}"
                )
            # Start after the V1..N availability point, not its pass threshold:
            # V1..N are available, N is still the newest stage, and N+1 stays locked.
            self.phase = "variant"
            self.unlocked_items = list(range(1, start_variant + 1))
            self.phase_seed = (
                seed ^ DRILL_VARIANT_SEED_OFFSET if curriculum == "drills" else seed
            )
        self.focus_variant: Optional[int] = None
        self.batches_since_evaluation = 0
        self.last_evaluation_win_rates: dict[int, float] = {}
        self.stagnation_trials = 0
        self.drill_refresh_count = 0
        self.completed_batches = 0
        self.complete = False
        self.trace: list[dict] = []
        self._plan: Optional[TrainingBatchPlan] = None
        self._batch_phase = self.phase
        self._batch_focus: Optional[int] = None
        self._batch_last_rates: dict[int, float] = {}
        self._evaluation_due = False

    @property
    def item_count(self) -> int:
        return self.drill_count if self.phase == "drill" else self.variant_count

    @property
    def all_variants_unlocked(self) -> bool:
        return self.phase == "variant" and len(self.unlocked_items) == self.variant_count

    @property
    def newest_item(self) -> int:
        return self.unlocked_items[-1]

    @property
    def batch_phase(self) -> str:
        return self._batch_phase

    @property
    def batch_focus_variant(self) -> Optional[int]:
        return self._batch_focus

    @property
    def batch_last_evaluation_win_rates(self) -> dict[int, float]:
        return dict(self._batch_last_rates)

    def plan_batch(self) -> TrainingBatchPlan:
        self.phase_batch += 1
        self._batch_phase = self.phase
        self._batch_focus = (
            self.focus_variant
            if self.all_variants_unlocked and not self.no_training
            else None
        )
        self._batch_last_rates = dict(self.last_evaluation_win_rates)
        self._evaluation_due = False
        self._plan = _plan_training_batch(
            self.trials,
            self.survive,
            self.unlocked_items,
            self.phase_seed,
            self.phase_batch,
            focus_variant=self._batch_focus,
        )
        self.trace.append(
            {
                "event": "plan",
                "batch": self.completed_batches + 1,
                "phase": self._batch_phase,
                "unlocked": list(self.unlocked_items),
                "focus": self._batch_focus,
                "episode_items": list(self._plan.episode_items),
                "required_newest_wins": self._plan.required_newest_wins,
            }
        )
        return self._plan

    def record_batch(
        self,
        results: list[tuple[int, object]],
        *,
        trained_episodes: Optional[int] = None,
        require_error_free: bool = False,
    ) -> BatchDecision:
        """Apply unlock/stagnation rules to a completed training batch.

        Only wins on the newest scheduled item can unlock the next item; this
        batch result does not substitute for the later frozen final evaluation.
        """
        if self._plan is None:
            raise RuntimeError("record_batch() requires a preceding plan_batch()")
        plan = self._plan
        batch_phase = self._batch_phase
        newest_item = self.newest_item
        wins = _count_newest_item_wins(
            results, newest_item, require_error_free=require_error_free
        )
        required = plan.required_newest_wins
        self.completed_batches += 1
        unlocked = False
        transitioned = False
        progress = False
        if self._batch_focus is not None:
            self.batches_since_evaluation += 1
        elif wins >= required and newest_item < self.item_count:
            self.unlocked_items.append(newest_item + 1)
            unlocked = True
            if batch_phase == "variant" and not self.no_training:
                self.stagnation_trials = 0
                progress = True
        elif newest_item == self.item_count and wins >= required:
            if batch_phase == "drill":
                transitioned = True
                self.phase = "variant"
                self.unlocked_items = [1]
                self.phase_batch = 0
                self.phase_seed = (
                    self.seed ^ DRILL_VARIANT_SEED_OFFSET
                    if self.curriculum == "drills"
                    else self.seed
                )
        if batch_phase == "variant" and not self.no_training and not unlocked:
            self.stagnation_trials += (
                len(plan.episode_items) if trained_episodes is None else trained_episodes
            )
        self._evaluation_due = (
            batch_phase == "variant"
            and len(self.unlocked_items) == self.variant_count
            and _frozen_evaluation_is_due(
                phase=batch_phase,
                all_variants_unlocked=True,
                no_training=self.no_training,
                focus_variant=self.focus_variant,
                batches_since_evaluation=self.batches_since_evaluation,
                focus_batches=self.focus_batches,
            )
        )
        self.trace.append(
            {
                "event": "batch",
                "batch": self.completed_batches,
                "phase": batch_phase,
                "newest_item": newest_item,
                "newest_wins": wins,
                "required_newest_wins": required,
                "unlocked": unlocked,
                "phase_transitioned": transitioned,
                "evaluation_due": self._evaluation_due,
                "stagnation_trials": self.stagnation_trials,
                "unlocked_items": list(self.unlocked_items),
            }
        )
        return BatchDecision(
            newest_item,
            wins,
            required,
            unlocked,
            transitioned,
            self._evaluation_due,
            progress,
        )

    def record_evaluation(
        self,
        win_rates: dict[int, float],
        *,
        complete: bool,
    ) -> EvaluationDecision:
        """Record a frozen all-variant evaluation and update focus/completion."""
        if not self._evaluation_due:
            raise RuntimeError("record_evaluation() requires evaluation_due")
        variants = tuple(range(1, self.variant_count + 1))
        improved: Optional[bool] = None
        progress = False
        if not complete and not self.no_training:
            focus_for_progress = (
                self._batch_focus
                if self._batch_focus is not None
                else self.focus_variant
            )
            if not self._batch_last_rates:
                self.stagnation_trials = 0
                progress = True
            else:
                improved = _frozen_evaluation_improved(
                    self._batch_last_rates,
                    win_rates,
                    self.eval_trials,
                    focus_for_progress,
                    variants,
                )
                if improved:
                    self.stagnation_trials = 0
                    progress = True
        self.last_evaluation_win_rates = dict(win_rates)
        previous = self.focus_variant
        if not complete:
            self.focus_variant = _select_weakest_variant(win_rates, variants)
            self.batches_since_evaluation = 0
        self.complete = complete
        self.trace.append(
            {
                "event": "evaluation",
                "batch": self.completed_batches,
                "win_rates": {str(v): win_rates[v] for v in sorted(win_rates)},
                "complete": complete,
                "improved": improved,
                "progress_detected": progress,
                "focus": self.focus_variant,
                "stagnation_trials": self.stagnation_trials,
            }
        )
        return EvaluationDecision(
            complete, improved, self.focus_variant, previous, progress
        )

    def refresh_due(self, *, evaluation_performed: bool) -> bool:
        return _drill_refresh_is_due(
            curriculum=self.curriculum,
            no_training=self.no_training,
            phase=self._batch_phase,
            stagnation_trials=self.stagnation_trials,
            refresh_after=self.drill_refresh_after,
            final_complete=self.complete,
            all_variants_unlocked=len(self.unlocked_items) == self.variant_count,
            evaluation_performed=evaluation_performed,
        )

    def refresh_completed(self) -> None:
        """Unlock/focus progress is kept; only the stagnation counter resets."""
        self.drill_refresh_count += 1
        self.stagnation_trials = 0
        self.trace.append(
            {
                "event": "drill_refresh",
                "batch": self.completed_batches,
                "refresh_count": self.drill_refresh_count,
                "unlocked_items": list(self.unlocked_items),
                "focus": self.focus_variant,
            }
        )

def _weight_update_diagnostics(
    round_start_weights: dict[str, float],
    worker_weights: list[dict[str, float]],
    merged_weights: dict[str, float],
    worker_td_errors: list[tuple[float, int]],
    max_abs_q: float,
) -> dict:
    weight_deltas = {
        name: merged_weights[name] - round_start_weights[name]
        for name in round_start_weights
    }
    abs_deltas = [abs(value) for value in weight_deltas.values()]
    canonical_delta_l2 = math.sqrt(math.fsum(value * value for value in weight_deltas.values()))
    starting_weight_l2 = math.sqrt(
        math.fsum(value * value for value in round_start_weights.values())
    )
    mean_worker_delta = {
        name: (
            math.fsum(
                worker_weights_i[name] - round_start_weights[name]
                for worker_weights_i in worker_weights
            )
            / len(worker_weights)
            if worker_weights
            else 0.0
        )
        for name in round_start_weights
    }
    mean_of_workers_delta_l2 = math.sqrt(
        math.fsum(value * value for value in mean_worker_delta.values())
    )
    worker_delta_l2 = [
        math.sqrt(
            math.fsum(
                (worker_weights_i[name] - round_start_weights[name]) ** 2
                for name in round_start_weights
            )
        )
        for worker_weights_i in worker_weights
    ]
    mean_worker_delta_l2 = (
        math.fsum(worker_delta_l2) / len(worker_delta_l2)
        if worker_delta_l2
        else 0.0
    )
    if mean_worker_delta_l2 <= 1e-9:
        update_agreement_ratio = 1.0 if mean_of_workers_delta_l2 <= 1e-9 else 0.0
    else:
        update_agreement_ratio = max(
            0.0,
            min(1.0, mean_of_workers_delta_l2 / mean_worker_delta_l2),
        )
    update_count = sum(count for _, count in worker_td_errors)
    mean_abs_td_error = (
        math.fsum(value * count for value, count in worker_td_errors) / update_count
        if update_count
        else 0.0
    )
    return {
        "mean_abs_weight_delta": (
            math.fsum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
        ),
        "max_abs_weight_delta": max(abs_deltas, default=0.0),
        "canonical_delta_l2": canonical_delta_l2,
        "genetic_child_delta_l2": canonical_delta_l2,
        "mean_of_workers_delta_l2": mean_of_workers_delta_l2,
        "relative_weight_delta_l2": (
            canonical_delta_l2 / starting_weight_l2
            if starting_weight_l2 > RELATIVE_WEIGHT_NORM_THRESHOLD
            else None
        ),
        "mean_worker_delta_l2": mean_worker_delta_l2,
        "update_agreement_ratio": update_agreement_ratio,
        "mean_abs_td_error": mean_abs_td_error,
        "max_abs_q": max_abs_q,
        "weight_deltas": weight_deltas,
    }


def _append_history_record(history_path: Optional[Path], record: dict) -> None:
    if history_path is None:
        return
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, allow_nan=False) + "\n")


def _print_learning_progress(metrics: dict) -> None:
    relative_delta = metrics["relative_weight_delta_l2"]
    merge_strategy = metrics.get("merge_strategy")
    child_delta_label = (
        "Genetic child delta L2"
        if merge_strategy == "genetic"
        else "Canonical merged delta L2"
    )
    relative_delta_text = (
        f"{relative_delta:.6f}"
        if relative_delta is not None
        else "N/A (near-zero starting norm)"
    )
    print(
        "LEARNING PROGRESS\n"
        "------------------------------------------------\n"
        f"Mean abs weight delta:  {metrics['mean_abs_weight_delta']:.4f}\n"
        f"Max abs weight delta:   {metrics['max_abs_weight_delta']:.4f}\n"
        f"Canonical delta L2:     {metrics['canonical_delta_l2']:.4f}\n"
        f"{child_delta_label}: {metrics['canonical_delta_l2']:.4f}\n"
        f"Mean-of-workers delta L2: {metrics['mean_of_workers_delta_l2']:.4f}\n"
        f"Relative delta L2:      {relative_delta_text}\n\n"
        f"Mean worker delta L2:   {metrics['mean_worker_delta_l2']:.4f}\n"
        f"Update agreement:       {metrics['update_agreement_ratio']:.3f}\n\n"
        f"Mean |TD error|:         {metrics['mean_abs_td_error']:.3f}\n"
        f"Max |Q|:                {metrics['max_abs_q']:.3f}\n"
        "------------------------------------------------",
        flush=True,
    )


def _print_mixed_batch_summary(
    episode_variants: tuple[int, ...],
    results: list[tuple[int, object]],
    newest_variant: int,
    required_newest_wins: int,
    *,
    phase: str = "variant",
    item_names: Optional[dict[int, str]] = None,
    show_newest: bool = True,
) -> None:
    planned = _training_distribution(episode_variants)
    wins = {
        variant: sum(
            result_variant == variant
            and result.outcome == "WON"
            and getattr(result, "error", None) is None
            for result_variant, result in results
        )
        for variant in planned
    }
    print("\nTRAINING BATCH RESULTS", flush=True)
    item_prefix = "Drill" if phase == "drill" else "Variant"
    for variant in sorted(planned):
        print(
            f"{item_prefix} {variant}: {wins[variant]}/{planned[variant]} wins",
            flush=True,
        )
    if show_newest:
        print(
            f"Newest {item_prefix.lower()} {newest_variant}: "
            f"{item_names.get(newest_variant, '') + ' ' if item_names else ''}"
            f"{wins.get(newest_variant, 0)}/"
            f"{planned.get(newest_variant, 0)} wins; "
            f"need {required_newest_wins} to unlock the next variant",
            flush=True,
        )
    print(
        f"Total wins: {sum(wins.values())}/{len(episode_variants)} | "
        f"Bombs: {sum(result.bombs_placed for _, result in results)} | "
        f"Average ticks: "
        f"{(sum(result.ticks for _, result in results) / len(results)) if results else 0.0:.1f}",
        flush=True,
    )


def _print_batch_learning_summary(
    metrics: dict,
    valid_workers: int,
    attempted_workers: int,
    merge_strategy: str,
) -> None:
    """Print one compact aggregate for the weight updates in a batch."""
    print("Learning summary:", flush=True)
    if merge_strategy == "genetic":
        print(
            f"Genetic merge: {valid_workers}/{attempted_workers} valid workers",
            flush=True,
        )
    elif merge_strategy == "none":
        print("Weight merge: disabled (frozen evaluation)", flush=True)
    else:
        print(
            f"Weight merge: {merge_strategy} "
            f"({valid_workers}/{attempted_workers} valid workers)",
            flush=True,
        )
    print(
        f"Mean |TD|: {metrics['mean_abs_td_error']:.3f} | "
        f"Canonical delta L2: {metrics['canonical_delta_l2']:.4f} | "
        f"Mean worker delta L2: {metrics['mean_worker_delta_l2']:.4f} | "
        f"Update agreement: {metrics['update_agreement_ratio']:.3f}",
        flush=True,
    )


def _make_round_tasks(
    challenge: Challenge | Drill,
    trial_numbers: tuple[int, ...],
    batch_number: int,
    first_trial_index: int,
    trial_count: int,
    round_number: int,
    base_seed: int,
    starting_weights: dict[str, float],
    guis: int,
    q_contributions_diagnostic: bool,
    learning_parameters: Optional[tuple[float, float, float]] = None,
    *,
    no_training: bool = False,
    evaluation_timeout_seconds: Optional[float] = None,
    phase: str = "variant",
    trial_challenges: Optional[tuple[Challenge | Drill, ...]] = None,
    worker_slots: Optional[int] = None,
    worker_details: bool = False,
) -> tuple[ParallelTrainingTask, ...]:
    from team01.agent.evaluation import Q_FEATURE_VERSION, q_feature_names

    if learning_parameters is None:
        learning_parameters = _default_learning_parameters()
    alpha, gamma, epsilon = learning_parameters
    feature_names = tuple(q_feature_names())
    tasks = []
    for slot, trial_number in enumerate(trial_numbers, start=1):
        trial_index = first_trial_index + slot - 1
        task_challenge = (
            trial_challenges[slot - 1]
            if trial_challenges is not None
            else challenge
        )
        worker_id = (
            (slot - 1) % worker_slots + 1
            if worker_slots is not None
            else slot
        )
        tasks.append(
            ParallelTrainingTask(
                worker_id=worker_id,
                trial_index=trial_index,
                trial_number=trial_number,
                trial_count=trial_count,
                batch_number=batch_number,
                round_number=round_number,
                seed=base_seed + trial_index - 1,
                challenge=task_challenge,
                feature_version=Q_FEATURE_VERSION,
                feature_names=feature_names,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                starting_weights=dict(starting_weights),
                displayed=slot <= guis,
                q_contributions_diagnostic=(
                    q_contributions_diagnostic and slot <= guis
                ),
                no_training=no_training,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
                phase=phase,
                worker_details=worker_details,
            )
        )
    return tuple(tasks)


# ---------------------------------------------------------------------------
# Weight synchronization and genetic crossover

def _worker_result_validation_error(
    result: ParallelTrainingResult,
    expected_features: tuple[str, ...],
    expected_version: int,
    expected_weights: Optional[dict[str, float]] = None,
) -> Optional[str]:
    expected = set(expected_features)
    if result.error:
        return result.error
    if result.feature_version != expected_version:
        return (
            f"feature version {result.feature_version} does not match "
            f"{expected_version}"
        )
    if set(result.feature_names) != expected:
        return "feature names do not match the active schema"
    if set(result.weights) != expected:
        return "weight names do not match the active schema"
    if result.outcome in {
        "CANCELLED",
        "WORKER_ERROR",
        "NON_FINITE",
        "EVALUATION_TIMEOUT",
    }:
        return f"worker outcome {result.outcome} is not mergeable"
    if result.non_finite_q_values or result.non_finite_q_fallbacks:
        return "worker reported non-finite Q values or fallbacks"
    if not _weights_are_finite(result.weights):
        return "worker weights contain non-finite values"
    if expected_weights is not None and result.weights != expected_weights:
        return "worker changed weights during evaluation"
    return None


def _validate_worker_results(
    round_start_weights: dict[str, float],
    results: list[ParallelTrainingResult],
    feature_names: tuple[str, ...],
    feature_version: int,
    *,
    require_unchanged_weights: bool = False,
) -> list[ParallelTrainingResult]:
    if set(round_start_weights) != set(feature_names):
        raise ValueError("Round snapshot does not match the active feature schema")
    if not _weights_are_finite(round_start_weights):
        raise ValueError("Round snapshot contains non-finite weights")

    valid_results = []
    for result in results:
        reason = _worker_result_validation_error(
            result,
            feature_names,
            feature_version,
            expected_weights=(
                round_start_weights if require_unchanged_weights else None
            ),
        )
        if reason is None:
            valid_results.append(result)
        else:
            result.error = reason
    return valid_results


def _average_worker_deltas(
    round_start_weights: dict[str, float],
    results: list[ParallelTrainingResult],
    feature_names: tuple[str, ...],
    feature_version: int,
) -> tuple[dict[str, float], list[ParallelTrainingResult]]:
    valid_results = _validate_worker_results(
        round_start_weights,
        results,
        feature_names,
        feature_version,
    )

    if not valid_results:
        return dict(round_start_weights), []

    merged = {
        name: round_start_weights[name]
        + math.fsum(result.weights[name] - round_start_weights[name] for result in valid_results)
        / len(valid_results)
        for name in feature_names
    }
    if not _weights_are_finite(merged):
        for result in valid_results:
            result.error = "mean worker delta produced non-finite canonical weights"
        return dict(round_start_weights), []
    return merged, valid_results


def _genetic_worker_crossover(
    round_start_weights: dict[str, float],
    results: list[ParallelTrainingResult],
    feature_names: tuple[str, ...],
    feature_version: int,
    episode_variants: tuple[int, ...],
    seed: int,
    round_number: int,
    rank_power: float = GENETIC_RANK_POWER,
    trial_numbers: Optional[tuple[int, ...]] = None,
) -> tuple[dict[str, float], list[ParallelTrainingResult], list[dict], dict[int, float]]:
    if rank_power <= 0 or not math.isfinite(rank_power):
        raise ValueError("genetic rank power must be positive and finite")

    valid_results = _validate_worker_results(
        round_start_weights,
        results,
        feature_names,
        feature_version,
    )
    if trial_numbers is None:
        trial_numbers = tuple(range(1, len(episode_variants) + 1))
    if len(trial_numbers) != len(episode_variants):
        raise ValueError("trial numbers and variant assignments must have equal lengths")
    assigned_variants = dict(zip(trial_numbers, episode_variants))
    candidates_by_variant: dict[int, list[ParallelTrainingResult]] = {}
    for result in valid_results:
        expected_variant = assigned_variants.get(result.trial_number)
        if expected_variant is None or result.stage_number != expected_variant:
            result.error = "worker variant does not match its round assignment"
            continue
        candidates_by_variant.setdefault(expected_variant, []).append(result)
    valid_results = [
        result for candidates in candidates_by_variant.values() for result in candidates
    ]
    if not valid_results:
        return dict(round_start_weights), [], [], {}

    outcome_rank = {"LOST": 0, "TIMEOUT": 1, "WON": 2}
    ranked_candidates = []
    variant_fitness_totals = {}
    for variant in sorted(candidates_by_variant):
        candidates = sorted(
            candidates_by_variant[variant],
            key=lambda result: (
                outcome_rank.get(result.outcome, -1),
                result.total_reward,
                -result.ticks if result.outcome == "WON" else 0,
                result.worker_id,
                result.trial_index,
            ),
        )
        fitness_weights = [rank**rank_power for rank in range(1, len(candidates) + 1)]
        variant_fitness_totals[variant] = math.fsum(fitness_weights)
        ranked_candidates.extend(
            (result, variant, rank, fitness_weight)
            for rank, (result, fitness_weight) in enumerate(
                zip(candidates, fitness_weights),
                start=1,
            )
        )

    requested_variant_counts = _training_distribution(episode_variants)
    represented_variants = set(candidates_by_variant)
    represented_episode_count = sum(
        requested_variant_counts[variant] for variant in represented_variants
    )
    variant_shares = {
        variant: requested_variant_counts[variant] / represented_episode_count
        for variant in sorted(represented_variants)
    }
    candidate_metadata = []
    for result, variant, rank, fitness_weight in ranked_candidates:
        within_variant_probability = fitness_weight / variant_fitness_totals[variant]
        candidate_metadata.append(
            {
                "worker": result.worker_id,
                "trial_index": result.trial_index,
                "variant": variant,
                "outcome": result.outcome,
                "reward": float(result.total_reward),
                "ticks": result.ticks,
                "rank": rank,
                "fitness_weight": fitness_weight,
                "variant_share": variant_shares[variant],
                "selection_probability": (
                    variant_shares[variant] * within_variant_probability
                ),
                "genes_selected": 0,
                "weights": result.weights,
            }
        )

    feature_count = len(feature_names)
    exact_gene_counts = [
        item["selection_probability"] * feature_count
        for item in candidate_metadata
    ]
    gene_counts = [math.floor(count) for count in exact_gene_counts]
    remaining_genes = feature_count - sum(gene_counts)
    remainder_order = sorted(
        range(len(candidate_metadata)),
        key=lambda index: (
            -(exact_gene_counts[index] - gene_counts[index]),
            candidate_metadata[index]["variant"],
            candidate_metadata[index]["worker"],
            candidate_metadata[index]["trial_index"],
        ),
    )
    for index in remainder_order[:remaining_genes]:
        gene_counts[index] += 1
    for item, gene_count in zip(candidate_metadata, gene_counts):
        item["genes_selected"] = gene_count

    feature_order = list(feature_names)
    crossover_seed = seed ^ (round_number * 0x9E3779B1) ^ GENETIC_CROSSOVER_OFFSET
    random.Random(crossover_seed).shuffle(feature_order)
    parent_indices = [
        index
        for index, gene_count in enumerate(gene_counts)
        for _ in range(gene_count)
    ]
    child = {
        feature: candidate_metadata[parent_index]["weights"][feature]
        for feature, parent_index in zip(feature_order, parent_indices)
    }
    for item in candidate_metadata:
        item.pop("weights")
    if not _weights_are_finite(child):
        for result in valid_results:
            result.error = "genetic crossover produced non-finite canonical weights"
        return dict(round_start_weights), [], [], {}
    return child, valid_results, candidate_metadata, variant_shares


def _print_genetic_selection(
    candidate_metadata: list[dict],
    phase: str = "variant",
    *,
    detailed: bool = True,
) -> None:
    if not detailed:
        return
    item_label = "Drill" if phase in {"drill", "drill_refresh"} else "Variant"
    print("GENETIC WEIGHT SELECTION", flush=True)
    print(f"Worker  {item_label:<7} Outcome  Reward    Rank  Probability  Genes", flush=True)
    for candidate in candidate_metadata:
        print(
            f"W{candidate['worker']:<6} {item_label[0]}{candidate['variant']:<7} "
            f"{candidate['outcome']:<8} {candidate['reward']:>8.1f} "
            f"{candidate['rank']:>4} "
            f"{candidate['selection_probability']:>12.3f} "
            f"{candidate['genes_selected']:>5}",
            flush=True,
        )
    print("Variant contribution shares:", flush=True)
    for variant, share in sorted(
        {
            item["variant"]: item["variant_share"]
            for item in candidate_metadata
        }.items()
    ):
        print(f"{item_label[0]}{variant}: {share:.3f}", flush=True)
    print("Merge strategy: genetic", flush=True)


def _print_worker_result(
    result: ParallelTrainingResult, *, detailed: bool = True
) -> None:
    if not detailed:
        if result.error or result.outcome in {
            "CANCELLED",
            "WORKER_ERROR",
            "NON_FINITE",
            "EVALUATION_TIMEOUT",
        }:
            print(
                f"WARNING: W{result.worker_id} trial {result.trial_index} "
                f"({result.phase} {result.stage_number}) {result.outcome}: "
                f"{result.error or 'worker result is not mergeable'}",
                flush=True,
            )
            for line in result.diagnostic_output.splitlines():
                print(f"  {line}", flush=True)
        return

    item_label = "Drill" if result.phase.startswith("drill") else "Stage"
    context = (
        f"[W{result.worker_id} | Trial {result.trial_index} "
        f"(batch {result.trial_number}/{result.trial_count}) | "
        f"{item_label} {result.stage_number}"
        f" | Seed {result.seed}]"
    )
    print(
        f"{context}\n"
        f"Outcome: {result.outcome}\n"
        f"Ticks: {result.ticks}\n"
        f"Reward: {result.total_reward:.0f}\n"
        f"Bombs: {result.bombs_placed}\n"
        f"Mean |TD|: {result.mean_abs_td_error:.3f}\n"
        f"Max |Q|: {result.max_abs_q:.3f}",
        flush=True,
    )
    if result.error:
        print(f"{context} Details: {result.error}", flush=True)
    if result.diagnostic_output:
        diagnostic_prefix = (
            f"[W{result.worker_id} | Trial {result.trial_index} | "
            f"{item_label} {result.stage_number}]"
        )
        for line in result.diagnostic_output.splitlines():
            print(f"{diagnostic_prefix} {line}", flush=True)


def _failed_worker_result(
    task: ParallelTrainingTask,
    error: BaseException,
) -> ParallelTrainingResult:
    return ParallelTrainingResult(
        worker_id=task.worker_id,
        trial_index=task.trial_index,
        outcome="WORKER_ERROR",
        weights={},
        feature_version=None,
        feature_names=(),
        trial_number=task.trial_number,
        trial_count=task.trial_count,
        round_number=task.round_number,
        stage_number=task.challenge.number,
        seed=task.seed,
        displayed=task.displayed,
        error=f"{type(error).__name__}: {error}",
        phase=task.phase,
    )


def _run_parallel_round(
    tasks: tuple[ParallelTrainingTask, ...],
    *,
    max_workers: Optional[int] = None,
    progress_label: Optional[str] = None,
) -> list[ParallelTrainingResult]:
    if not tasks:
        return []
    show_worker_details = any(task.worker_details for task in tasks)
    executor = ProcessPoolExecutor(max_workers=max_workers or len(tasks))
    futures = {}
    pending = set()
    results = []
    interrupted = False
    try:
        for task in tasks:
            futures[executor.submit(_parallel_training_worker, task)] = task
        pending = set(futures)
        while pending:
            done, pending = wait(
                pending,
                timeout=PARALLEL_HEARTBEAT_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            if not done and show_worker_details:
                completed = len(results)
                print(
                    f"[{progress_label or f'Round {tasks[0].round_number}'}] running... "
                    f"completed={completed}/{len(tasks)} active={len(pending)}",
                    flush=True,
                )
                continue
            if not done:
                continue

            for future in done:
                task = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = _failed_worker_result(task, error)
                results.append(result)
                _print_worker_result(result, detailed=show_worker_details)
        return results
    except KeyboardInterrupt:
        interrupted = True
        for future in pending:
            future.cancel()
        processes = tuple((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    finally:
        if not interrupted:
            executor.shutdown(wait=True)


def _run_parallel_drill_refresh(
    drills: tuple[Drill, ...],
    refresh_trials: int,
    workers: int,
    seed: int,
    refresh_number: int,
    batch_number: int,
    round_counter: int,
    guis: int,
    q_contributions_diagnostic: bool,
    learning_parameters: tuple[float, float, float],
    weights_state: list[dict[str, float]],
    weights_path: Path,
    save_agent,
    checkpoint_state: list[bool],
    summary: TrainingSummary,
    merge_strategy: str,
    history_path: Optional[Path],
    worker_details: bool = False,
) -> tuple[int, dict[int, dict[str, float | int]]]:
    """Replay every drill in parallel and return aggregate results by drill."""
    from team01.agent.evaluation import Q_FEATURE_VERSION, q_feature_names

    feature_names = tuple(q_feature_names())
    drill_results: dict[int, dict[str, float | int]] = {}
    print("DRILL REFRESH SCHEDULE", flush=True)
    for drill in drills:
        print(f"D{drill.number} - {drill.name}: {refresh_trials} episodes", flush=True)

    for drill in drills:
        totals = {
            "attempts": 0,
            "wins": 0,
            "losses": 0,
            "reward_total": 0.0,
            "ticks_total": 0.0,
            "mean_abs_td_total": 0.0,
            "canonical_delta_l2_total": 0.0,
            "valid_results": 0,
        }
        for trial_numbers in _trial_rounds(refresh_trials, workers):
            round_counter += 1
            round_start_weights = dict(weights_state[0])
            refresh_seed = (
                seed
                ^ DRILL_REFRESH_SEED_OFFSET
                ^ (refresh_number * 0x9E3779B1)
                ^ (drill.number * 0x85EBCA77)
            )
            tasks = _make_round_tasks(
                challenge=drill,
                trial_numbers=trial_numbers,
                batch_number=batch_number,
                first_trial_index=summary.completed_trials + 1,
                trial_count=refresh_trials,
                round_number=round_counter,
                base_seed=refresh_seed,
                starting_weights=round_start_weights,
                guis=guis,
                q_contributions_diagnostic=q_contributions_diagnostic,
                learning_parameters=learning_parameters,
                phase="drill_refresh",
                trial_challenges=(drill,) * len(trial_numbers),
                worker_slots=workers,
                worker_details=worker_details,
            )
            summary.current_round = round_counter
            results = _run_parallel_round(tasks)
            summary.current_round = 0
            summary.completed_trials += len(tasks)
            totals["attempts"] += len(tasks)
            if merge_strategy == "mean":
                merged_weights, valid_results = _average_worker_deltas(
                    round_start_weights,
                    results,
                    feature_names,
                    Q_FEATURE_VERSION,
                )
                candidate_metadata = []
                variant_shares = {}
            else:
                (
                    merged_weights,
                    valid_results,
                    candidate_metadata,
                    variant_shares,
                ) = _genetic_worker_crossover(
                    round_start_weights,
                    results,
                    feature_names,
                    Q_FEATURE_VERSION,
                    (drill.number,) * len(trial_numbers),
                    refresh_seed,
                    round_counter,
                    trial_numbers=trial_numbers,
                )
                for candidate in candidate_metadata:
                    candidate["phase"] = "drill_refresh"
                    candidate["curriculum_item"] = drill.number

            for result in results:
                if result.error:
                    print(
                        f"[D{drill.number} | W{result.worker_id} | "
                        f"Trial {result.trial_number}] Not merged: {result.error}",
                        flush=True,
                    )
            if valid_results:
                totals["wins"] += sum(
                    result.outcome == "WON" for result in valid_results
                )
                totals["losses"] += sum(
                    result.outcome in {"LOST", "TIMEOUT"}
                    for result in valid_results
                )
                totals["reward_total"] += math.fsum(
                    result.total_reward for result in valid_results
                )
                totals["ticks_total"] += math.fsum(
                    result.ticks for result in valid_results
                )
                totals["mean_abs_td_total"] += math.fsum(
                    result.mean_abs_td_error for result in valid_results
                )
                totals["valid_results"] += len(valid_results)
                weights_state[0] = merged_weights
                _save_weights(save_agent, weights_state[0], weights_path)
                checkpoint_state[0] = True
                summary.weights_saved = True
                if merge_strategy == "genetic":
                    _print_genetic_selection(
                        candidate_metadata,
                        "drill_refresh",
                        detailed=worker_details,
                    )
            else:
                print(
                    f"WARNING: No valid candidates in drill refresh D{drill.number} round; "
                    "canonical weights retained.",
                    flush=True,
                )

            diagnostics = _weight_update_diagnostics(
                round_start_weights,
                [result.weights for result in valid_results],
                merged_weights if valid_results else round_start_weights,
                [
                    (result.mean_abs_td_error, result.update_count)
                    for result in valid_results
                ],
                max((result.max_abs_q for result in valid_results), default=0.0),
            )
            diagnostics["merge_strategy"] = merge_strategy if valid_results else "none"
            if merge_strategy != "genetic":
                diagnostics["genetic_child_delta_l2"] = None
            totals["canonical_delta_l2_total"] += diagnostics["canonical_delta_l2"]
            if worker_details:
                _print_learning_progress(diagnostics)
            _append_history_record(
                history_path,
                {
                    "record_type": "training_round",
                    "batch": batch_number,
                    "round": round_counter,
                    "phase": "drill_refresh",
                    "curriculum_item": drill.number,
                    "unlocked_drills": [item.number for item in drills],
                    "episode_items": [drill.number] * len(trial_numbers),
                    "episode_drills": [drill.number] * len(trial_numbers),
                    "training_enabled": True,
                    "training_mode": "drill_refresh",
                    **diagnostics,
                    "weights": dict(weights_state[0]),
                    "genetic_candidates": candidate_metadata,
                    "item_contribution_shares": {
                        str(item): share for item, share in sorted(variant_shares.items())
                    },
                },
            )

        valid_count = int(totals["valid_results"])
        attempts = int(totals["attempts"])
        wins = int(totals["wins"])
        drill_results[drill.number] = {
            "wins": wins,
            "losses": int(totals["losses"]),
            "attempts": attempts,
            "mean_reward": (
                float(totals["reward_total"]) / valid_count if valid_count else 0.0
            ),
            "mean_ticks": (
                float(totals["ticks_total"]) / valid_count if valid_count else 0.0
            ),
            "mean_abs_td_error": (
                float(totals["mean_abs_td_total"]) / valid_count
                if valid_count
                else 0.0
            ),
            "mean_canonical_delta_l2": (
                float(totals["canonical_delta_l2_total"])
                / max(1, math.ceil(refresh_trials / workers))
            ),
        }
        print(
            f"Drill refresh D{drill.number} results: {wins} wins, "
            f"{drill_results[drill.number]['losses']} losses / {attempts} attempts | "
            f"Mean reward: {drill_results[drill.number]['mean_reward']:.1f} | "
            f"Mean ticks: {drill_results[drill.number]['mean_ticks']:.1f} | "
            f"Mean |TD|: {drill_results[drill.number]['mean_abs_td_error']:.3f} | "
            f"Mean canonical delta L2: "
            f"{drill_results[drill.number]['mean_canonical_delta_l2']:.4f}",
            flush=True,
        )
        if wins == 0:
            print(
                f"WARNING: Drill refresh D{drill.number} produced 0/{attempts} wins. "
                "The current policy may have substantially forgotten this skill.",
                flush=True,
            )
    return round_counter, drill_results


def _run_sequential_drill_refresh(
    *,
    drills: tuple[Drill, ...],
    refresh_trials: int,
    seed: int,
    refresh_number: int,
    batch_number: int,
    round_counter: int,
    weights: Optional[dict[str, float]],
    save_agent,
    weights_path: Path,
    summary: TrainingSummary,
    agent_factory: Callable,
    game_factory: Callable,
    episode_runner: Callable,
    display: bool,
    q_contributions_diagnostic: bool,
    no_training: bool,
    history_path: Optional[Path],
    worker_details: bool = False,
    agent_type: str = "q",
    dqn_updates_per_batch: int = DQN_UPDATES_PER_BATCH,
    workers: int = 1,
    guis: int = 0,
) -> tuple[int, Optional[dict[str, float]], object, dict[int, dict[str, float | int]]]:
    """Replay every drill sequentially and return updated weights and aggregates."""
    if no_training:
        raise ValueError("drill refresh cannot run in no-training mode")

    drill_results: dict[int, dict[str, float | int]] = {}
    for drill in drills:
        wins = 0
        losses = 0
        reward_total = 0.0
        ticks_total = 0
        td_total = 0.0
        canonical_delta_total = 0.0
        dqn_episode_results = []
        parallel_drill_results = None
        drill_batch_failed = False
        if agent_type == "dqn" and workers > 1:
            rollout_agent = agent_factory()
            assignments = tuple(
                (
                    drill,
                    seed
                    ^ DRILL_REFRESH_SEED_OFFSET
                    ^ (refresh_number * 0x9E3779B1)
                    ^ (drill.number * 0x85EBCA77)
                    ^ trial_number,
                )
                for trial_number in range(1, refresh_trials + 1)
            )
            parallel_drill_results = _run_dqn_rollout_batch(
                agent=rollout_agent,
                assignments=assignments,
                workers=workers,
                guis=guis,
                batch_number=batch_number,
                epsilon=rollout_agent.epsilon,
                worker_details=worker_details,
            )
            _, missing_items = _dqn_rollout_coverage(
                (drill.number,) * refresh_trials,
                parallel_drill_results,
            )
            failed_results = [
                result
                for result in parallel_drill_results
                if not _dqn_result_is_numerically_stable(result)
            ]
            for result in failed_results:
                print(
                    f"WARNING: DQN drill-refresh worker {result.worker_id} trial "
                    f"{result.trial_index} D{result.challenge_number} "
                    f"{result.outcome}: {result.error or 'invalid rollout result'}",
                    flush=True,
                )
            if missing_items:
                summary.stopped_reason = (
                    f"DQN drill refresh D{drill.number} had no valid rollout "
                    "results; no replay updates or checkpoint save performed"
                )
                drill_batch_failed = True
        for trial_number in range(1, refresh_trials + 1):
            if drill_batch_failed:
                break
            round_counter += 1
            start_weights = dict(weights) if weights is not None else None
            episode_seed = (
                assignments[trial_number - 1][1]
                if parallel_drill_results is not None
                else (
                    seed
                    ^ DRILL_REFRESH_SEED_OFFSET
                    ^ (refresh_number * 0x9E3779B1)
                    ^ (drill.number * 0x85EBCA77)
                    ^ trial_number
                )
            )
            if parallel_drill_results is not None:
                active_agent = agent_factory()
                game = None
                result = parallel_drill_results[trial_number - 1]
            else:
                game, active_agent = game_factory(
                    drill,
                    episode_seed,
                    weights,
                    agent_factory,
                )
            if agent_type == "dqn" and parallel_drill_results is None:
                active_agent.set_rollout(
                    epsilon=active_agent.epsilon,
                    collect_experience=True,
                )
            elif agent_type == "q":
                active_agent.set_learning(False, epsilon=active_agent.epsilon)
            active_agent.q_contributions_diagnostic = q_contributions_diagnostic
            updates_before = getattr(active_agent, "td_update_count", 0)
            label = (
                f"Drill refresh {refresh_number} | D{drill.number} | "
                f"Episode {trial_number}/{refresh_trials} | "
                f"Global episode {summary.completed_trials + 1}"
            )
            if parallel_drill_results is None:
                result = episode_runner(
                    game,
                    active_agent,
                    summary.completed_trials + 1,
                    label,
                    display,
                )
            if agent_type == "dqn":
                stable = _dqn_result_is_numerically_stable(result)
                if stable:
                    dqn_episode_results.append(result)
                    if parallel_drill_results is not None:
                        summary.completed_trials += 1
                elif parallel_drill_results is not None:
                    _append_history_record(
                        history_path,
                        {
                            "record_type": "training_round",
                            "agent": "dqn",
                            "batch": batch_number,
                            "round": round_counter,
                            "phase": "drill_refresh",
                            "curriculum_item": drill.number,
                            "worker_id": result.worker_id,
                            "trial_index": result.trial_index,
                            "seed": result.seed,
                            "outcome": result.outcome,
                            "error": result.error,
                        },
                    )
                    continue
            else:
                stable = _result_is_numerically_stable(result, active_agent.weights)
            if parallel_drill_results is None:
                summary.completed_trials += 1
            if agent_type == "q":
                if stable:
                    weights = dict(active_agent.weights)
                    save_agent = active_agent
                    _save_weights(save_agent, weights, weights_path)
                    summary.weights_saved = True
                else:
                    weights = start_weights
                    active_agent.weights = dict(start_weights)
                    print(
                        f"WARNING: Drill refresh D{drill.number} returned unstable weights; "
                        "retaining the last canonical vector.",
                        flush=True,
                    )

            wins += result.outcome == "WON" and stable
            losses += result.outcome in {"LOST", "TIMEOUT"} and stable
            reward_total += result.total_reward
            ticks_total += result.ticks
            td_error = getattr(result, "mean_abs_td_error", 0.0)
            update_count = getattr(
                result, "td_update_count", getattr(active_agent, "td_update_count", 0)
            )
            td_total += td_error
            if agent_type == "q":
                diagnostics = _weight_update_diagnostics(
                    start_weights,
                    [dict(active_agent.weights)] if stable else [],
                    weights,
                    [(td_error, update_count)] if stable else [],
                    getattr(result, "max_abs_q", 0.0),
                )
                diagnostics["merge_strategy"] = "single_worker"
                diagnostics["genetic_child_delta_l2"] = None
                canonical_delta_total += diagnostics["canonical_delta_l2"]
                if worker_details:
                    _print_learning_progress(diagnostics)
                history_record = {
                    "weights": dict(weights),
                    **diagnostics,
                }
            else:
                history_record = {
                    "outcome": result.outcome,
                    "ticks": result.ticks,
                    "total_reward": result.total_reward,
                    "bombs_placed": result.bombs_placed,
                    "worker_id": getattr(result, "worker_id", 0),
                    "trial_index": getattr(result, "trial_index", trial_number),
                    "seed": getattr(result, "seed", episode_seed),
                    "transitions_collected": len(
                        getattr(result, "transitions", ())
                    ),
                }
            _append_history_record(
                history_path,
                {
                    "record_type": "training_round",
                    "agent": agent_type,
                    "batch": batch_number,
                    "round": round_counter,
                    "phase": "drill_refresh",
                    "curriculum_item": drill.number,
                    "episode_items": [drill.number],
                    "episode_drills": [drill.number],
                    "training_enabled": True,
                    "training_mode": "drill_refresh",
                    **history_record,
                },
            )
        if agent_type == "dqn" and dqn_episode_results:
            active_agent.set_learning(
                no_training=False,
                epsilon=active_agent.epsilon,
            )
            refresh_metrics = _consume_dqn_episode_results(
                active_agent,
                dqn_episode_results,
                dqn_updates_per_batch,
            )
            assigned_refresh_episodes = (
                len(parallel_drill_results)
                if parallel_drill_results is not None
                else len(dqn_episode_results)
            )
            refresh_metrics.update(
                worker_limit=min(workers, assigned_refresh_episodes),
                episodes_assigned=(
                    assigned_refresh_episodes
                ),
                episodes_valid=len(dqn_episode_results),
                episode_errors=(
                    len(parallel_drill_results) - len(dqn_episode_results)
                    if parallel_drill_results is not None
                    else 0
                ),
            )
            _print_dqn_batch_summary(refresh_metrics)
            if dqn_episode_results:
                active_agent.save_checkpoint(weights_path)
                summary.weights_saved = True
            if refresh_metrics["mean_abs_td_error"] is not None:
                mean_td_error = refresh_metrics["mean_abs_td_error"]
            else:
                mean_td_error = 0.0
        else:
            refresh_metrics = None
            mean_td_error = td_total / refresh_trials
        drill_results[drill.number] = {
            "wins": int(wins),
            "losses": int(losses),
            "attempts": refresh_trials,
            "mean_reward": reward_total / refresh_trials,
            "mean_ticks": ticks_total / refresh_trials,
            "mean_abs_td_error": mean_td_error,
            **(
                {
                    "transitions_collected": refresh_metrics[
                        "transitions_collected"
                    ],
                    "optimizer_steps": refresh_metrics["updates_this_batch"],
                    "episode_errors": (
                        len(parallel_drill_results) - len(dqn_episode_results)
                        if parallel_drill_results is not None
                        else 0
                    ),
                }
                if refresh_metrics is not None
                else {
                    "mean_canonical_delta_l2": (
                        canonical_delta_total / refresh_trials
                    )
                }
            ),
        }
        summary_text = (
            f"Drill refresh D{drill.number} results: {wins} wins, "
            f"{losses} losses / {refresh_trials} attempts | "
            f"Mean reward: {reward_total / refresh_trials:.1f} | "
            f"Mean ticks: {ticks_total / refresh_trials:.1f} | "
            f"Mean |TD|: {mean_td_error:.3f}"
        )
        if refresh_metrics is not None:
            summary_text += (
                f" | Transitions: {refresh_metrics['transitions_collected']} "
                f"| Optimizer steps: {refresh_metrics['updates_this_batch']}"
            )
        else:
            summary_text += (
                f" | Mean canonical delta L2: "
                f"{canonical_delta_total / refresh_trials:.4f}"
            )
        print(summary_text, flush=True)
        if drill_batch_failed:
            drill_results[drill.number]["batch_rejected"] = True
            break
        if wins == 0:
            print(
                f"WARNING: Drill refresh D{drill.number} produced 0/{refresh_trials} wins. "
                "The current policy may have substantially forgotten this skill.",
                flush=True,
            )
    return round_counter, weights, save_agent, drill_results


# ---------------------------------------------------------------------------
# Frozen evaluation and curriculum decisions

def _all_variants_pass(
    unlocked_variants: tuple[int, ...] | list[int],
    evaluation_win_rates: dict[int, float],
    valid_counts: dict[int, int],
    eval_trials: int,
) -> bool:
    """Check the final win-rate gate for every existing variant."""
    return len(unlocked_variants) == len(progression()) and all(
        valid_counts.get(variant, 0) == eval_trials
        and evaluation_win_rates.get(variant, 0.0) >= FINAL_WIN_RATE_THRESHOLD
        for variant in range(1, len(progression()) + 1)
    )


def _timed_out_evaluation_result(task: ParallelTrainingTask) -> ParallelTrainingResult:
    return ParallelTrainingResult(
        worker_id=task.worker_id,
        trial_index=task.trial_index,
        outcome="EVALUATION_TIMEOUT",
        weights=dict(task.starting_weights),
        feature_version=task.feature_version,
        feature_names=task.feature_names,
        trial_number=task.trial_number,
        trial_count=task.trial_count,
        round_number=task.round_number,
        stage_number=task.challenge.number,
        seed=task.seed,
        error="frozen evaluation worker wave exceeded its wall-clock deadline",
    )


def _terminate_executor_processes(executor: ProcessPoolExecutor) -> None:
    processes = tuple((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


def _run_parallel_evaluation_wave(
    tasks: tuple[ParallelTrainingTask, ...],
    timeout_seconds: float,
    variant: int,
    completed_before_wave: int,
    total_trials: int,
) -> tuple[list[ParallelTrainingResult], int]:
    if not tasks:
        return [], 0
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("evaluation wave timeout must be positive and finite")
    show_worker_details = any(task.worker_details for task in tasks)
    executor = ProcessPoolExecutor(max_workers=len(tasks))
    futures = {}
    results = []
    pending = set()
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    timed_out = False
    try:
        for task in tasks:
            futures[executor.submit(_parallel_training_worker, task)] = task
        pending = set(futures)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            done, pending = wait(
                pending,
                timeout=min(EVALUATION_HEARTBEAT_SECONDS, remaining),
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                task = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = _failed_worker_result(task, error)
                results.append(result)
                _print_worker_result(result, detailed=show_worker_details)

            if show_worker_details:
                elapsed = time.monotonic() - started_at
                completed = completed_before_wave + sum(
                    result.outcome != "EVALUATION_TIMEOUT" for result in results
                )
                print(
                    f"Frozen evaluation | V{variant} | {completed}/{total_trials} "
                    f"complete | active={len(pending)} | elapsed={elapsed:.1f}s "
                    f"| deadline={timeout_seconds:g}s",
                    flush=True,
                )

        if pending:
            timed_out = True
            for future in pending:
                future.cancel()
                results.append(_timed_out_evaluation_result(futures[future]))

        timed_out_count = sum(
            result.outcome == "EVALUATION_TIMEOUT" for result in results
        )
        if timed_out:
            _terminate_executor_processes(executor)
        return results, timed_out_count
    except BaseException:
        timed_out = True
        _terminate_executor_processes(executor)
        raise
    finally:
        if not timed_out:
            executor.shutdown(wait=True, cancel_futures=True)


def _evaluate_frozen_policy(
    unlocked_challenges: tuple[Challenge, ...],
    weights: dict[str, float],
    eval_trials: int,
    seed: int,
    workers: int,
    batch_number: int,
    learning_parameters: tuple[float, float, float],
    eval_timeout_seconds: float,
    worker_details: bool = False,
) -> tuple[dict[int, float], dict[int, int], dict]:
    from team01.agent.evaluation import Q_FEATURE_VERSION, q_feature_names

    feature_names = tuple(q_feature_names())
    evaluation_win_rates: dict[int, float] = {}
    valid_counts: dict[int, int] = {}
    timeout_counts: dict[int, int] = {}
    skipped_variants = []
    for challenge_index, challenge in enumerate(unlocked_challenges):
        variant_seed = seed + (1 << 32) + (challenge.number - 1) * eval_trials
        variant_tasks = _make_round_tasks(
            challenge=challenge,
            trial_numbers=tuple(range(1, eval_trials + 1)),
            batch_number=batch_number,
            first_trial_index=1,
            trial_count=eval_trials,
            round_number=0,
            base_seed=variant_seed,
            starting_weights=dict(weights),
            guis=0,
            q_contributions_diagnostic=False,
            learning_parameters=learning_parameters,
            no_training=True,
            evaluation_timeout_seconds=eval_timeout_seconds,
            worker_slots=workers,
            worker_details=worker_details,
        )
        results = []
        variant_timeouts = 0
        for wave_start in range(0, eval_trials, workers):
            wave_tasks = tuple(variant_tasks[wave_start : wave_start + workers])
            wave_results, wave_timeouts = _run_parallel_evaluation_wave(
                wave_tasks,
                eval_timeout_seconds,
                challenge.number,
                wave_start,
                eval_trials,
            )
            results.extend(wave_results)
            variant_timeouts += wave_timeouts
            if wave_timeouts:
                break

        valid_results = _validate_worker_results(
            weights,
            results,
            feature_names,
            Q_FEATURE_VERSION,
            require_unchanged_weights=True,
        )
        valid_ids = {id(result) for result in valid_results}
        for result in results:
            if result.error:
                print(
                    f"[Evaluation V{result.stage_number} | Trial {result.trial_number}] "
                    f"Not counted: {result.error}",
                    flush=True,
                )

        variant_results = [
            result for result in results if id(result) in valid_ids
        ]
        wins = sum(result.outcome == "WON" for result in variant_results)
        valid_counts[challenge.number] = len(variant_results)
        evaluation_win_rates[challenge.number] = wins / eval_trials
        timeout_counts[challenge.number] = variant_timeouts
        if variant_timeouts:
            print(
                f"WARNING: Frozen evaluation timed out on Variant {challenge.number}. "
                f"Completed: {len(variant_results)}/{eval_trials}; "
                f"Timed out: {variant_timeouts}/{eval_trials}. "
                "Evaluation cannot satisfy final completion; returning to training.",
                flush=True,
            )
            skipped_variants = [
                remaining.number for remaining in unlocked_challenges[challenge_index + 1 :]
            ]
            for variant in skipped_variants:
                evaluation_win_rates[variant] = 0.0
                valid_counts[variant] = 0
                timeout_counts[variant] = 0
            break

    return evaluation_win_rates, valid_counts, {
        "timed_out_counts": timeout_counts,
        "skipped_variants": skipped_variants,
    }


def _print_frozen_evaluation(
    unlocked_variants: tuple[int, ...] | list[int],
    evaluation_win_rates: dict[int, float],
    valid_counts: dict[int, int],
    eval_trials: int,
) -> bool:
    print("\nFROZEN EVALUATION", flush=True)
    print("=" * 54, flush=True)
    for variant in sorted(unlocked_variants):
        win_rate = evaluation_win_rates.get(variant, 0.0)
        wins = round(win_rate * eval_trials)
        passed = (
            valid_counts.get(variant, 0) == eval_trials
            and win_rate >= FINAL_WIN_RATE_THRESHOLD
        )
        print(
            f"V{variant}: {wins}/{eval_trials} "
            f"{win_rate * 100:.1f}%   {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    print("-" * 54, flush=True)
    print(
        f"Required: >={FINAL_WIN_RATE_THRESHOLD * 100:.1f}% on EVERY variant",
        flush=True,
    )
    complete = _all_variants_pass(
        unlocked_variants,
        evaluation_win_rates,
        valid_counts,
        eval_trials,
    )
    overall = "COMPLETE" if complete else "NOT COMPLETE"
    if len(unlocked_variants) < len(progression()):
        overall += f" ({len(unlocked_variants)}/{len(progression())} variants unlocked)"
    print(f"Overall: {overall}", flush=True)
    if unlocked_variants:
        weakest = _select_weakest_variant(
            evaluation_win_rates, tuple(sorted(unlocked_variants))
        )
        print(f"Weakest variant: V{weakest}", flush=True)
        if not complete:
            print(f"Next focus: V{weakest}", flush=True)
    print("=" * 54, flush=True)
    return complete


def _evaluate_batch(
    unlocked_challenges: tuple[Challenge, ...],
    weights: Optional[dict[str, float]],
    eval_trials: int,
    seed: int,
    workers: int,
    batch_number: int,
    learning_parameters: tuple[float, float, float],
    evaluation_runner: Optional[Callable],
    history_path: Optional[Path],
    eval_timeout_seconds: float,
    worker_details: bool = False,
    frozen_evaluator: Optional[Callable] = None,
) -> tuple[dict[int, float], dict[int, int], bool, dict]:
    """Evaluate unlocked variants and return rates, counts, completion, and details."""
    if evaluation_runner is not None:
        rates, valid_counts = evaluation_runner(
            unlocked_challenges=unlocked_challenges,
            weights=dict(weights) if weights is not None else None,
            eval_trials=eval_trials,
            seed=seed,
            workers=workers,
            batch_number=batch_number,
            learning_parameters=learning_parameters,
        )
        evaluation_details = {"timed_out_counts": {}, "skipped_variants": []}
    elif frozen_evaluator is not None:
        rates, valid_counts, evaluation_details = frozen_evaluator(
            unlocked_challenges=unlocked_challenges,
            eval_trials=eval_trials,
            seed=seed,
            batch_number=batch_number,
        )
    else:
        rates, valid_counts, evaluation_details = _evaluate_frozen_policy(
            unlocked_challenges,
            weights,
            eval_trials,
            seed,
            workers,
            batch_number,
            learning_parameters,
            eval_timeout_seconds,
            worker_details,
        )

    completed = _print_frozen_evaluation(
        tuple(challenge.number for challenge in unlocked_challenges),
        rates,
        valid_counts,
        eval_trials,
    )
    _append_history_record(
        history_path,
        {
            "record_type": "evaluation",
            "phase": "variant",
            "batch": batch_number,
            "unlocked_variants": [challenge.number for challenge in unlocked_challenges],
            "eval_trials_per_variant": eval_trials,
            "eval_timeout_seconds": eval_timeout_seconds,
            "evaluation_win_rates": {
                str(variant): rate for variant, rate in sorted(rates.items())
            },
            "evaluation_valid_counts": {
                str(variant): count for variant, count in sorted(valid_counts.items())
            },
            "evaluation_timeout_counts": {
                str(variant): count
                for variant, count in sorted(
                    evaluation_details["timed_out_counts"].items()
                )
            },
            "evaluation_skipped_variants": evaluation_details["skipped_variants"],
            "complete": completed,
        },
    )
    return rates, valid_counts, completed, evaluation_details


def _evaluate_dqn_policy(
    agent,
    unlocked_challenges: tuple[Challenge, ...],
    eval_trials: int,
    seed: int,
    batch_number: int,
    game_factory: Callable,
    episode_runner: Callable,
    timeout_seconds: float,
) -> tuple[dict[int, float], dict[int, int], dict]:
    """Evaluate a canonical DQN frozen in place without changing learner state."""
    from team01.agent.deep_q_learning import checkpoint_snapshots_equal

    snapshot_before = agent.get_checkpoint_snapshot()
    replay_size_before = len(agent.replay_buffer)
    episodes_before = agent.episodes
    was_training = agent.training
    was_learning_enabled = agent.learning_enabled
    was_collecting = agent.collect_experience
    previous_epsilon = agent.epsilon
    rates: dict[int, float] = {}
    valid_counts: dict[int, int] = {}
    timeout_counts: dict[int, int] = {}
    skipped_variants = []
    timed_out = False

    try:
        agent.set_learning(no_training=True, epsilon=0.0)
        for challenge_index, challenge in enumerate(unlocked_challenges):
            wins = 0
            valid = 0
            timeouts = 0
            for trial_number in range(1, eval_trials + 1):
                episode_seed = (
                    seed
                    + (1 << 32)
                    + (challenge.number - 1) * eval_trials
                    + trial_number
                )
                game, evaluation_agent = game_factory(
                    challenge,
                    episode_seed,
                    None,
                    lambda: agent,
                )
                evaluation_agent.set_learning(no_training=True, epsilon=0.0)
                if episode_runner is _run_episode or getattr(
                    episode_runner, "func", None
                ) is _run_episode:
                    try:
                        result = _run_frozen_evaluation_episode(
                            game,
                            evaluation_agent,
                            trial_number,
                            timeout_seconds,
                        )
                    except FrozenEvaluationTimeout:
                        timeouts += 1
                        timed_out = True
                        break
                else:
                    result = episode_runner(
                        game,
                        evaluation_agent,
                        trial_number,
                        f"DQN frozen evaluation V{challenge.number} "
                        f"{trial_number}/{eval_trials}",
                        False,
                    )
                if _dqn_result_is_numerically_stable(result):
                    valid += 1
                    wins += result.outcome == "WON"
            valid_counts[challenge.number] = valid
            rates[challenge.number] = wins / eval_trials
            timeout_counts[challenge.number] = timeouts
            if timed_out:
                skipped_variants = [
                    item.number
                    for item in unlocked_challenges[challenge_index + 1 :]
                ]
                for item_number in skipped_variants:
                    rates[item_number] = 0.0
                    valid_counts[item_number] = 0
                    timeout_counts[item_number] = 0
                break
    finally:
        agent.episodes = episodes_before
        unchanged = (
            checkpoint_snapshots_equal(
                snapshot_before,
                agent.get_checkpoint_snapshot(),
            )
            and len(agent.replay_buffer) == replay_size_before
        )
        if was_training and was_learning_enabled and was_collecting:
            agent.set_learning(no_training=False, epsilon=previous_epsilon)
        elif was_collecting:
            agent.set_rollout(
                epsilon=previous_epsilon,
                collect_experience=True,
            )
        else:
            agent.set_learning(no_training=True, epsilon=0.0)
        if not unchanged:
            raise RuntimeError("Frozen DQN evaluation mutated the training learner")

    return rates, valid_counts, {
        "timed_out_counts": timeout_counts,
        "skipped_variants": skipped_variants,
    }


def _maybe_warn_stale_policy(
    batch_metrics: dict,
    recent_batches,
    all_variants_unlocked: bool,
    final_complete: bool,
) -> None:
    if not all_variants_unlocked:
        recent_batches.clear()
        return
    if (
        batch_metrics["relative_weight_delta_l2"] is None
        or batch_metrics["evaluation_score"] is None
    ):
        recent_batches.clear()
        return
    recent_batches.append(batch_metrics)
    if final_complete or len(recent_batches) < STALE_WINDOW:
        return

    window = list(recent_batches)
    evaluation_scores = [item["evaluation_score"] for item in window]
    if (
        max(item["relative_weight_delta_l2"] for item in window)
        <= STALE_RELATIVE_DELTA_THRESHOLD
        and max(item["mean_worker_delta_l2"] for item in window)
        <= STALE_RELATIVE_DELTA_THRESHOLD
        and max(item["mean_abs_td_error"] for item in window)
        <= STALE_MEAN_TD_THRESHOLD
        and max(evaluation_scores) - min(evaluation_scores)
        <= STALE_EVAL_IMPROVEMENT_THRESHOLD
    ):
        print(
            "WARNING: Policy may be stale. Relative weight updates and TD errors "
            f"have remained small for {STALE_WINDOW} batches, frozen evaluation "
            "has not meaningfully improved, and at least one variant remains below threshold.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Curriculum coordinators

def _run_parallel_curriculum(
    trials: int,
    survive: int,
    base_seed: int,
    weights_path: Path,
    max_batches: Optional[int],
    workers: int,
    guis: int,
    q_contributions_diagnostic: bool,
    weights_state: list[dict[str, float]],
    save_agent,
    checkpoint_state: list[bool],
    summary: TrainingSummary,
    no_training: bool,
    eval_trials: int,
    history_path: Optional[Path],
    evaluation_runner: Optional[Callable],
    merge_strategy: str,
    eval_timeout_seconds: float,
    curriculum: str,
    stop_after_drills: bool,
    focus_batches: int,
    drill_refresh_after: int,
    drill_refresh_trials: int,
    worker_details: bool = False,
) -> None:
    """Coordinate drill/variant batches, unlocks, evaluation, focus, and refresh."""
    from team01.agent.evaluation import Q_FEATURE_VERSION, q_feature_names

    feature_names = tuple(q_feature_names())
    learning_parameters = _default_learning_parameters()
    round_counter = 0
    variant_items = progression()
    drill_items = drill_progression()
    engine = CurriculumEngine(
        trials=trials,
        survive=survive,
        seed=base_seed,
        eval_trials=eval_trials,
        curriculum=curriculum,
        focus_batches=focus_batches,
        drill_refresh_after=drill_refresh_after,
        no_training=no_training,
    )
    summary.decision_trace = engine.trace
    phase = engine.phase
    active_items, item_by_number = _curriculum_items(phase)
    unlocked_items = engine.unlocked_items
    phase_batch = engine.phase_batch
    phase_seed = engine.phase_seed
    recent_batches = deque(maxlen=STALE_WINDOW)
    focus_variant: Optional[int] = None
    batches_since_evaluation = 0
    last_evaluation_win_rates: dict[int, float] = {}
    variant_stagnation_trials = 0
    drill_refresh_count = 0

    # Keep the drill and variant phases on one loop so they share task
    # execution and weight synchronization, while retaining separate unlock rules.
    while not summary.completed:
        if max_batches is not None and summary.completed_batches >= max_batches:
            summary.stopped_reason = "configured batch limit reached"
            break

        batch_number = summary.completed_batches + 1
        summary.current_batch = batch_number
        batch_plan = engine.plan_batch()
        phase_batch = engine.phase_batch
        batch_phase = engine.batch_phase
        batch_unlocked_items = tuple(unlocked_items)
        newest_item = unlocked_items[-1]
        summary.current_stage = item_by_number[newest_item]
        batch_last_evaluation_win_rates = engine.batch_last_evaluation_win_rates
        batch_focus_variant = engine.batch_focus_variant
        focus_batch_index = (
            batches_since_evaluation + 1 if batch_focus_variant is not None else None
        )
        training_mode = (
            "weakest_variant_focus"
            if batch_focus_variant is not None
            else ("read_only" if no_training else f"progressive_{phase}")
        )
        episode_items = batch_plan.episode_items
        distribution = batch_plan.distribution
        item_prefix = "Drill" if phase == "drill" else "Variant"
        item_names = {
            number: item.name.removeprefix(f"{item_prefix} {number}: ")
            for number, item in item_by_number.items()
        }
        unlocked_label = ", ".join(
            f"{item_prefix[0]}{number}" if phase == "drill" else str(number)
            for number in unlocked_items
        )
        newest_label = (
            f"{item_prefix[0]}{newest_item}"
            if phase == "drill"
            else str(newest_item)
        )
        batch_heading = (
            "FOCUSED VARIANT TRAINING"
            if batch_focus_variant is not None
            else f"{phase.upper()} TRAINING"
        )
        item_heading = (
            f"Focus variant: V{batch_focus_variant} "
            f"{item_names[batch_focus_variant]}"
            if batch_focus_variant is not None
            else f"Newest {item_prefix.lower()}: {newest_label} "
            f"{item_names[newest_item]}"
        )
        print(
            f"\n{'=' * 54}\nBatch {batch_number} | "
            f"{batch_heading} | {trials} total episodes\n"
            f"Unlocked {item_prefix.lower()}s: {unlocked_label}\n"
            f"{item_heading}\nTraining distribution:",
            flush=True,
        )
        if batch_focus_variant is not None:
            print(
                f"Frozen evaluation focus: V{batch_focus_variant} "
                f"{item_by_number[batch_focus_variant].name}\n"
                "Last frozen win rates:",
                flush=True,
            )
            for variant in range(1, len(variant_items) + 1):
                print(
                    f"V{variant}: "
                    f"{last_evaluation_win_rates.get(variant, 0.0) * 100:.1f}%",
                    flush=True,
                )
            print(
                f"Focus batch {focus_batch_index}/{focus_batches}", flush=True
            )
        for item_number, count in distribution.items():
            print(
                f"{item_prefix[0]}{item_number} {item_names[item_number]}: {count}",
                flush=True,
            )

        batch_results: list[tuple[int, ParallelTrainingResult]] = []
        batch_round_metrics = []
        batch_valid_workers = 0
        batch_worker_count = 0
        batch_stopped = False

        # Every worker in a round starts from the same synchronized weights.
        for trial_numbers in _trial_rounds(trials, workers):
            round_counter += 1
            round_start_weights = {
                name: float(weights_state[0].get(name, 0.0))
                for name in feature_names
            }
            round_items = tuple(
                episode_items[trial_number - 1]
                for trial_number in trial_numbers
            )
            tasks = _make_round_tasks(
                challenge=item_by_number[newest_item],
                trial_numbers=trial_numbers,
                batch_number=batch_number,
                first_trial_index=summary.completed_trials + 1,
                trial_count=trials,
                round_number=round_counter,
                base_seed=phase_seed,
                starting_weights=round_start_weights,
                guis=guis,
                q_contributions_diagnostic=q_contributions_diagnostic,
                learning_parameters=learning_parameters,
                no_training=no_training,
                phase=phase,
                trial_challenges=tuple(
                    item_by_number[item_number] for item_number in round_items
                ),
                worker_slots=workers,
                worker_details=worker_details,
            )
            summary.current_round = round_counter
            results = _run_parallel_round(tasks)
            summary.current_round = 0
            summary.completed_trials += len(tasks)
            batch_worker_count += len(tasks)

            if no_training:
                valid_results = _validate_worker_results(
                    round_start_weights,
                    results,
                    feature_names,
                    Q_FEATURE_VERSION,
                    require_unchanged_weights=True,
                )
                merged_weights = dict(round_start_weights)
                candidate_metadata = []
                variant_shares = {}
            elif merge_strategy == "mean":
                merged_weights, valid_results = _average_worker_deltas(
                    round_start_weights,
                    results,
                    feature_names,
                    Q_FEATURE_VERSION,
                )
                candidate_metadata = []
                variant_shares = {}
            else:
                (
                    merged_weights,
                    valid_results,
                    candidate_metadata,
                    variant_shares,
                ) = _genetic_worker_crossover(
                    round_start_weights,
                    results,
                    feature_names,
                    Q_FEATURE_VERSION,
                    round_items,
                    phase_seed,
                    round_counter,
                    trial_numbers=trial_numbers,
                )
                for candidate in candidate_metadata:
                    candidate["phase"] = phase
                    candidate["curriculum_item"] = candidate["variant"]
            batch_results.extend((result.stage_number, result) for result in results)
            batch_valid_workers += len(valid_results)
            for result in results:
                if result.error:
                    print(
                        f"[W{result.worker_id} | Trial {result.trial_index}] "
                        f"Not merged: {result.error}",
                        flush=True,
                    )
            if not valid_results:
                summary.stopped_reason = (
                    f"no valid worker results in parallel round {round_counter}"
                )
                if not no_training and not checkpoint_state[0] and not weights_path.exists():
                    checkpoint_state[0] = True
                print(
                    f"STOP: {summary.stopped_reason}; retaining the last synchronized weights.",
                    flush=True,
                )
                batch_stopped = True
                break

            if no_training:
                if worker_details:
                    print(
                        "Evaluation-mode training round; canonical weights unchanged.",
                        flush=True,
                    )
            else:
                if merge_strategy == "genetic":
                    _print_genetic_selection(
                        candidate_metadata,
                        phase,
                        detailed=worker_details,
                    )
                weights_state[0] = merged_weights
                _save_weights(save_agent, weights_state[0], weights_path)
                checkpoint_state[0] = True
                summary.weights_saved = True
                if worker_details:
                    print("Weights synchronized.", flush=True)
                    print(
                        f"Max |canonical weight|: "
                        f"{max((abs(value) for value in weights_state[0].values()), default=0.0):.3f}",
                        flush=True,
                    )
                    print(f"Weights saved: {weights_path}", flush=True)

            diagnostics = _weight_update_diagnostics(
                round_start_weights,
                [result.weights for result in valid_results],
                merged_weights,
                [(result.mean_abs_td_error, result.update_count) for result in valid_results],
                max((result.max_abs_q for result in valid_results), default=0.0),
            )
            diagnostics["merge_strategy"] = (
                "none" if no_training else merge_strategy
            )
            if no_training or merge_strategy != "genetic":
                diagnostics["genetic_child_delta_l2"] = None
            if worker_details:
                _print_learning_progress(diagnostics)
            if (
                diagnostics["mean_worker_delta_l2"] > 1e-9
                and diagnostics["update_agreement_ratio"] < UPDATE_CANCELLATION_RATIO
            ):
                print("Worker updates are active but strongly conflicting/cancelling.", flush=True)
            batch_round_metrics.append(diagnostics)
            _append_history_record(
                history_path,
                {
                    "record_type": "training_round",
                    "batch": batch_number,
                    "round": round_counter,
                    **_phase_history_fields(
                        batch_phase, batch_unlocked_items, round_items
                    ),
                    "training_enabled": not no_training,
                    "training_mode": training_mode,
                    "focus_variant": batch_focus_variant,
                    **diagnostics,
                    "weights": dict(merged_weights),
                    "merge_strategy": "none" if no_training else merge_strategy,
                    "genetic_candidates": candidate_metadata,
                    "variant_contribution_shares": {
                        str(variant): share
                        for variant, share in sorted(variant_shares.items())
                    },
                    "item_contribution_shares": {
                        str(variant): share
                        for variant, share in sorted(variant_shares.items())
                    },
                },
            )
            print(
                f"[Round {round_counter}] complete | "
                f"Valid workers: {len(valid_results)}/{len(tasks)} | "
                f"Wins: {sum(result.outcome == 'WON' for result in valid_results)}",
                flush=True,
            )

        if batch_stopped:
            break

        # Unlock progress depends on wins for the newest drill or variant.
        newest_count = distribution.get(newest_item, 0)
        required_wins = batch_plan.required_newest_wins
        _print_mixed_batch_summary(
            episode_items,
            batch_results,
            newest_item,
            required_wins,
            phase=batch_phase,
            item_names=item_names,
            show_newest=batch_focus_variant is None,
        )
        if batch_focus_variant is not None:
            _print_focus_variant_results(batch_focus_variant, batch_results)
        summary.completed_batches += 1
        decision = engine.record_batch(batch_results, require_error_free=True)
        phase = engine.phase
        active_items, item_by_number = _curriculum_items(phase)
        unlocked_items = engine.unlocked_items
        phase_batch = engine.phase_batch
        phase_seed = engine.phase_seed
        batches_since_evaluation = engine.batches_since_evaluation
        variant_stagnation_trials = engine.stagnation_trials
        newest_wins = decision.newest_wins
        phase_transitioned = decision.phase_transitioned
        stage_unlocked = decision.stage_unlocked
        frozen_progress_detected = decision.progress_detected

        if batch_focus_variant is not None:
            pass
        elif stage_unlocked:
            summary.stages_passed += 1
            summary.current_stage = item_by_number[newest_item + 1]
            if frozen_progress_detected:
                print(
                    f"Variant progress detected: V{newest_item + 1} unlocked. "
                    "Stagnation counter reset.",
                    flush=True,
                )
            print(
                f"{item_prefix[0]}{newest_item} PASSED "
                f"({newest_wins}/{newest_count}) -> "
                f"{item_prefix[0]}{newest_item + 1} unlocked",
                flush=True,
            )
        elif phase_transitioned:
            print(
                f"\n{'=' * 54}\nDRILL CURRICULUM COMPLETE\n"
                f"{'=' * 54}\nAll skill drills passed.\n\n"
                "Carrying learned weights into the Project 2 variant curriculum.\n"
                "Starting Variant 1.\n"
                f"{'=' * 54}",
                flush=True,
            )
            recent_batches.clear()
            summary.current_stage = item_by_number[1]
        elif (
            batch_phase == "variant"
            and newest_item == len(variant_items)
            and newest_wins >= required_wins
        ):
            print(
                f"Newest variant V{newest_item} met its training ratio; "
                "final frozen evaluation still controls completion.",
                flush=True,
            )
        else:
            print(
                f"Newest {item_prefix.lower()} {newest_item} did not unlock "
                "another stage "
                f"({newest_wins}/{newest_count}, need {required_wins}).",
                flush=True,
            )

        if (
            batch_phase == "variant"
            and not no_training
            and not stage_unlocked
        ):
            print(
                f"Variant progress: trials since last progress: "
                f"{variant_stagnation_trials}/"
                f"{drill_refresh_after if drill_refresh_after else 'disabled'}",
                flush=True,
            )

        evaluation_performed = False
        # Evaluation is read-only and begins only after all variants unlock.
        if batch_phase == "drill":
            print(
                "Frozen evaluation skipped: drill training does not use frozen evaluation.",
                flush=True,
            )
            evaluation_win_rates = {}
            final_complete = False
            evaluation_details = {
                "timed_out_counts": {},
                "skipped_variants": [],
            }
        elif len(unlocked_items) < len(variant_items):
            print(
                "Frozen evaluation skipped: all five variants are not yet unlocked.",
                flush=True,
            )
            evaluation_win_rates = {}
            final_complete = False
            evaluation_details = {
                "timed_out_counts": {},
                "skipped_variants": [],
            }
        elif decision.evaluation_due:
            print(
                "All variants unlocked. Beginning frozen evaluation.",
                flush=True,
            )
            unlocked_variant_challenges = tuple(
                item_by_number[variant] for variant in unlocked_items
            )
            (
                evaluation_win_rates,
                _,
                final_complete,
                evaluation_details,
            ) = _evaluate_batch(
                unlocked_variant_challenges,
                weights_state[0],
                eval_trials,
                base_seed,
                workers,
                batch_number,
                learning_parameters,
                evaluation_runner,
                history_path,
                eval_timeout_seconds,
            )
            evaluation_performed = True
            eval_decision = engine.record_evaluation(
                evaluation_win_rates, complete=final_complete
            )
            last_evaluation_win_rates = dict(engine.last_evaluation_win_rates)
            variant_stagnation_trials = engine.stagnation_trials
            frozen_progress_detected = (
                frozen_progress_detected or eval_decision.progress_detected
            )
            if final_complete:
                print(
                    f"All variants meet >="
                    f"{FINAL_WIN_RATE_THRESHOLD * 100:.0f}%. Training complete.",
                    flush=True,
                )
            else:
                if not no_training:
                    if not batch_last_evaluation_win_rates:
                        print(
                            "Initial all-variant frozen baseline established; "
                            "stagnation counter reset.",
                            flush=True,
                        )
                    elif eval_decision.improved:
                        print(
                            "Frozen progress detected in passing count, "
                            "bottleneck, or focused variant; stagnation counter reset.",
                            flush=True,
                        )
                    else:
                        print(
                            f"Frozen progress not detected; variant stagnation "
                            f"remains {variant_stagnation_trials} trials.",
                            flush=True,
                        )
                previous_focus = eval_decision.previous_focus
                focus_variant = eval_decision.focus_variant
                if previous_focus != focus_variant:
                    _append_history_record(
                        history_path,
                        {
                            "record_type": "focus_change",
                            "previous_focus": previous_focus,
                            "new_focus": focus_variant,
                            "evaluation_win_rates": {
                                str(variant): rate
                                for variant, rate in sorted(
                                    evaluation_win_rates.items()
                                )
                            },
                        },
                    )
                _print_focus_selection(
                    evaluation_win_rates, focus_variant, focus_batches
                )
                batches_since_evaluation = engine.batches_since_evaluation
        else:
            evaluation_win_rates = dict(last_evaluation_win_rates)
            final_complete = False
            evaluation_details = {
                "timed_out_counts": {},
                "skipped_variants": [],
            }
            print(
                f"Frozen reevaluation deferred: focused batch "
                f"{batches_since_evaluation}/{focus_batches} complete.",
                flush=True,
            )
        relative_deltas = [
            item["relative_weight_delta_l2"]
            for item in batch_round_metrics
            if item["relative_weight_delta_l2"] is not None
        ]
        batch_metrics = {
            "relative_weight_delta_l2": (
                math.fsum(relative_deltas) / len(relative_deltas)
                if relative_deltas
                else None
            ),
            "mean_worker_delta_l2": math.fsum(
                item["mean_worker_delta_l2"] for item in batch_round_metrics
            ) / max(1, len(batch_round_metrics)),
            "mean_abs_td_error": math.fsum(
                item["mean_abs_td_error"] for item in batch_round_metrics
            ) / max(1, len(batch_round_metrics)),
            "canonical_delta_l2": math.fsum(
                item["canonical_delta_l2"] for item in batch_round_metrics
            ) / max(1, len(batch_round_metrics)),
            "update_agreement_ratio": math.fsum(
                item["update_agreement_ratio"] for item in batch_round_metrics
            ) / max(1, len(batch_round_metrics)),
            "evaluation_score": (
                math.fsum(evaluation_win_rates.values())
                / len(evaluation_win_rates)
                if evaluation_win_rates
                else None
            ),
        }
        _print_batch_learning_summary(
            batch_metrics,
            batch_valid_workers,
            batch_worker_count,
            "none" if no_training else merge_strategy,
        )
        _append_history_record(
            history_path,
            {
                "record_type": "training_batch",
                "batch": batch_number,
                **_phase_history_fields(
                    batch_phase, batch_unlocked_items, episode_items
                ),
                "merge_strategy": "none" if no_training else merge_strategy,
                "training_mode": training_mode,
                "focus_variant": batch_focus_variant,
                "last_evaluation_win_rates": {
                    str(variant): rate
                    for variant, rate in sorted(
                        batch_last_evaluation_win_rates.items()
                    )
                },
                "focus_batch_index": focus_batch_index,
                "focus_batches_per_cycle": focus_batches,
                "evaluation_performed": evaluation_performed,
                "variant_stagnation_trials": variant_stagnation_trials,
                "drill_refresh_count": drill_refresh_count,
                "variant_progress_detected": frozen_progress_detected,
                "training_wins_by_item": {
                    str(item_number): sum(
                        result_item == item_number
                        and result.outcome == "WON"
                        and result.error is None
                        for result_item, result in batch_results
                    )
                    for item_number in distribution
                },
                "evaluation_win_rates": {
                    str(variant): rate
                    for variant, rate in sorted(evaluation_win_rates.items())
                },
                "evaluation_timeout_counts": {
                    str(variant): count
                    for variant, count in sorted(
                        evaluation_details["timed_out_counts"].items()
                    )
                },
                "evaluation_skipped_variants": evaluation_details[
                    "skipped_variants"
                ],
                **batch_metrics,
            },
        )
        # Refresh only after variant stagnation; drills do not advance the
        # variant batch counter or alter the saved focus/evaluation state.
        refresh_due = engine.refresh_due(evaluation_performed=evaluation_performed)
        if refresh_due:
            refresh_number = engine.drill_refresh_count + 1
            refresh_reason = (
                "stage progression did not advance"
                if len(unlocked_items) < len(variant_items)
                else "frozen evaluation showed no meaningful variant progress"
            )
            # Drill refresh changes weights, but must not reset variant progress.
            saved_variant_state = VariantResumeState(
                unlocked_variants=tuple(unlocked_items),
                variant_batch=phase_batch,
                variant_seed=phase_seed,
                focus_variant=focus_variant,
                batches_since_evaluation=batches_since_evaluation,
                evaluation_win_rates=dict(last_evaluation_win_rates),
                current_stage=summary.current_stage,
                completed_batches=summary.completed_batches,
            )
            print(
                f"\n{'=' * 54}\nDRILL REFRESH TRIGGERED\n"
                f"{'=' * 54}\n"
                f"Reason: {variant_stagnation_trials} variant training episodes "
                f"without meaningful progress; {refresh_reason}.\n"
                f"Current variant state:\nUnlocked: "
                f"{', '.join(f'V{item}' for item in unlocked_items)}\n"
                f"Newest: V{newest_item} - {item_by_number[newest_item].name}\n"
                f"Current focus: "
                f"{f'V{focus_variant}' if focus_variant is not None else 'none'}\n"
                f"Refresh schedule:",
                flush=True,
            )
            for drill in drill_items:
                print(
                    f"D{drill.number} - {drill.name}: "
                    f"{drill_refresh_trials} episodes",
                    flush=True,
                )
            print(
                "Canonical weights continue learning during the refresh.",
                flush=True,
            )
            _append_history_record(
                history_path,
                {
                    "record_type": "drill_refresh_start",
                    "refresh_number": refresh_number,
                    "trigger_variant_trials": drill_refresh_after,
                    "variant_stagnation_trials": variant_stagnation_trials,
                    "phase_before_refresh": "variant",
                    "unlocked_variants": list(unlocked_items),
                    "newest_variant": newest_item,
                    "focus_variant": focus_variant,
                    "last_evaluation_win_rates": {
                        str(variant): rate
                        for variant, rate in sorted(
                            last_evaluation_win_rates.items()
                        )
                    },
                },
            )
            refresh_episode_start = summary.completed_trials
            round_counter, drill_results = _run_parallel_drill_refresh(
                drills=drill_items,
                refresh_trials=drill_refresh_trials,
                workers=workers,
                seed=phase_seed,
                refresh_number=refresh_number,
                batch_number=batch_number,
                round_counter=round_counter,
                guis=guis,
                q_contributions_diagnostic=q_contributions_diagnostic,
                learning_parameters=learning_parameters,
                weights_state=weights_state,
                weights_path=weights_path,
                save_agent=save_agent,
                checkpoint_state=checkpoint_state,
                summary=summary,
                merge_strategy=merge_strategy,
                history_path=history_path,
                worker_details=worker_details,
            )
            engine.refresh_completed()
            drill_refresh_count = engine.drill_refresh_count
            _append_history_record(
                history_path,
                {
                    "record_type": "drill_refresh_complete",
                    "refresh_number": refresh_number,
                    "drill_results": {
                        str(drill): result
                        for drill, result in sorted(drill_results.items())
                    },
                    "episodes_completed": summary.completed_trials
                    - refresh_episode_start,
                    "weights_saved": summary.weights_saved,
                    "unlocked_variants": list(saved_variant_state.unlocked_variants),
                    "newest_variant": saved_variant_state.unlocked_variants[-1],
                    "focus_variant": saved_variant_state.focus_variant,
                    "variant_stagnation_trials": 0,
                },
            )
            phase = "variant"
            active_items = variant_items
            item_by_number = {item.number: item for item in variant_items}
            unlocked_items = list(saved_variant_state.unlocked_variants)
            phase_batch = saved_variant_state.variant_batch
            phase_seed = saved_variant_state.variant_seed
            focus_variant = saved_variant_state.focus_variant
            batches_since_evaluation = saved_variant_state.batches_since_evaluation
            last_evaluation_win_rates = dict(
                saved_variant_state.evaluation_win_rates
            )
            summary.current_stage = saved_variant_state.current_stage
            if summary.completed_batches != saved_variant_state.completed_batches:
                raise RuntimeError("drill refresh changed the variant batch counter")
            variant_stagnation_trials = 0
            newest_name = item_by_number[unlocked_items[-1]].name.removeprefix(
                f"Variant {unlocked_items[-1]}: "
            )
            print(
                f"\n{'=' * 54}\nDRILL REFRESH COMPLETE\n"
                f"{'=' * 54}\nReturning to variant training.\n"
                f"Unlocked variants: {', '.join(f'V{item}' for item in unlocked_items)}\n"
                f"Newest variant: V{unlocked_items[-1]} "
                f"{newest_name}\n"
                f"Current focus: "
                f"{f'V{focus_variant}' if focus_variant is not None else 'none'}\n"
                "Variant stagnation counter reset to 0.\n"
                f"{'=' * 54}",
                flush=True,
            )
        if not no_training and evaluation_performed:
            _maybe_warn_stale_policy(
                batch_metrics,
                recent_batches,
                batch_phase == "variant"
                and len(unlocked_items) == len(variant_items),
                final_complete,
            )
        elif no_training:
            recent_batches.clear()
        if phase_transitioned:
            _append_history_record(
                history_path,
                {
                    "record_type": "phase_transition",
                    "from": "drills",
                    "to": "variants",
                    "completed_drills": [item.number for item in drill_items],
                    "completed_batches": summary.completed_batches,
                    "completed_trials": summary.completed_trials,
                },
            )
            if stop_after_drills:
                summary.stopped_reason = (
                    "drill pretraining complete; stopped by request"
                )
                print(
                    "Drill pretraining complete; stopping by request.\n"
                    "Project 2 variant completion has NOT been evaluated.",
                    flush=True,
                )
                break
        if (
            no_training
            and evaluation_performed
            and batch_phase == "variant"
            and len(unlocked_items) == len(variant_items)
        ):
            if not final_complete:
                summary.stopped_reason = (
                    "no-training evaluation complete; focused training is disabled"
                )
                print(
                    "No-training mode: frozen evaluation complete; "
                    "focused training is disabled.",
                    flush=True,
                )
                break
        if final_complete:
            summary.completed = True



def _run_parallel_progressive_training(
    options: TrainingOptions,
) -> TrainingSummary:
    """Load weights, run the parallel curriculum, then save its canonical state."""
    trials = options.trials
    survive = options.survive
    seed = options.seed
    weights_path = options.weights_path
    guis = options.guis
    assert guis is not None
    fresh = options.fresh
    max_batches = options.max_batches
    q_contributions_diagnostic = options.q_contributions_diagnostic
    workers = options.workers
    no_training = options.no_training
    eval_trials = options.eval_trials
    history_path = options.history_path
    evaluation_runner = options.evaluation_runner
    merge_strategy = options.merge_strategy
    eval_timeout_seconds = options.eval_timeout_seconds
    curriculum = options.curriculum
    stop_after_drills = options.stop_after_drills
    focus_batches = options.focus_batches
    drill_refresh_after = options.drill_refresh_after
    drill_refresh_trials = options.drill_refresh_trials
    worker_details = options.worker_details

    _configure_runtime(guis > 0)
    if curriculum == "drills":
        validate_drill_maps()
    weights_path = Path(weights_path)
    weights, save_agent, can_save = _load_initial_weights(
        weights_path,
        fresh,
        _create_qagent,
    )
    checkpoint_state = [can_save]
    weights_state = [weights]
    summary = TrainingSummary()
    _append_history_record(
        history_path,
        {
            "record_type": "run_start",
            "seed": seed,
            "workers": workers,
            "trials_per_batch": trials,
            "eval_trials_per_variant": eval_trials,
            "eval_timeout_seconds": eval_timeout_seconds,
            "focus_batches": focus_batches,
            "drill_refresh_after": drill_refresh_after,
            "drill_refresh_trials": drill_refresh_trials,
            "merge_strategy": merge_strategy,
            "curriculum_mode": curriculum,
            "stop_after_drills": stop_after_drills,
            **_phase_history_fields(
                "drill" if curriculum == "drills" else "variant",
                [1],
                (),
            ),
            "fresh": fresh,
            "no_training": no_training,
        },
    )

    try:
        print("Project 2 Progressive Q-Learning Training", flush=True)
        print(f"Map: {DEFAULT_MAP_PATH}", flush=True)
        print(f"Trials per batch: {trials}", flush=True)
        print(f"Successes required: {survive}", flush=True)
        print(f"Base seed: {seed}", flush=True)
        print(f"Workers: {workers}", flush=True)
        print(f"Merge strategy: {merge_strategy}", flush=True)
        print(f"Curriculum: {curriculum}", flush=True)
        print(f"Focus batches: {focus_batches}", flush=True)
        print(
            f"Drill refresh: every {drill_refresh_after} stalled variant trials "
            f"({drill_refresh_trials} episodes per drill)"
            if curriculum == "drills" and drill_refresh_after > 0 and not no_training
            else "Drill refresh: disabled",
            flush=True,
        )
        print(f"GUI workers: {guis}", flush=True)
        print(
            "Learning: Disabled (greedy evaluation)"
            if no_training
            else "Learning: Enabled",
            flush=True,
        )
        print(
            f"Q contribution diagnostic: "
            f"{'Enabled' if q_contributions_diagnostic else 'Disabled'}",
            flush=True,
        )
        if q_contributions_diagnostic and guis:
            print(
                f"Detailed Q diagnostics are enabled for {guis} GUI workers. "
                "Terminal output may be very verbose.",
                flush=True,
            )

        _run_parallel_curriculum(
            trials=trials,
            survive=survive,
            base_seed=seed,
            weights_path=weights_path,
            max_batches=max_batches,
            workers=workers,
            guis=guis,
            q_contributions_diagnostic=q_contributions_diagnostic,
            weights_state=weights_state,
            save_agent=save_agent,
            checkpoint_state=checkpoint_state,
            summary=summary,
            no_training=no_training,
            eval_trials=eval_trials,
            history_path=history_path,
            evaluation_runner=evaluation_runner,
            merge_strategy=merge_strategy,
            eval_timeout_seconds=eval_timeout_seconds,
            curriculum=curriculum,
            stop_after_drills=stop_after_drills,
            focus_batches=focus_batches,
            drill_refresh_after=drill_refresh_after,
            drill_refresh_trials=drill_refresh_trials,
            worker_details=worker_details,
        )
    except KeyboardInterrupt:
        summary.interrupted = True
    except TrainingAborted as error:
        summary.stopped_reason = str(error)
        print(f"Training stopped: {error}", flush=True)
    finally:
        weights = weights_state[0]
        if no_training:
            print(
                "Evaluation mode: weights were not modified or saved. "
                f"Loaded checkpoint remains at: {weights_path}",
                flush=True,
            )
        elif checkpoint_state[0]:
            _save_weights(save_agent, weights, weights_path)
            summary.weights_saved = True
            print(f"Current learned weights saved to: {weights_path}", flush=True)
        else:
            print(
                "No trained checkpoint was produced; existing weights were left "
                f"unchanged at: {weights_path}",
                flush=True,
            )

    if summary.interrupted:
        print("\nTraining interrupted by user.", flush=True)
        if summary.current_round:
            print(
                f"Parallel round {summary.current_round} was incomplete; "
                "round changes discarded.",
                flush=True,
            )
            print(
                f"Completed trials before current round: {summary.completed_trials}",
                flush=True,
            )
        else:
            print(f"Completed trials: {summary.completed_trials}", flush=True)
        if no_training:
            print(
                f"Evaluation interrupted; loaded checkpoint remains unchanged at: "
                f"{weights_path}",
                flush=True,
            )
        else:
            print(
                f"Last synchronized weights saved to: "
                f"{weights_path if summary.weights_saved else 'not saved'}",
                flush=True,
            )
    elif summary.completed:
        print("\nPROGRESSIVE TRAINING COMPLETE", flush=True)
        print("All existing challenge levels passed.", flush=True)
        print(f"Final stage: Variant {progression()[-1].number}", flush=True)
        print(f"Total training episodes: {summary.completed_trials}", flush=True)
        if no_training:
            print(f"Evaluation complete; checkpoint unchanged at: {weights_path}", flush=True)
        else:
            print(f"Final weights saved to: {weights_path}", flush=True)
    elif summary.stopped_reason is not None:
        print(f"Training stopped: {summary.stopped_reason}", flush=True)
        print(f"Completed trials: {summary.completed_trials}", flush=True)
        print(f"Weights: {weights_path if summary.weights_saved else 'not saved'}", flush=True)

    return summary


def _validate_training_options(options: TrainingOptions) -> TrainingOptions:
    """Validate run settings and resolve the display/GUIs defaults."""
    if options.agent_type not in {"q", "dqn"}:
        raise ValueError("--agent must be q or dqn")
    if options.trials < 1:
        raise ValueError("trials must be at least 1")
    if not 1 <= options.survive <= options.trials:
        raise ValueError("survive must be between 1 and trials")
    if options.max_batches is not None and options.max_batches < 1:
        raise ValueError("max_batches must be at least 1")
    if options.eval_trials < 1:
        raise ValueError("--eval-trials must be at least 1")
    if options.merge_strategy not in {"genetic", "mean"}:
        raise ValueError("--merge-strategy must be genetic or mean")
    if options.curriculum not in {"variants", "drills"}:
        raise ValueError("--curriculum must be variants or drills")
    if options.stop_after_drills and options.curriculum != "drills":
        raise ValueError("--stop-after-drills requires --curriculum drills")
    if options.focus_batches < 1:
        raise ValueError("--focus-batches must be at least 1")
    if options.drill_refresh_after < 0:
        raise ValueError("--drill-refresh-after must be at least 0")
    if options.drill_refresh_trials < 1:
        raise ValueError("--drill-refresh-trials must be at least 1")
    if options.curriculum != "drills" and options.drill_refresh_after not in (
        0,
        DEFAULT_DRILL_REFRESH_AFTER,
    ):
        print(
            "Ignoring --drill-refresh-after: it only applies with --curriculum drills.",
            flush=True,
        )
    if options.eval_timeout_seconds <= 0 or not math.isfinite(
        options.eval_timeout_seconds
    ):
        raise ValueError("--eval-timeout-seconds must be positive and finite")
    if options.no_training and options.fresh:
        raise ValueError("--fresh cannot be used with --no-training")
    weights_path = Path(
        options.weights_path
        if options.weights_path is not None
        else (
            DEFAULT_WEIGHTS_PATH
            if options.agent_type == "q"
            else DEFAULT_DQN_CHECKPOINT_PATH
        )
    )
    if options.no_training and not weights_path.is_file():
        raise FileNotFoundError(
            f"Frozen evaluation requires an existing compatible weights file: {weights_path}"
        )
    if options.workers < 1:
        raise ValueError("--workers must be at least 1")
    if options.dqn_updates_per_batch < 0:
        raise ValueError("--dqn-updates-per-batch must be at least 0")
    if options.start_variant is not None:
        if options.agent_type != "dqn":
            raise ValueError("--start-variant is only supported with --agent dqn")
        if not 1 <= options.start_variant <= len(progression()):
            raise ValueError(
                f"--start-variant must be between 1 and {len(progression())}"
            )
    if options.agent_type != "dqn" and (
        options.dqn_learning_rate is not None or options.dqn_epsilon is not None
    ):
        raise ValueError("--dqn-learning-rate/--dqn-epsilon require --agent dqn")
    if options.dqn_learning_rate is not None and not (
        math.isfinite(options.dqn_learning_rate) and options.dqn_learning_rate > 0
    ):
        raise ValueError("--dqn-learning-rate must be positive and finite")
    if options.dqn_epsilon is not None and not (
        math.isfinite(options.dqn_epsilon) and 0.0 <= options.dqn_epsilon <= 1.0
    ):
        raise ValueError("--dqn-epsilon must be between 0 and 1")
    guis = options.guis
    if guis is None:
        guis = 1 if options.display else 0
    if guis < 0:
        raise ValueError("--guis must be at least 0")
    if guis > options.workers:
        raise ValueError("--guis cannot exceed --workers")

    display = guis > 0
    return replace(
        options,
        weights_path=weights_path,
        guis=guis,
        display=display,
    )


def run_progressive_training(
    trials: int = DEFAULT_TRIALS,
    survive: int = DEFAULT_SURVIVE,
    seed: int = DEFAULT_SEED,
    weights_path: Optional[Path] = None,
    display: bool = DEFAULT_DISPLAY,
    fresh: bool = False,
    max_batches: Optional[int] = None,
    q_contributions_diagnostic: bool = False,
    no_training: bool = False,
    *,
    workers: int = 1,
    guis: Optional[int] = None,
    eval_trials: int = DEFAULT_EVAL_TRIALS,
    history_path: Optional[Path] = None,
    merge_strategy: str = "genetic",
    eval_timeout_seconds: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
    curriculum: str = "variants",
    stop_after_drills: bool = False,
    focus_batches: int = DEFAULT_FOCUS_BATCHES,
    drill_refresh_after: int = DEFAULT_DRILL_REFRESH_AFTER,
    drill_refresh_trials: int = DEFAULT_DRILL_REFRESH_TRIALS,
    worker_details: bool = False,
    evaluation_runner: Optional[Callable] = None,
    agent_factory: Optional[Callable] = None,
    game_factory: Optional[Callable] = None,
    episode_runner: Optional[Callable] = None,
    agent_type: str = "q",
    dqn_updates_per_batch: int = DQN_UPDATES_PER_BATCH,
    start_variant: Optional[int] = None,
    dqn_learning_rate: Optional[float] = None,
    dqn_epsilon: Optional[float] = None,
) -> TrainingSummary:
    """Run the selected curriculum through parallel or sequential workers.

    Drill runs progress through D1-D4 before V1-V5. Once all variants unlock,
    frozen evaluation guides weakest-variant focus and any stalled drill refresh.
    """
    options = _validate_training_options(
        TrainingOptions(
            trials=trials,
            survive=survive,
            seed=seed,
            weights_path=weights_path,
            display=display,
            fresh=fresh,
            max_batches=max_batches,
            q_contributions_diagnostic=q_contributions_diagnostic,
            no_training=no_training,
            workers=workers,
            guis=guis,
            eval_trials=eval_trials,
            history_path=history_path,
            merge_strategy=merge_strategy,
            eval_timeout_seconds=eval_timeout_seconds,
            curriculum=curriculum,
            stop_after_drills=stop_after_drills,
            focus_batches=focus_batches,
            drill_refresh_after=drill_refresh_after,
            drill_refresh_trials=drill_refresh_trials,
            worker_details=worker_details,
            evaluation_runner=evaluation_runner,
            agent_factory=agent_factory,
            game_factory=game_factory,
            episode_runner=episode_runner,
            agent_type=agent_type,
            dqn_updates_per_batch=dqn_updates_per_batch,
            start_variant=start_variant,
            dqn_learning_rate=dqn_learning_rate,
            dqn_epsilon=dqn_epsilon,
        )
    )

    if options.agent_type == "dqn" and options.workers > 1:
        if any(
            factory is not None
            for factory in (
                options.agent_factory,
                options.game_factory,
                options.episode_runner,
            )
        ):
            raise ValueError(
                "Custom agent, game, and episode factories are only supported "
                "for DQN with --workers 1."
            )
    if options.agent_type == "dqn":
        return _run_sequential_progressive_training(options)

    if options.workers > 1:
        if any(
            factory is not None
            for factory in (
                options.agent_factory,
                options.game_factory,
                options.episode_runner,
            )
        ):
            raise ValueError(
                "Custom agent, game, and episode factories are only supported "
                "with --workers 1"
            )
        return _run_parallel_progressive_training(options)

    return _run_sequential_progressive_training(options)


def _run_sequential_progressive_training(
    options: TrainingOptions,
) -> TrainingSummary:
    """Run the one-worker curriculum with sequential episodes and checkpoints."""
    trials = options.trials
    survive = options.survive
    seed = options.seed
    weights_path = options.weights_path
    display = options.display
    fresh = options.fresh
    max_batches = options.max_batches
    q_contributions_diagnostic = options.q_contributions_diagnostic
    no_training = options.no_training
    workers = options.workers
    guis = options.guis
    assert guis is not None
    eval_trials = options.eval_trials
    history_path = options.history_path
    merge_strategy = options.merge_strategy
    eval_timeout_seconds = options.eval_timeout_seconds
    curriculum = options.curriculum
    stop_after_drills = options.stop_after_drills
    focus_batches = options.focus_batches
    drill_refresh_after = options.drill_refresh_after
    drill_refresh_trials = options.drill_refresh_trials
    worker_details = options.worker_details
    evaluation_runner = options.evaluation_runner
    agent_factory = options.agent_factory
    game_factory = options.game_factory
    episode_runner = options.episode_runner
    agent_type = options.agent_type
    dqn_updates_per_batch = options.dqn_updates_per_batch

    _configure_runtime(display)
    if curriculum == "drills":
        validate_drill_maps()
    game_factory = game_factory or _build_variant_game
    episode_runner = episode_runner or partial(
        _run_episode,
        worker_details=worker_details,
    )
    weights_path = Path(weights_path)
    variant_items = progression()
    drill_items = drill_progression()
    if agent_type == "q":
        agent_factory = agent_factory or _create_qagent
        weights, save_agent, can_save = _load_initial_weights(
            weights_path,
            fresh,
            agent_factory,
        )
        last_valid_weights = dict(weights)
        dqn_agent = None
    else:
        from team01.agent.deep_q_learning import checkpoint_snapshots_equal

        agent_factory = agent_factory or _create_dqn_agent
        dqn_agent, can_save = _load_dqn_training_state(
            weights_path,
            fresh,
            agent_factory,
            options.dqn_learning_rate,
            options.dqn_epsilon,
        )
        weights = None
        save_agent = dqn_agent
        last_valid_weights = {}

        def agent_factory():
            dqn_agent.x = 0
            dqn_agent.y = 0
            return dqn_agent

    summary = TrainingSummary()
    engine = CurriculumEngine(
        trials=trials,
        survive=survive,
        seed=seed,
        eval_trials=eval_trials,
        curriculum=curriculum,
        focus_batches=focus_batches,
        drill_refresh_after=drill_refresh_after,
        no_training=no_training,
        start_variant=options.start_variant,
    )
    summary.decision_trace = engine.trace
    phase = engine.phase
    active_items, item_by_number = _curriculum_items(phase)
    unlocked_items = engine.unlocked_items
    phase_batch = engine.phase_batch
    phase_seed = engine.phase_seed
    seed_rng = random.Random(phase_seed)
    recent_batches = deque(maxlen=STALE_WINDOW)
    round_counter = 0
    focus_variant: Optional[int] = None
    batches_since_evaluation = 0
    last_evaluation_win_rates: dict[int, float] = {}
    variant_stagnation_trials = 0
    drill_refresh_count = 0
    if history_path is not None:
        _append_history_record(
            history_path,
            {
                "record_type": "run_start",
                "seed": seed,
                "agent": agent_type,
                "workers": workers,
                "trials_per_batch": trials,
                "eval_trials_per_variant": eval_trials,
                "eval_timeout_seconds": eval_timeout_seconds,
                "focus_batches": focus_batches,
                "drill_refresh_after": drill_refresh_after,
                "drill_refresh_trials": drill_refresh_trials,
                "merge_strategy": "sequential",
                "curriculum_mode": curriculum,
                "stop_after_drills": stop_after_drills,
                **_phase_history_fields(phase, unlocked_items, ()),
                "fresh": fresh,
                "no_training": no_training,
            },
        )
    active_agent = None

    try:
        print(
            "Project 2 Progressive "
            f"{'Q-Learning' if agent_type == 'q' else 'DQN'} Training",
            flush=True,
        )
        if agent_type == "dqn":
            print("Agent: dqn", flush=True)
        print(f"Map: {DEFAULT_MAP_PATH}", flush=True)
        print(f"Trials per batch: {trials}", flush=True)
        print(f"Successes required: {survive}", flush=True)
        print(f"Base seed: {seed}", flush=True)
        print(f"Curriculum: {curriculum}", flush=True)
        print(f"Focus batches: {focus_batches}", flush=True)
        print(
            f"Drill refresh: every {drill_refresh_after} stalled variant trials "
            f"({drill_refresh_trials} episodes per drill)"
            if curriculum == "drills" and drill_refresh_after > 0 and not no_training
            else "Drill refresh: disabled",
            flush=True,
        )
        print(f"Display: {'Enabled' if display else 'Disabled'}", flush=True)
        print(
            "Learning: Disabled (greedy evaluation)"
            if no_training
            else "Learning: Enabled",
            flush=True,
        )
        if agent_type == "q":
            print(
                f"Q contribution diagnostic: "
                f"{'Enabled' if q_contributions_diagnostic else 'Disabled'}",
                flush=True,
            )
        else:
            print(
                f"Maximum DQN updates per batch: {dqn_updates_per_batch}",
                flush=True,
            )

        while not summary.completed:
            if max_batches is not None and summary.completed_batches >= max_batches:
                summary.stopped_reason = "configured batch limit reached"
                break

            batch_number = summary.completed_batches + 1
            summary.current_batch = batch_number
            batch_plan = engine.plan_batch()
            phase_batch = engine.phase_batch
            batch_phase = engine.batch_phase
            batch_unlocked_items = tuple(unlocked_items)
            newest_item = unlocked_items[-1]
            summary.current_stage = item_by_number[newest_item]
            batch_last_evaluation_win_rates = engine.batch_last_evaluation_win_rates
            batch_focus_variant = engine.batch_focus_variant
            focus_batch_index = (
                batches_since_evaluation + 1 if batch_focus_variant is not None else None
            )
            training_mode = (
                "weakest_variant_focus"
                if batch_focus_variant is not None
                else ("read_only" if no_training else f"progressive_{phase}")
            )
            episode_items = batch_plan.episode_items
            distribution = batch_plan.distribution
            item_prefix = "Drill" if phase == "drill" else "Variant"
            item_names = {
                number: item.name.removeprefix(f"{item_prefix} {number}: ")
                for number, item in item_by_number.items()
            }
            unlocked_label = ", ".join(
                f"{item_prefix[0]}{number}" if phase == "drill" else str(number)
                for number in unlocked_items
            )
            newest_label = (
                f"{item_prefix[0]}{newest_item}"
                if phase == "drill"
                else str(newest_item)
            )
            batch_heading = (
                "FOCUSED VARIANT TRAINING"
                if batch_focus_variant is not None
                else f"{phase.upper()} TRAINING"
            )
            item_heading = (
                f"Focus variant: V{batch_focus_variant} "
                f"{item_names[batch_focus_variant]}"
                if batch_focus_variant is not None
                else f"Newest {item_prefix.lower()}: {newest_label} "
                f"{item_names[newest_item]}"
            )
            print(
                f"\n{'=' * 54}\nBatch {batch_number} | "
                f"{batch_heading} | {trials} total episodes\n"
                f"Unlocked {item_prefix.lower()}s: {unlocked_label}\n"
                f"{item_heading}\nTraining distribution:",
                flush=True,
            )
            if batch_focus_variant is not None:
                print(
                    f"Frozen evaluation focus: V{batch_focus_variant} "
                    f"{item_by_number[batch_focus_variant].name}\n"
                    "Last frozen win rates:",
                    flush=True,
                )
                for variant in range(1, len(variant_items) + 1):
                    print(
                        f"V{variant}: "
                        f"{last_evaluation_win_rates.get(variant, 0.0) * 100:.1f}%",
                        flush=True,
                    )
                print(f"Focus batch {focus_batch_index}/{focus_batches}", flush=True)
            for item_number, count in distribution.items():
                print(
                    f"{item_prefix[0]}{item_number} {item_names[item_number]}: {count}",
                    flush=True,
                )

            batch_results: list[tuple[int, object]] = []
            batch_round_metrics = []
            dqn_episode_results = []
            dqn_batch_metrics = None
            batch_stopped = False
            parallel_dqn_assignments = None
            parallel_dqn_results = None

            if agent_type == "dqn" and workers > 1 and not no_training:
                parallel_dqn_assignments = tuple(
                    (
                        item_by_number[item_number],
                        seed_rng.randint(0, 2**32 - 1),
                    )
                    for item_number in episode_items
                )
                parallel_dqn_results = _run_dqn_rollout_batch(
                    agent=dqn_agent,
                    assignments=parallel_dqn_assignments,
                    workers=workers,
                    guis=guis,
                    batch_number=batch_number,
                    epsilon=dqn_agent.epsilon,
                    worker_details=worker_details,
                )
                valid_counts, missing_items = _dqn_rollout_coverage(
                    episode_items,
                    parallel_dqn_results,
                )
                failed_results = [
                    result
                    for result in parallel_dqn_results
                    if not _dqn_result_is_numerically_stable(result)
                ]
                for result in failed_results:
                    print(
                        f"WARNING: DQN worker {result.worker_id} trial "
                        f"{result.trial_index} ({result.challenge_number}) "
                        f"{result.outcome}: {result.error or 'invalid rollout result'}",
                        flush=True,
                    )
                if missing_items:
                    labels = ", ".join(f"{item_prefix[0]}{item}" for item in missing_items)
                    summary.stopped_reason = (
                        "DQN rollout batch had no valid result for assigned "
                        f"curriculum item(s): {labels}; no replay updates or "
                        "checkpoint save performed"
                    )
                    print(f"STOP: {summary.stopped_reason}", flush=True)
                    batch_stopped = True
                    _append_history_record(
                        history_path,
                        {
                            "record_type": "dqn_rollout_failure",
                            "agent": "dqn",
                            "batch": batch_number,
                            "worker_limit": min(workers, len(parallel_dqn_results)),
                            "episodes_assigned": len(parallel_dqn_results),
                            "episodes_valid": len(parallel_dqn_results)
                            - len(failed_results),
                            "episode_errors": len(failed_results),
                            "missing_curriculum_items": list(missing_items),
                        },
                    )
                if missing_items:
                    print(
                        f"Parallel DQN rollout: "
                        f"{len(parallel_dqn_results) - len(failed_results)}/"
                        f"{len(parallel_dqn_results)} episodes valid | "
                        f"Worker limit: {min(workers, len(parallel_dqn_results))} | "
                        f"Episode errors: {len(failed_results)}",
                        flush=True,
                    )

            for trial_number, item_number in enumerate(episode_items, start=1):
                if batch_stopped:
                    break
                round_counter += 1
                challenge = item_by_number[item_number]
                summary.current_stage = challenge
                round_start_weights = dict(weights) if weights is not None else None
                label = (
                    f"Stage {challenge.number} | Batch {batch_number} | "
                    f"Trial {trial_number}/{trials} | Episode {summary.completed_trials + 1}"
                )
                if parallel_dqn_results is not None:
                    trial_seed = parallel_dqn_assignments[trial_number - 1][1]
                    active_agent = dqn_agent
                    game = None
                    parallel_result = parallel_dqn_results[trial_number - 1]
                else:
                    trial_seed = seed_rng.randint(0, 2**32 - 1)
                    game, active_agent = game_factory(
                        challenge,
                        trial_seed,
                        weights,
                        agent_factory,
                    )
                    parallel_result = None
                frozen_dqn_snapshot = (
                    dqn_agent.get_checkpoint_snapshot()
                    if agent_type == "dqn" and no_training
                    else None
                )
                frozen_dqn_replay_size = (
                    len(dqn_agent.replay_buffer)
                    if agent_type == "dqn" and no_training
                    else None
                )
                if agent_type == "dqn":
                    if parallel_result is None:
                        if no_training:
                            active_agent.set_learning(no_training=True, epsilon=0.0)
                        else:
                            active_agent.set_rollout(
                                epsilon=active_agent.epsilon,
                                collect_experience=True,
                            )
                else:
                    active_agent.set_learning(
                        no_training,
                        epsilon=active_agent.epsilon,
                    )
                active_agent.q_contributions_diagnostic = q_contributions_diagnostic
                updates_before = getattr(active_agent, "td_update_count", 0)
                result = (
                    parallel_result
                    if parallel_result is not None
                    else episode_runner(
                        game,
                        active_agent,
                        summary.completed_trials + 1,
                        label,
                        display,
                    )
                )
                if parallel_result is None:
                    summary.completed_trials += 1
                batch_results.append((item_number, result))

                if agent_type == "dqn":
                    if no_training:
                        dqn_agent.episodes = frozen_dqn_snapshot["episodes"]
                    if no_training and (
                        not checkpoint_snapshots_equal(
                            frozen_dqn_snapshot, dqn_agent.get_checkpoint_snapshot()
                        )
                        or len(dqn_agent.replay_buffer) != frozen_dqn_replay_size
                    ):
                        raise RuntimeError(
                            "Frozen DQN episode mutated training learner state"
                        )
                    stable = _dqn_result_is_numerically_stable(result)
                    if stable and not no_training:
                        dqn_episode_results.append(result)
                    if parallel_result is not None and stable:
                        summary.completed_trials += 1
                        dqn_agent.episodes += 1
                else:
                    frozen_unchanged = (
                        not no_training
                        or (
                            active_agent.weights == round_start_weights
                            and getattr(active_agent, "td_update_count", 0) == updates_before
                        )
                    )
                    stable = (
                        _result_is_numerically_stable(result, active_agent.weights)
                        and frozen_unchanged
                    )
                    if stable and not no_training:
                        weights = active_agent.weights
                        _save_weights(active_agent, weights, weights_path)
                        save_agent = active_agent
                        last_valid_weights = dict(weights)
                        can_save = True

                if worker_details:
                    print(
                        f"[Trial {summary.completed_trials} | "
                        f"Worker {getattr(result, 'worker_id', 0)} | "
                        f"{item_prefix} {challenge.number} | "
                        f"Batch {batch_number} {trial_number}/{trials}] "
                        f"Outcome: {result.outcome} | Ticks: {result.ticks} | "
                        f"Reward: {result.total_reward:.0f} | "
                        f"Bombs: {result.bombs_placed} | "
                        + (
                            "DQN transitions: "
                            f"{len(getattr(result, 'transitions', ()))}"
                            if agent_type == "dqn"
                            else (
                                "Weights saved: "
                                f"{'no (evaluation mode)' if no_training else ('yes' if stable else 'no (unstable values)')}"
                            )
                        ),
                        flush=True,
                    )

                if not stable:
                    if parallel_result is not None:
                        _append_history_record(
                            history_path,
                            {
                                "record_type": "training_round",
                                "agent": "dqn",
                                "batch": batch_number,
                                "round": round_counter,
                                **_phase_history_fields(
                                    batch_phase, batch_unlocked_items, (item_number,)
                                ),
                                "training_enabled": True,
                                "training_mode": training_mode,
                                "worker_id": result.worker_id,
                                "trial_index": result.trial_index,
                                "seed": result.seed,
                                "outcome": result.outcome,
                                "error": result.error,
                            },
                        )
                        continue
                    if no_training:
                        summary.stopped_reason = (
                            "frozen DQN evaluation returned non-finite values"
                            if agent_type == "dqn"
                            else (
                                "frozen evaluation changed weights, updated TD state, "
                                "or returned non-finite values"
                            )
                        )
                    else:
                        if weights is not None:
                            weights.clear()
                            weights.update(last_valid_weights)
                        summary.stopped_reason = (
                            f"numerical instability in stage {challenge.number}, "
                            f"batch {batch_number}, trial {trial_number}"
                        )
                    print(f"STOP: {summary.stopped_reason}", flush=True)
                    active_agent = None
                    batch_stopped = True
                    break

                if agent_type == "q":
                    merged_weights = (
                        round_start_weights
                        if no_training
                        else dict(active_agent.weights)
                    )
                    diagnostics = _weight_update_diagnostics(
                        round_start_weights,
                        [dict(active_agent.weights)],
                        merged_weights,
                        [
                            (
                                result.mean_abs_td_error,
                                getattr(active_agent, "td_update_count", 0),
                            )
                        ],
                        result.max_abs_q,
                    )
                    _print_learning_progress(diagnostics)
                    batch_round_metrics.append(diagnostics)
                    _append_history_record(
                        history_path,
                        {
                            "record_type": "training_round",
                            "agent": "q",
                            "batch": batch_number,
                            "round": round_counter,
                            **_phase_history_fields(
                                batch_phase, batch_unlocked_items, (item_number,)
                            ),
                            "training_enabled": not no_training,
                            "training_mode": training_mode,
                            "focus_variant": batch_focus_variant,
                            **diagnostics,
                            "weights": dict(merged_weights),
                        },
                    )
                else:
                    _append_history_record(
                        history_path,
                        {
                            "record_type": "training_round",
                            "agent": "dqn",
                            "batch": batch_number,
                            "round": round_counter,
                            **_phase_history_fields(
                                batch_phase, batch_unlocked_items, (item_number,)
                            ),
                            "training_enabled": not no_training,
                            "training_mode": training_mode,
                            "focus_variant": batch_focus_variant,
                            "outcome": result.outcome,
                            "ticks": result.ticks,
                            "total_reward": result.total_reward,
                            "bombs_placed": result.bombs_placed,
                            "transitions_collected": len(
                                getattr(result, "transitions", ())
                            ),
                            "max_abs_q": result.max_abs_q,
                        },
                    )

            if agent_type == "dqn" and not no_training and not batch_stopped:
                dqn_agent.set_learning(
                    no_training=False,
                    epsilon=dqn_agent.epsilon,
                )
                dqn_batch_metrics = _consume_dqn_episode_results(
                    dqn_agent,
                    dqn_episode_results,
                    dqn_updates_per_batch,
                )
                dqn_batch_metrics.update(
                    worker_limit=min(
                        workers,
                        (
                            len(parallel_dqn_results)
                            if parallel_dqn_results is not None
                            else len(episode_items)
                        ),
                    ),
                    episodes_assigned=(
                        len(parallel_dqn_results)
                        if parallel_dqn_results is not None
                        else len(episode_items)
                    ),
                    episodes_valid=len(dqn_episode_results),
                    episode_errors=(
                        len(parallel_dqn_results) - len(dqn_episode_results)
                        if parallel_dqn_results is not None
                        else 0
                    ),
                )
                _print_dqn_batch_summary(dqn_batch_metrics)
                if len(batch_results) == len(episode_items):
                    dqn_agent.save_checkpoint(weights_path)
                    can_save = True
                    summary.weights_saved = True

            if batch_stopped:
                break

            newest_count = distribution.get(newest_item, 0)
            required_wins = batch_plan.required_newest_wins
            _print_mixed_batch_summary(
                episode_items,
                batch_results,
                newest_item,
                required_wins,
                phase=batch_phase,
                item_names=item_names,
                show_newest=batch_focus_variant is None,
            )
            if batch_focus_variant is not None:
                _print_focus_variant_results(batch_focus_variant, batch_results)
            summary.completed_batches += 1
            decision = engine.record_batch(
                batch_results,
                trained_episodes=(
                    len(dqn_episode_results) if agent_type == "dqn" else None
                ),
            )
            phase = engine.phase
            active_items, item_by_number = _curriculum_items(phase)
            unlocked_items = engine.unlocked_items
            phase_batch = engine.phase_batch
            phase_seed = engine.phase_seed
            batches_since_evaluation = engine.batches_since_evaluation
            variant_stagnation_trials = engine.stagnation_trials
            newest_wins = decision.newest_wins
            phase_transitioned = decision.phase_transitioned
            stage_unlocked = decision.stage_unlocked

            if batch_focus_variant is not None:
                pass
            elif stage_unlocked:
                summary.stages_passed += 1
                summary.current_stage = item_by_number[newest_item + 1]
                if decision.progress_detected:
                    print(
                        f"Variant progress detected: V{newest_item + 1} unlocked. "
                        "Stagnation counter reset.",
                        flush=True,
                    )
                print(
                    f"Newest {item_prefix.lower()} {newest_item} passed its training "
                    f"ratio ({newest_wins}/{newest_count}); unlocked "
                    f"{item_prefix[0]}{newest_item + 1}.",
                    flush=True,
                )
            elif phase_transitioned:
                print(
                    f"\n{'=' * 54}\nDRILL CURRICULUM COMPLETE\n"
                    f"{'=' * 54}\nAll skill drills passed.\n\n"
                    "Carrying learned weights into the Project 2 variant curriculum.\n"
                    "Starting Variant 1.\n"
                    f"{'=' * 54}",
                    flush=True,
                )
                seed_rng = random.Random(phase_seed)
                recent_batches.clear()
                summary.current_stage = item_by_number[1]
            elif (
                batch_phase == "variant"
                and newest_item == len(variant_items)
                and newest_wins >= required_wins
            ):
                print(
                    f"Newest variant V{newest_item} met its training ratio; "
                    "final frozen evaluation still controls completion.",
                    flush=True,
                )
            else:
                print(
                    f"Newest {item_prefix.lower()} {newest_item} did not unlock "
                    "another stage "
                    f"({newest_wins}/{newest_count}, need {required_wins}).",
                    flush=True,
                )

            if (
                batch_phase == "variant"
                and not no_training
                and not stage_unlocked
            ):
                print(
                    f"Variant progress: trials since last progress: "
                    f"{variant_stagnation_trials}/"
                    f"{drill_refresh_after if drill_refresh_after else 'disabled'}",
                    flush=True,
                )

            evaluation_performed = False
            if batch_phase == "drill":
                print(
                    "Frozen evaluation skipped: drill training does not use frozen evaluation.",
                    flush=True,
                )
                evaluation_win_rates = {}
                final_complete = False
                evaluation_details = {
                    "timed_out_counts": {},
                    "skipped_variants": [],
                }
            elif len(unlocked_items) < len(variant_items):
                print(
                    "Frozen evaluation skipped: all five variants are not yet unlocked.",
                    flush=True,
                )
                evaluation_win_rates = {}
                final_complete = False
                evaluation_details = {
                    "timed_out_counts": {},
                    "skipped_variants": [],
                }
            elif decision.evaluation_due:
                print(
                    "All variants unlocked. Beginning frozen evaluation.",
                    flush=True,
                )
                (
                    evaluation_win_rates,
                    _,
                    final_complete,
                    evaluation_details,
                ) = _evaluate_batch(
                    unlocked_challenges=tuple(
                        item_by_number[variant] for variant in unlocked_items
                    ),
                    weights=weights,
                    eval_trials=eval_trials,
                    seed=seed,
                    workers=workers,
                    batch_number=batch_number,
                    learning_parameters=_default_learning_parameters(),
                    evaluation_runner=evaluation_runner,
                    history_path=history_path,
                    eval_timeout_seconds=eval_timeout_seconds,
                    worker_details=worker_details,
                    frozen_evaluator=(
                        partial(
                            _evaluate_dqn_policy,
                            dqn_agent,
                            game_factory=game_factory,
                            episode_runner=episode_runner,
                            timeout_seconds=eval_timeout_seconds,
                        )
                        if agent_type == "dqn" and evaluation_runner is None
                        else None
                    ),
                )
                evaluation_performed = True
                eval_decision = engine.record_evaluation(
                    evaluation_win_rates, complete=final_complete
                )
                variant_stagnation_trials = engine.stagnation_trials
                if final_complete:
                    print(
                        f"All variants meet >="
                        f"{FINAL_WIN_RATE_THRESHOLD * 100:.0f}%. Training complete.",
                        flush=True,
                    )
                elif not no_training:
                    if not batch_last_evaluation_win_rates:
                        print(
                            "Initial all-variant frozen baseline established; "
                            "stagnation counter reset.",
                            flush=True,
                        )
                    elif eval_decision.improved:
                        print(
                            "Frozen progress detected in passing count, bottleneck, "
                            "or focused variant; stagnation counter reset.",
                            flush=True,
                        )
                    else:
                        print(
                            f"Frozen progress not detected; variant stagnation "
                            f"remains {variant_stagnation_trials} trials.",
                            flush=True,
                        )
                last_evaluation_win_rates = dict(engine.last_evaluation_win_rates)
                if not final_complete:
                    previous_focus = eval_decision.previous_focus
                    focus_variant = eval_decision.focus_variant
                    if previous_focus != focus_variant:
                        _append_history_record(
                            history_path,
                            {
                                "record_type": "focus_change",
                                "previous_focus": previous_focus,
                                "new_focus": focus_variant,
                                "evaluation_win_rates": {
                                    str(variant): rate
                                    for variant, rate in sorted(
                                        evaluation_win_rates.items()
                                    )
                                },
                            },
                        )
                    _print_focus_selection(
                        evaluation_win_rates, focus_variant, focus_batches
                    )
                    batches_since_evaluation = engine.batches_since_evaluation
            else:
                evaluation_win_rates = dict(last_evaluation_win_rates)
                final_complete = False
                evaluation_details = {
                    "timed_out_counts": {},
                    "skipped_variants": [],
                }
                print(
                    f"Frozen reevaluation deferred: focused batch "
                    f"{batches_since_evaluation}/{focus_batches} complete.",
                    flush=True,
                )
            evaluation_score = (
                math.fsum(evaluation_win_rates.values()) / len(evaluation_win_rates)
                if evaluation_win_rates
                else None
            )
            if agent_type == "q":
                relative_deltas = [
                    item["relative_weight_delta_l2"]
                    for item in batch_round_metrics
                    if item["relative_weight_delta_l2"] is not None
                ]
                batch_metrics = {
                    "relative_weight_delta_l2": (
                        math.fsum(relative_deltas) / len(relative_deltas)
                        if relative_deltas
                        else None
                    ),
                    "mean_worker_delta_l2": (
                        math.fsum(
                            item["mean_worker_delta_l2"]
                            for item in batch_round_metrics
                        )
                        / max(1, len(batch_round_metrics))
                    ),
                    "mean_abs_td_error": (
                        math.fsum(
                            item["mean_abs_td_error"]
                            for item in batch_round_metrics
                        )
                        / max(1, len(batch_round_metrics))
                    ),
                    "canonical_delta_l2": (
                        math.fsum(
                            item["canonical_delta_l2"]
                            for item in batch_round_metrics
                        )
                        / max(1, len(batch_round_metrics))
                    ),
                    "update_agreement_ratio": (
                        math.fsum(
                            item["update_agreement_ratio"]
                            for item in batch_round_metrics
                        )
                        / max(1, len(batch_round_metrics))
                    ),
                    "evaluation_score": evaluation_score,
                }
                _print_batch_learning_summary(
                    batch_metrics,
                    len(batch_round_metrics),
                    len(batch_round_metrics),
                    "none" if no_training else "sequential",
                )
            else:
                batch_metrics = {
                    "relative_weight_delta_l2": None,
                    "mean_worker_delta_l2": None,
                    "mean_abs_td_error": (
                        dqn_batch_metrics["mean_abs_td_error"]
                        if dqn_batch_metrics is not None
                        else None
                    ),
                    "canonical_delta_l2": None,
                    "update_agreement_ratio": None,
                    "evaluation_score": evaluation_score,
                    **(dqn_batch_metrics or {}),
                }
            _append_history_record(
                history_path,
                {
                    "record_type": "training_batch",
                    "agent": agent_type,
                    "worker_limit": (
                        dqn_batch_metrics["worker_limit"]
                        if dqn_batch_metrics is not None
                        else workers
                    ),
                    "batch": batch_number,
                    "merge_strategy": "sequential",
                    "training_mode": training_mode,
                    "focus_variant": batch_focus_variant,
                    "variant_stagnation_trials": variant_stagnation_trials,
                    "drill_refresh_count": drill_refresh_count,
                    "last_evaluation_win_rates": {
                        str(variant): rate
                        for variant, rate in sorted(
                            batch_last_evaluation_win_rates.items()
                        )
                    },
                    "focus_batch_index": focus_batch_index,
                    "focus_batches_per_cycle": focus_batches,
                    "evaluation_performed": evaluation_performed,
                    **_phase_history_fields(
                        batch_phase, batch_unlocked_items, episode_items
                    ),
                    "training_wins_by_item": {
                        str(item_number): sum(
                            result.outcome == "WON"
                            for result_item, result in batch_results
                            if result_item == item_number
                        )
                        for item_number in distribution
                    },
                    "evaluation_win_rates": {
                        str(variant): rate
                        for variant, rate in sorted(evaluation_win_rates.items())
                    },
                    "evaluation_timeout_counts": {
                        str(variant): count
                        for variant, count in sorted(
                            evaluation_details["timed_out_counts"].items()
                        )
                    },
                    "evaluation_skipped_variants": evaluation_details[
                        "skipped_variants"
                    ],
                    **batch_metrics,
                },
            )
            refresh_due = engine.refresh_due(
                evaluation_performed=evaluation_performed
            )
            if refresh_due:
                refresh_number = engine.drill_refresh_count + 1
                # Keep unlock and focus progress while refreshing the drill skills.
                saved_variant_state = VariantResumeState(
                    unlocked_variants=tuple(unlocked_items),
                    variant_batch=phase_batch,
                    variant_seed=phase_seed,
                    focus_variant=focus_variant,
                    batches_since_evaluation=batches_since_evaluation,
                    evaluation_win_rates=dict(last_evaluation_win_rates),
                    current_stage=summary.current_stage,
                    completed_batches=summary.completed_batches,
                )
                refresh_reason = (
                    "stage progression did not advance"
                    if len(unlocked_items) < len(variant_items)
                    else "frozen evaluation showed no meaningful variant progress"
                )
                print(
                    f"\n{'=' * 54}\nDRILL REFRESH TRIGGERED\n"
                    f"{'=' * 54}\n"
                    f"Reason: {variant_stagnation_trials} variant training episodes "
                    f"without meaningful progress; {refresh_reason}.\n"
                    f"Unlocked: {', '.join(f'V{item}' for item in unlocked_items)}\n"
                    f"Newest: V{newest_item} - {item_by_number[newest_item].name}\n"
                    f"Current focus: "
                    f"{f'V{focus_variant}' if focus_variant is not None else 'none'}",
                    flush=True,
                )
                for drill in drill_items:
                    print(
                        f"D{drill.number} - {drill.name}: "
                        f"{drill_refresh_trials} episodes",
                        flush=True,
                    )
                _append_history_record(
                    history_path,
                    {
                        "record_type": "drill_refresh_start",
                        "refresh_number": refresh_number,
                        "trigger_variant_trials": drill_refresh_after,
                        "variant_stagnation_trials": variant_stagnation_trials,
                        "phase_before_refresh": "variant",
                        "unlocked_variants": list(unlocked_items),
                        "newest_variant": newest_item,
                        "focus_variant": focus_variant,
                        "last_evaluation_win_rates": {
                            str(variant): rate
                            for variant, rate in sorted(
                                last_evaluation_win_rates.items()
                            )
                        },
                    },
                )
                refresh_episode_start = summary.completed_trials
                round_counter, weights, save_agent, drill_results = (
                    _run_sequential_drill_refresh(
                        drills=drill_items,
                        refresh_trials=drill_refresh_trials,
                        seed=phase_seed,
                        refresh_number=refresh_number,
                        batch_number=batch_number,
                        round_counter=round_counter,
                        weights=weights,
                        save_agent=save_agent,
                        weights_path=weights_path,
                        summary=summary,
                        agent_factory=agent_factory,
                        game_factory=game_factory,
                        episode_runner=episode_runner,
                        display=display,
                        q_contributions_diagnostic=q_contributions_diagnostic,
                        no_training=no_training,
                        history_path=history_path,
                        worker_details=worker_details,
                        agent_type=agent_type,
                        dqn_updates_per_batch=dqn_updates_per_batch,
                        workers=workers,
                        guis=guis,
                    )
                )
                refresh_failed = any(
                    result.get("batch_rejected", False)
                    for result in drill_results.values()
                )
                if weights is not None:
                    last_valid_weights = dict(weights)
                can_save = can_save or summary.weights_saved
                active_agent = None
                engine.refresh_completed()
                drill_refresh_count = engine.drill_refresh_count
                _append_history_record(
                    history_path,
                    {
                        "record_type": "drill_refresh_complete",
                        "refresh_number": refresh_number,
                        "drill_results": {
                            str(drill): result
                            for drill, result in sorted(drill_results.items())
                        },
                        "episodes_completed": summary.completed_trials
                        - refresh_episode_start,
                        "weights_saved": summary.weights_saved,
                        "unlocked_variants": list(saved_variant_state.unlocked_variants),
                        "newest_variant": saved_variant_state.unlocked_variants[-1],
                        "focus_variant": saved_variant_state.focus_variant,
                        "variant_stagnation_trials": 0,
                    },
                )
                phase = "variant"
                active_items = variant_items
                item_by_number = {item.number: item for item in variant_items}
                unlocked_items = list(saved_variant_state.unlocked_variants)
                phase_batch = saved_variant_state.variant_batch
                phase_seed = saved_variant_state.variant_seed
                focus_variant = saved_variant_state.focus_variant
                batches_since_evaluation = saved_variant_state.batches_since_evaluation
                last_evaluation_win_rates = dict(
                    saved_variant_state.evaluation_win_rates
                )
                summary.current_stage = saved_variant_state.current_stage
                if summary.completed_batches != saved_variant_state.completed_batches:
                    raise RuntimeError("drill refresh changed the variant batch counter")
                variant_stagnation_trials = 0
                newest_name = item_by_number[unlocked_items[-1]].name.removeprefix(
                    f"Variant {unlocked_items[-1]}: "
                )
                print(
                    "DRILL REFRESH COMPLETE; returning to variant training.\n"
                    f"Unlocked variants: {', '.join(f'V{item}' for item in unlocked_items)}\n"
                    f"Newest variant: V{unlocked_items[-1]} "
                    f"{newest_name}\n"
                    f"Current focus: "
                    f"{f'V{focus_variant}' if focus_variant is not None else 'none'}\n"
                    "Variant stagnation counter reset to 0.",
                    flush=True,
                )
                if refresh_failed:
                    break
            if not no_training and evaluation_performed:
                _maybe_warn_stale_policy(
                    batch_metrics,
                    recent_batches,
                    batch_phase == "variant"
                    and len(unlocked_items) == len(variant_items),
                    final_complete,
                )
            elif no_training:
                recent_batches.clear()
            if phase_transitioned:
                _append_history_record(
                    history_path,
                    {
                        "record_type": "phase_transition",
                        "from": "drills",
                        "to": "variants",
                        "completed_drills": [item.number for item in drill_items],
                        "completed_batches": summary.completed_batches,
                        "completed_trials": summary.completed_trials,
                    },
                )
                if stop_after_drills:
                    summary.stopped_reason = (
                        "drill pretraining complete; stopped by request"
                    )
                    print(
                        "Drill pretraining complete; stopping by request.\n"
                        "Project 2 variant completion has NOT been evaluated.",
                        flush=True,
                    )
                    break
            if (
                no_training
                and evaluation_performed
                and batch_phase == "variant"
                and len(unlocked_items) == len(variant_items)
                and not final_complete
            ):
                summary.stopped_reason = (
                    "no-training evaluation complete; focused training is disabled"
                )
                print(
                    "No-training mode: frozen evaluation complete; "
                    "focused training is disabled.",
                    flush=True,
                )
                break
            if final_complete:
                summary.completed = True

    except KeyboardInterrupt:
        summary.interrupted = True
    except TrainingAborted as error:
        summary.stopped_reason = str(error)
        print(f"Training stopped: {error}", flush=True)
    finally:
        if (
            agent_type == "q"
            and active_agent is not None
            and not no_training
        ):
            active_weights = active_agent.weights
            active_updates = getattr(active_agent, "td_update_count", 0)
            if _weights_are_finite(active_weights) and (
                can_save or active_updates > 0
            ):
                weights = active_weights
                can_save = True

        if no_training:
            print(
                (
                    "Evaluation mode: weights were not modified or saved. "
                    if agent_type == "q"
                    else "Evaluation mode: training state was not modified or saved. "
                )
                + f"Loaded checkpoint remains at: {weights_path}",
                flush=True,
            )
        elif can_save and not (
            agent_type == "dqn" and dqn_agent._central_update_in_progress
        ):
            if agent_type == "dqn":
                dqn_agent.save_checkpoint(weights_path)
            elif agent_type == "dqn" and dqn_agent._central_update_in_progress:
                print(
                    "DQN update was interrupted before a stable checkpoint boundary; "
                    f"the previous checkpoint was left unchanged at: {weights_path}",
                    flush=True,
                )
            else:
                if not _weights_are_finite(weights):
                    weights.clear()
                    weights.update(last_valid_weights)
                _save_weights(save_agent, weights, weights_path)
            summary.weights_saved = True
            print(
                f"Current learned "
                f"{'DQN checkpoint' if agent_type == 'dqn' else 'weights'} "
                f"saved to: {weights_path}",
                flush=True,
            )
        else:
            print(
                "No trained checkpoint was produced; existing training state was left "
                f"unchanged at: {weights_path}",
                flush=True,
            )

    if summary.interrupted:
        print("\nTraining interrupted by user.", flush=True)
        print(f"Completed trials: {summary.completed_trials}", flush=True)
        if summary.current_stage is not None:
            print(
                f"Current stage: Variant {summary.current_stage.number} "
                f"({summary.current_stage.name})",
                flush=True,
            )
        if no_training:
            print(f"Evaluation interrupted; loaded checkpoint unchanged at: {weights_path}", flush=True)
        elif agent_type == "q":
            print(f"Current weights saved to: {weights_path if summary.weights_saved else 'not saved'}", flush=True)
        else:
            print(
                f"Current checkpoint saved to: "
                f"{weights_path if summary.weights_saved else 'not saved'}",
                flush=True,
            )
    elif summary.completed:
        print("\nPROGRESSIVE TRAINING COMPLETE", flush=True)
        print("All existing challenge levels passed.", flush=True)
        print(f"Final stage: Variant {variant_items[-1].number}", flush=True)
        print(f"Total training episodes: {summary.completed_trials}", flush=True)
        if no_training:
            print(f"Evaluation complete; checkpoint unchanged at: {weights_path}", flush=True)
        elif agent_type == "q":
            print(f"Final weights saved to: {weights_path}", flush=True)
        else:
            print(f"Final checkpoint saved to: {weights_path}", flush=True)
    elif summary.stopped_reason is not None:
        print(f"Training stopped: {summary.stopped_reason}", flush=True)
        print(f"Completed trials: {summary.completed_trials}", flush=True)
        if agent_type == "q":
            print(
                f"Weights: {weights_path if summary.weights_saved else 'not saved'}",
                flush=True,
            )
        else:
            print(
                f"Checkpoint: {weights_path if summary.weights_saved else 'not saved'}",
                flush=True,
            )

    return summary


# ---------------------------------------------------------------------------
# Command-line interface

def parse_args(argv=None):
    """Parse and validate the training command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        dest="agent_type",
        choices=("q", "dqn"),
        default="q",
        help="Training backend (Q-learning remains the default).",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Total number of training trials to run.")
    parser.add_argument("--survive", type=int, default=DEFAULT_SURVIVE, help="Number of top-performing trials to retain after each round.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint path (Q JSON weights or DQN .pt checkpoint).",
    )
    parser.add_argument(
        "--dqn-updates-per-batch",
        type=int,
        default=DQN_UPDATES_PER_BATCH,
        help="Maximum central DQN optimizer updates after each completed rollout batch.",
    )
    parser.add_argument(
        "--start-variant",
        type=int,
        default=None,
        help=(
            "DQN only: begin in the variant phase with V1..N already unlocked "
            "(e.g. 4 resumes at V4). Does not mark N passed."
        ),
    )
    parser.add_argument(
        "--dqn-learning-rate",
        type=float,
        default=None,
        help="DQN only: override the Adam learning rate after loading the checkpoint.",
    )
    parser.add_argument(
        "--dqn-epsilon",
        type=float,
        default=None,
        help="DQN only: override the training exploration epsilon.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent rollout worker processes; does not set episode count.",
    )
    parser.add_argument("--guis", type=int, help="Number of GUI instances to launch for visualization.")
    parser.add_argument("--eval-trials", type=int, default=DEFAULT_EVAL_TRIALS, help="Number of evaluation trials per variant.")
    parser.add_argument(
        "--eval-timeout-seconds",
        type=float,
        default=DEFAULT_EVAL_TIMEOUT_SECONDS,
        help="Maximum wall-clock seconds per frozen-evaluation worker wave.",
    )
    parser.add_argument(
        "--merge-strategy",
        choices=("genetic", "mean"),
        default="genetic",
        help="Parallel worker synchronization strategy.",
    )
    parser.add_argument(
        "--curriculum",
        choices=("variants", "drills"),
        default="variants",
        help="Train existing variants directly or pretrain on skill drills first.",
    )
    parser.add_argument(
        "--stop-after-drills",
        action="store_true",
        help="Stop after drill pretraining and before Project 2 variant training.",
    )
    parser.add_argument(
        "--focus-batches",
        type=int,
        default=DEFAULT_FOCUS_BATCHES,
        help="Focused variant batches to run before frozen reevaluation.",
    )
    parser.add_argument(
        "--drill-refresh-after",
        type=int,
        default=DEFAULT_DRILL_REFRESH_AFTER,
        help=(
            "Completed-batch variant episode attempts without progress before a drill "
            "refresh; cancelled/worker-error tasks follow completed-trial accounting. "
            "0 disables refreshes."
        ),
    )
    parser.add_argument(
        "--drill-refresh-trials",
        type=int,
        default=DEFAULT_DRILL_REFRESH_TRIALS,
        help="Training episodes to run on each drill during refresh.",
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--no-training", action="store_true", help="Run the learned policy without updating weights or using epsilon exploration.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing weights and start from zero.")
    parser.add_argument(
        "--q-contributions",
        action="store_true",
        help="Explain candidate Q-values when PLACE_BOMB is selected or greedily preferred.",
    )
    parser.add_argument("--max-batches", type=int, help="Stop after this many complete batches (useful for short runs).")
    parser.add_argument(
        "--worker-details",
        action="store_true",
        help="Print individual worker results, progress heartbeats, and genetic rankings.",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="display", action="store_true")
    display_group.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=None)
    args = parser.parse_args(argv)

    if args.weights is None:
        args.weights = (
            DEFAULT_WEIGHTS_PATH
            if args.agent_type == "q"
            else DEFAULT_DQN_CHECKPOINT_PATH
        )
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.dqn_updates_per_batch < 0:
        parser.error("--dqn-updates-per-batch must be at least 0")
    if args.eval_trials < 1:
        parser.error("--eval-trials must be at least 1")
    if args.eval_timeout_seconds <= 0 or not math.isfinite(args.eval_timeout_seconds):
        parser.error("--eval-timeout-seconds must be positive and finite")
    if args.stop_after_drills and args.curriculum != "drills":
        parser.error("--stop-after-drills requires --curriculum drills")
    if args.focus_batches < 1:
        parser.error("--focus-batches must be at least 1")
    if args.drill_refresh_after < 0:
        parser.error("--drill-refresh-after must be at least 0")
    if args.drill_refresh_trials < 1:
        parser.error("--drill-refresh-trials must be at least 1")
    if args.fresh and args.no_training:
        parser.error("--fresh cannot be used with --no-training")
    if args.guis is None:
        args.guis = 0 if args.display is False else 1
    elif args.display is False and args.guis > 0:
        parser.error("--no-display conflicts with a positive --guis value")
    elif args.display is True and args.guis == 0:
        parser.error("--display conflicts with --guis 0")
    if args.guis < 0:
        parser.error("--guis must be at least 0")
    if args.guis > args.workers:
        parser.error("--guis cannot exceed --workers")

    args.display = args.guis > 0
    return args


def main(argv=None) -> TrainingSummary:
    """Run progressive training with the parsed command-line options."""
    args = parse_args(argv)
    start_time = time.time()

    summary = run_progressive_training(
        trials=args.trials,
        survive=args.survive,
        seed=args.seed,
        weights_path=args.weights,
        display=args.display,
        fresh=args.fresh,
        max_batches=args.max_batches,
        q_contributions_diagnostic=args.q_contributions,
        workers=args.workers,
        guis=args.guis,
        no_training=args.no_training,
        eval_trials=args.eval_trials,
        history_path=args.history,
        merge_strategy=args.merge_strategy,
        eval_timeout_seconds=args.eval_timeout_seconds,
        curriculum=args.curriculum,
        stop_after_drills=args.stop_after_drills,
        focus_batches=args.focus_batches,
        drill_refresh_after=args.drill_refresh_after,
        drill_refresh_trials=args.drill_refresh_trials,
        worker_details=args.worker_details,
        agent_type=args.agent_type,
        dqn_updates_per_batch=args.dqn_updates_per_batch,
        start_variant=args.start_variant,
        dqn_learning_rate=args.dqn_learning_rate,
        dqn_epsilon=args.dqn_epsilon,
    )
    
    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Training ended:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Elapsed time: {hours:02}:{minutes:02}:{seconds:02}")

    return summary


if __name__ == "__main__":
    main()