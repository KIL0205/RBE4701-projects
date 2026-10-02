"""Progressively train the Project 2 approximate Q-learning agent."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import DEFAULT


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BOMBERMAN_DIR = _ROOT / "Bomberman"
if str(_BOMBERMAN_DIR) not in sys.path:
    sys.path.insert(0, str(_BOMBERMAN_DIR))

DEFAULT_MAP_PATH = Path(__file__).resolve().with_name("map.txt")
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().with_name("q_learning_weights.json")
DEFAULT_TRIALS = 10
DEFAULT_SURVIVE = 5
DEFAULT_DISPLAY = True
HEARTBEAT_INTERVAL = 100
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


@dataclass
class TrainingSummary:
    completed_trials: int = 0
    completed_batches: int = 0
    stages_passed: int = 0
    current_stage: Optional[Challenge] = None
    current_batch: int = 0
    completed: bool = False
    interrupted: bool = False
    stopped_reason: Optional[str] = None
    weights_saved: bool = False


class TrainingAborted(Exception):
    """Raised when the user closes the active game window."""


def progression(map_path: Path = DEFAULT_MAP_PATH) -> tuple[Challenge, ...]:
    """Return the existing Project 2 variants in their documented order."""
    return (
        Challenge(1, "Variant 1: Alone in the world", map_path, ()),
        Challenge(
            2,
            "Variant 2: Random monster",
            map_path,
            (MonsterConfig("stupid", "stupid", "S", 3, 9),),
        ),
        Challenge(
            3,
            "Variant 3: Self-preserving monster",
            map_path,
            (MonsterConfig("smart", "selfpreserving", "S", 3, 9, 1),),
        ),
        Challenge(
            4,
            "Variant 4: Aggressive monster",
            map_path,
            (MonsterConfig("smart", "aggressive", "A", 3, 13, 2),),
        ),
        Challenge(
            5,
            "Variant 5: Stupid and aggressive monsters",
            map_path,
            (
                MonsterConfig("stupid", "stupid", "S", 3, 5),
                MonsterConfig("smart", "aggressive", "A", 3, 13, 1),
            ),
        ),
    )


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


def _create_qagent():
    return _qagent_class()("me", "C", 0, 0)


def _weights_are_finite(weights: dict[str, float]) -> bool:
    return all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in weights.values()
    )


def _build_variant_game(
    challenge: Challenge,
    seed: int,
    weights: dict[str, float],
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
    agent.weights = weights
    game.add_character(agent)
    if not any(
        agent in characters for characters in game.world.characters.values()
    ):
        raise RuntimeError("training agent was not registered in its world")
    return game, agent


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
):
    from team01.eval.train_q_learning import run_episode

    on_tick = _display_callback(game, progress_label) if display else None
    return run_episode(
        game.world,
        agent,
        episode,
        progress_label=progress_label,
        heartbeat_interval=HEARTBEAT_INTERVAL,
        on_tick=on_tick,
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
                f"Could not load existing weights from {weights_path}; "
                f"training was not restarted: {error}",
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


def _print_batch_summary(
    challenge: Challenge,
    batch_number: int,
    results: list,
    successes: int,
    trials: int,
    survive: int,
) -> bool:
    count = len(results)
    avg_ticks = sum(result.ticks for result in results) / count if count else 0.0
    avg_reward = sum(result.total_reward for result in results) / count if count else 0.0
    bombs = sum(result.bombs_placed for result in results)
    passed = successes >= survive
    print(f"\nStage {challenge.number} / {challenge.name}", flush=True)
    print(f"Batch {batch_number} complete", flush=True)
    print(f"Survived: {successes} / {trials}", flush=True)
    print(f"Required: {survive} / {trials}", flush=True)
    print(f"Average ticks: {avg_ticks:.1f}", flush=True)
    print(f"Average reward: {avg_reward:.1f}", flush=True)
    print(f"Bombs: {bombs}", flush=True)
    if passed:
        print("PASSED - advancing to the next stage", flush=True)
    else:
        print("NOT PASSED - continuing at this stage", flush=True)
    return passed


def run_progressive_training(
    trials: int = DEFAULT_TRIALS,
    survive: int = DEFAULT_SURVIVE,
    seed: int = DEFAULT_SEED,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    display: bool = DEFAULT_DISPLAY,
    fresh: bool = False,
    max_batches: Optional[int] = None,
    q_contributions_diagnostic: bool = False,
    *,
    agent_factory: Optional[Callable] = None,
    game_factory: Optional[Callable] = None,
    episode_runner: Optional[Callable] = None,
) -> TrainingSummary:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 1 <= survive <= trials:
        raise ValueError("survive must be between 1 and trials")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1")

    _configure_runtime(display)
    agent_factory = agent_factory or _create_qagent
    game_factory = game_factory or _build_variant_game
    episode_runner = episode_runner or _run_episode
    weights_path = Path(weights_path)
    challenges = progression()
    weights, save_agent, can_save = _load_initial_weights(
        weights_path,
        fresh,
        agent_factory,
    )
    last_valid_weights = dict(weights)
    summary = TrainingSummary()
    seed_rng = random.Random(seed)
    active_agent = None

    try:
        print("Project 2 Progressive Q-Learning Training", flush=True)
        print(f"Map: {DEFAULT_MAP_PATH}", flush=True)
        print(f"Trials per batch: {trials}", flush=True)
        print(f"Successes required: {survive}", flush=True)
        print(f"Base seed: {seed}", flush=True)
        print(f"Display: {'Enabled' if display else 'Disabled'}", flush=True)
        print(
            f"Q contribution diagnostic: "
            f"{'Enabled' if q_contributions_diagnostic else 'Disabled'}",
            flush=True,
        )

        for challenge in challenges:
            summary.current_stage = challenge
            batch_number = 0
            stage_passed = False
            while not stage_passed:
                if max_batches is not None and summary.completed_batches >= max_batches:
                    summary.stopped_reason = "configured batch limit reached"
                    break

                batch_number += 1
                summary.current_batch = batch_number
                batch_results = []
                successes = 0
                print(
                    f"\n{'=' * 54}\nStage {challenge.number}: {challenge.name}\n"
                    f"Batch {batch_number} | {trials} trials | need {survive} successes",
                    flush=True,
                )

                for trial_number in range(1, trials + 1):
                    trial_seed = seed_rng.randint(0, 2**32 - 1)
                    label = (
                        f"Stage {challenge.number} | Batch {batch_number} | "
                        f"Trial {trial_number}/{trials}"
                    )
                    game, active_agent = game_factory(
                        challenge,
                        trial_seed,
                        weights,
                        agent_factory,
                    )
                    active_agent.q_contributions_diagnostic = q_contributions_diagnostic
                    result = episode_runner(
                        game,
                        active_agent,
                        summary.completed_trials + 1,
                        label,
                        display,
                    )
                    summary.completed_trials += 1
                    batch_results.append(result)
                    if result.outcome == "WON":
                        successes += 1

                    stable = _result_is_numerically_stable(result, active_agent.weights)
                    if stable:
                        weights = active_agent.weights
                        _save_weights(active_agent, weights, weights_path)
                        save_agent = active_agent
                        last_valid_weights = dict(weights)
                        can_save = True

                    print(
                        f"{'=' * 54}\n"
                        f"Stage {challenge.number}: {challenge.name}\n"
                        f"Batch {batch_number} | Trial {trial_number}/{trials}\n"
                        f"Outcome: {result.outcome}\n"
                        f"Ticks: {result.ticks}\n"
                        f"Reward: {result.total_reward:.0f}\n"
                        f"Bombs: {result.bombs_placed}\n"
                        f"Batch survival: {successes}/{trial_number}\n"
                        f"Required to advance: {survive}/{trials}\n"
                        f"Weights saved: {'yes' if stable else 'no (unstable values)'}\n"
                        f"{'=' * 54}",
                        flush=True,
                    )

                    if not stable:
                        weights.clear()
                        weights.update(last_valid_weights)
                        summary.stopped_reason = (
                            f"numerical instability in stage {challenge.number}, "
                            f"batch {batch_number}, trial {trial_number}"
                        )
                        print(
                            f"STOP: {summary.stopped_reason}; non-finite Q values="
                            f"{result.non_finite_q_values}, fallbacks="
                            f"{result.non_finite_q_fallbacks}, all weights finite="
                            f"{_weights_are_finite(active_agent.weights)}",
                            flush=True,
                        )
                        active_agent = None
                        break

                if summary.stopped_reason is not None:
                    break

                summary.completed_batches += 1
                stage_passed = _print_batch_summary(
                    challenge,
                    batch_number,
                    batch_results,
                    successes,
                    trials,
                    survive,
                )

            if summary.stopped_reason is not None:
                break
            if stage_passed:
                summary.stages_passed += 1
                if challenge.number == challenges[-1].number:
                    summary.completed = True
                    break

    except KeyboardInterrupt:
        summary.interrupted = True
    except TrainingAborted as error:
        summary.stopped_reason = str(error)
        print(f"Training stopped: {error}", flush=True)
    finally:
        if active_agent is not None:
            active_weights = active_agent.weights
            active_updates = getattr(active_agent, "td_update_count", 0)
            if _weights_are_finite(active_weights) and (
                can_save or active_updates > 0
            ):
                weights = active_weights
                can_save = True

        if can_save:
            if not _weights_are_finite(weights):
                weights.clear()
                weights.update(last_valid_weights)
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
        print(f"Completed trials: {summary.completed_trials}", flush=True)
        if summary.current_stage is not None:
            print(
                f"Current stage: Variant {summary.current_stage.number} "
                f"({summary.current_stage.name})",
                flush=True,
            )
        print(f"Current weights saved to: {weights_path if summary.weights_saved else 'not saved'}", flush=True)
    elif summary.completed:
        print("\nPROGRESSIVE TRAINING COMPLETE", flush=True)
        print("All existing challenge levels passed.", flush=True)
        print(f"Final stage: Variant {challenges[-1].number}", flush=True)
        print(f"Total training episodes: {summary.completed_trials}", flush=True)
        print(f"Final weights saved to: {weights_path}", flush=True)
    elif summary.stopped_reason is not None:
        print(f"Training stopped: {summary.stopped_reason}", flush=True)
        print(f"Completed trials: {summary.completed_trials}", flush=True)
        print(f"Weights: {weights_path if summary.weights_saved else 'not saved'}", flush=True)

    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--survive", type=int, default=DEFAULT_SURVIVE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing weights and start from zero.")
    parser.add_argument(
        "--q-contributions",
        action="store_true",
        help="Explain candidate Q-values when PLACE_BOMB is selected or greedily preferred.",
    )
    parser.add_argument("--max-batches", type=int, help="Stop after this many complete batches (useful for short runs).")
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="display", action="store_true")
    display_group.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=DEFAULT_DISPLAY)
    return parser.parse_args(argv)


def main(argv=None) -> TrainingSummary:
    args = parse_args(argv)
    return run_progressive_training(
        trials=args.trials,
        survive=args.survive,
        seed=args.seed,
        weights_path=args.weights,
        display=args.display,
        fresh=args.fresh,
        max_batches=args.max_batches,
        q_contributions_diagnostic=args.q_contributions,
    )


if __name__ == "__main__":
    main()