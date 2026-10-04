"""Estimate Project 2 grading performance with fresh randomized trials."""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
else:
    _ROOT = Path(__file__).resolve().parents[2]

_BOMBERMAN = _ROOT / "Bomberman"
if str(_BOMBERMAN) not in sys.path:
    sys.path.insert(0, str(_BOMBERMAN))

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
DEFAULT_WEIGHTS_PATH = _ROOT / "team01" / "project2" / "q_learning_weights.json"
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


class GradingAborted(Exception):
    """Raised when the visible grading window is closed."""


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
    weights: dict[str, float],
) -> tuple[Game, QAgent]:
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

    agent = QAgent("me", "C", 0, 0)
    agent.weights = dict(weights)
    agent.set_learning(no_training=True, epsilon=0.0)
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
    weights: dict[str, float],
    display: bool = False,
    game_builder: Optional[Callable[[int, int, dict[str, float]], tuple[Any, QAgent]]] = None,
    episode_runner: Optional[Callable[..., EpisodeResult]] = None,
) -> GradingTrialResult:
    """Run one fresh-seed Project 2 game with a clean frozen agent."""
    if weights is None:
        raise ValueError("run_trial requires loaded learned weights")
    canonical_weights = dict(weights)
    builder = game_builder or _build_trial_game
    runner = episode_runner or run_episode
    game, agent = builder(variant, seed, canonical_weights)
    if agent.training or agent.epsilon != 0.0:
        raise RuntimeError("Project 2 grading trial did not start with a frozen policy")
    before_weights = dict(agent.weights)
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

    if agent.td_update_count != 0 or episode.td_update_count != 0:
        raise RuntimeError("Project 2 grading trial unexpectedly performed TD updates")
    if agent.weights != before_weights or weights != canonical_weights:
        raise RuntimeError("Project 2 grading trial modified learned weights")
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
    weights: dict[str, float],
    display: bool,
    trial_runner: Callable[..., GradingTrialResult] = run_trial,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for run_number in range(1, runs + 1):
        seed = generate_seed(used_seeds)
        result = trial_runner(
            variant,
            seed=seed,
            weights=weights,
            display=display,
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
    weights: dict[str, float],
    *,
    runs: int = RUNS_PER_VARIANT,
    display: bool = DISPLAY,
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
    output_path: Path | str = RESULT_PATH,
    trial_runner: Callable[..., GradingTrialResult] = run_trial,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs per variant must be at least 1")
    canonical_weights = dict(weights)
    used_seeds: set[int] = set()
    variants: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_POINTS:
        print(f"\n=== Variant {variant} ===")
        summary = evaluate_variant(
            variant,
            runs,
            used_seeds,
            canonical_weights,
            display,
            trial_runner=trial_runner,
        )
        variants[str(variant)] = summary
        print_variant_summary(variant, summary)

    total_points = sum(summary["points_earned"] for summary in variants.values())
    maximum_points = sum(VARIANT_POINTS.values())
    total_wins = sum(summary["wins"] for summary in variants.values())
    solved_count = sum(summary["solved"] for summary in variants.values())
    result = {
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
    if weights != canonical_weights:
        raise RuntimeError("Project 2 grading evaluation changed canonical weights")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS_PER_VARIANT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
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
    if (args.variant is None) != (args.seed is None):
        parser.error("single-trial replay requires both --variant and --seed")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    return args


def main(argv: Optional[list[str]] = None) -> dict[str, Any] | GradingTrialResult:
    args = parse_args(argv)
    weights_path = Path(args.weights)
    print("PROJECT 2 GRADING EVALUATION")
    print(f"\nWeights: {weights_path.resolve()}")
    print("Mode: Frozen learned policy")
    print("Training: False")
    print("Epsilon: 0.0")
    print("Learning: DISABLED")
    print("Exploration: DISABLED")

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Project 2 learned weights not found: {weights_path}. Train or provide "
            "--weights; grading evaluation will not fall back to untrained weights."
        )
    checkpoint_before = weights_path.read_bytes()
    weights = load_frozen_weights(weights_path)
    configure_runtime(args.display)

    if args.variant is not None:
        result = run_trial(
            args.variant,
            seed=args.seed,
            weights=weights,
            display=args.display,
        )
        print(
            f"Replay result | V{result.variant} | seed={result.seed} | "
            f"{result.reason} | ticks={result.ticks} | success={result.success}"
        )
    else:
        result = run_evaluation(
            weights,
            runs=args.runs,
            display=args.display,
            weights_path=weights_path,
            output_path=args.output,
        )
        print_final_summary(result, output_path=args.output)

    if weights_path.read_bytes() != checkpoint_before:
        raise RuntimeError("Project 2 grading evaluation modified the learned checkpoint")
    return result


if __name__ == "__main__":
    main()
