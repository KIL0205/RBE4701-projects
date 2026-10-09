"""Shared standalone runner for the Project 2 variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from team01.project2.agent_backends import (
    AGENT_CHOICES,
    AGENT_LABELS,
    DEFAULT_AGENT,
    DEFAULT_AGENT_STATE_PATHS,
)

_ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = Path(__file__).resolve().with_name("map.txt")
SPRITE_DIRECTORY = str(_ROOT / "Bomberman" / "sprites") + os.sep


@dataclass(frozen=True)
class VariantMonster:
    kind: str
    name: str
    avatar: str
    x: int
    y: int
    detection_range: Optional[int] = None


@dataclass(frozen=True)
class VariantDefinition:
    number: int
    name: str
    monsters: tuple[VariantMonster, ...]


VARIANTS = {
    1: VariantDefinition(1, "Alone in the world", ()),
    2: VariantDefinition(
        2,
        "Stupid Monster",
        (VariantMonster("stupid", "stupid", "S", 3, 9),),
    ),
    3: VariantDefinition(
        3,
        "Aggressive Monster",
        (VariantMonster("smart", "aggressive", "A", 3, 13, 2),),
    ),
    4: VariantDefinition(
        4,
        "Self-preserving Monster",
        (VariantMonster("smart", "selfpreserving", "S", 3, 9, 1),),
    ),
    5: VariantDefinition(
        5,
        "Stupid and Aggressive Monsters",
        (
            VariantMonster("stupid", "stupid", "S", 3, 5),
            VariantMonster("smart", "aggressive", "A", 3, 13, 2),
        ),
    ),
}


def resolve_seed(seed: Optional[int]) -> int:
    """Use Project 1's random 0..1000 default-seed convention when omitted."""
    return random.randint(0, 1000) if seed is None else seed


def parse_variant_args(
    variant_number: int,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    try:
        definition = VARIANTS[variant_number]
    except KeyError as error:
        raise ValueError("Project 2 supports variants 1 through 5") from error

    parser = argparse.ArgumentParser(
        description=f"Run Project 2 V{variant_number}: {definition.name}."
    )
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default=DEFAULT_AGENT,
        help="Agent backend (default: qbt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed; omitted seeds are generated in the range 0..1000.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Learned-state file for the selected agent.",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--display", dest="display", action="store_true")
    display.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser.parse_args(argv)


def _configure_runtime(display: bool) -> None:
    os.environ["BOMBERMAN_DISPLAY"] = "1" if display else "0"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if display:
        os.environ.pop("SDL_VIDEODRIVER", None)
    else:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


def _new_game(variant_number: int):
    bomberman_path = str(_ROOT / "Bomberman")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    if bomberman_path not in sys.path:
        sys.path.insert(0, bomberman_path)

    from Bomberman.game import Game
    from Bomberman.monsters.selfpreserving_monster import SelfPreservingMonster
    from Bomberman.monsters.stupid_monster import StupidMonster

    definition = VARIANTS[variant_number]
    game = Game.fromfile(str(MAP_PATH), sprite_dir=SPRITE_DIRECTORY)
    for monster in definition.monsters:
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
    return game


def _load_backend(agent_type: str, weights_path: Path):
    from team01.eval.project2_grading_eval import _load_grading_agent

    return _load_grading_agent(agent_type, weights_path)


def _assert_frozen_agent(agent_type: str, agent: Any) -> None:
    if agent.training or agent.epsilon != 0.0:
        raise RuntimeError("Standalone variant agents must use frozen inference")

    if agent_type in {"q", "qbt"}:
        from team01.agent.q_learning import QAgent, QBTAgent

        expected_type = QBTAgent if agent_type == "qbt" else QAgent
        if type(agent) is not expected_type:
            raise TypeError(
                f"{agent_type} backend constructed {type(agent).__name__}, "
                f"expected {expected_type.__name__}"
            )
        return

    from team01.agent.deep_q_learning import DeepQAgent

    if not isinstance(agent, DeepQAgent):
        raise TypeError(f"DQN backend constructed {type(agent).__name__}")
    if (
        agent.learning_enabled
        or agent.collect_experience
        or agent.policy_network.training
        or len(agent.replay_buffer) != 0
    ):
        raise RuntimeError("Standalone DQN must use frozen inference without replay")


def _assert_standalone_state_unchanged(
    variant_number: int,
    agent_type: str,
    backend: Any,
    agent: Any,
    checkpoint_bytes: bytes,
    agent_snapshot: Any,
    weights_path: Path,
) -> None:
    if weights_path.read_bytes() != checkpoint_bytes:
        raise RuntimeError(
            f"Standalone Project 2 V{variant_number} run modified learned state: "
            f"{weights_path}"
        )

    if agent_type in {"q", "qbt"}:
        if agent.weights != agent_snapshot["weights"]:
            raise RuntimeError("Frozen Q policy changed during standalone execution")
        if agent.td_update_count != agent_snapshot["td_update_count"]:
            raise RuntimeError("Frozen Q policy performed a TD update")
        return

    from team01.agent.deep_q_learning import checkpoint_snapshots_equal

    if (
        agent.learning_enabled
        or agent.collect_experience
        or agent.policy_network.training
        or len(agent.replay_buffer) != 0
        or not checkpoint_snapshots_equal(
            agent_snapshot,
            agent.get_checkpoint_snapshot(),
        )
    ):
        raise RuntimeError("Frozen DQN state changed during standalone execution")
    if backend.canonical_agent is None or backend.canonical_snapshot is None:
        raise RuntimeError("DQN backend did not retain its canonical snapshot")
    if not checkpoint_snapshots_equal(
        backend.canonical_snapshot,
        backend.canonical_agent.get_checkpoint_snapshot(),
    ):
        raise RuntimeError("Standalone DQN run modified its canonical policy")


def run_variant(
    variant_number: int,
    *,
    agent_type: str = DEFAULT_AGENT,
    seed: Optional[int] = None,
    weights_path: Optional[Path | str] = None,
    display: bool = True,
    game_factory: Optional[Callable[[int], Any]] = None,
    backend_loader: Optional[Callable[[str, Path], Any]] = None,
) -> dict[str, Any]:
    """Run one interactive, frozen-policy game for a Project 2 variant."""
    if variant_number not in VARIANTS:
        raise ValueError("Project 2 supports variants 1 through 5")
    if agent_type not in AGENT_CHOICES:
        raise ValueError(f"unknown Project 2 agent type: {agent_type}")

    resolved_seed = resolve_seed(seed)
    resolved_weights = Path(
        weights_path
        if weights_path is not None
        else DEFAULT_AGENT_STATE_PATHS[agent_type]
    )
    if not resolved_weights.is_file():
        raise FileNotFoundError(
            f"{AGENT_LABELS[agent_type]} learned state not found: "
            f"{resolved_weights}. Train the agent first or provide --weights."
        )
    checkpoint_bytes = resolved_weights.read_bytes()

    _configure_runtime(display)
    # Resolve and apply the environment seed before agent construction so agent
    # choice or initialization cannot affect the Project 1-compatible scenario.
    random.seed(resolved_seed)
    game = (game_factory or _new_game)(variant_number)
    backend = (backend_loader or _load_backend)(agent_type, resolved_weights)
    agent = backend.agent_factory()
    _assert_frozen_agent(agent_type, agent)
    game.add_character(agent)

    definition = VARIANTS[variant_number]
    print(f"Variant: V{variant_number} - {definition.name}")
    print(f"Agent: {AGENT_LABELS[agent_type]}")
    print(f"Seed: {resolved_seed}")

    if agent_type in {"q", "qbt"}:
        agent_snapshot = {
            "weights": dict(agent.weights),
            "td_update_count": agent.td_update_count,
        }
    else:
        agent_snapshot = agent.get_checkpoint_snapshot()

    try:
        # Reset after construction as well: backend initialization must not
        # shift the random sequence used by the seeded game world.
        random.seed(resolved_seed)
        game.go(1)
    finally:
        if agent_type == "dqn":
            agent.episodes = agent_snapshot["episodes"]
        _assert_standalone_state_unchanged(
            variant_number,
            agent_type,
            backend,
            agent,
            checkpoint_bytes,
            agent_snapshot,
            resolved_weights,
        )

    return {
        "variant": variant_number,
        "agent_type": agent_type,
        "seed": resolved_seed,
        "weights_path": resolved_weights,
    }


def main_for_variant(
    variant_number: int,
    argv: Optional[list[str]] = None,
) -> dict[str, Any]:
    args = parse_variant_args(variant_number, argv)
    return run_variant(
        variant_number,
        agent_type=args.agent,
        seed=args.seed,
        weights_path=args.weights,
        display=args.display,
    )
