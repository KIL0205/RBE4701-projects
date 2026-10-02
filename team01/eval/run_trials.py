"""Headless evaluation of the real Bomberman simulation."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

if os.environ.get("BOMBERMAN_DISPLAY") != "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_ROOT = Path(__file__).resolve().parents[2]
_BOMBERMAN = _ROOT / "Bomberman"
_PROJECT1 = _ROOT / "team01" / "project1"
if str(_BOMBERMAN) not in sys.path:
    sys.path.insert(0, str(_BOMBERMAN))
if str(_PROJECT1.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT1.parent))

from events import Event
from game import Game
import pygame
from monsters.selfpreserving_monster import SelfPreservingMonster
from monsters.stupid_monster import StupidMonster
from team01.agent.controller import BombermanAgent
from team01.agent.safety import monster_immediate_reachable_cells

SMART_MONSTER_RANGE = 2
MIN_MONSTER_START_DISTANCE = 3


@dataclass
class TickTrace:
    """Compact per-tick controller observation used by replay and debugging."""
    tick: int
    character_position: Optional[tuple[int, int]]
    monster_positions: list[tuple[int, int]]
    threat_cells: set[tuple[int, int]] = field(default_factory=set)
    safe_actions: list[Any] = field(default_factory=list)
    selected_action: Any = None
    active_behavior: Optional[str] = None
    selected_reason: Optional[str] = None
    planned_path: list[tuple[int, int]] = field(default_factory=list)
    events: list[int] = field(default_factory=list)
    bomb_positions: list[tuple[int, int]] = field(default_factory=list)
    bomb_timers: dict[str, int] = field(default_factory=dict)
    explosion_positions: list[tuple[int, int]] = field(default_factory=list)
    evaluation_breakdowns: list[dict[str, Any]] = field(default_factory=list)
    candidate_safety: list[dict[str, Any]] = field(default_factory=list)
    monster_immediate_reachable_cells: set[tuple[int, int]] = field(default_factory=set)
    monster_t2_reachable_cells: set[tuple[int, int]] = field(default_factory=set)
    current_cell_threatened_before_move: bool = False


@dataclass
class TrialResult:
    """Outcome of one headless Project 1 or custom trial."""
    variant: int
    success: bool
    reason: str
    ticks: int
    seed: Optional[int]
    trace: list[TickTrace] = field(default_factory=list)
    map_name: str = "map.txt"
    monster_spawns: list["MonsterSpawn"] = field(default_factory=list)


@dataclass(frozen=True)
class MonsterSpawn:
    monster_type: str
    x: int
    y: int


class EvaluationAborted(Exception):
    """Raised when the user closes the visible evaluation window."""


def _characters(world):
    return [character for characters in world.characters.values() for character in characters]


def _monsters(world):
    return [monster for monsters in world.monsters.values() for monster in monsters]


def _bombs(world):
    return list(world.bombs.values())


def _explosions(world):
    return list(world.explosions.values())


def _map_game(map_path: Path) -> Game:
    sprite_dir = str(_BOMBERMAN / "sprites") + os.sep
    return Game.fromfile(str(map_path), sprite_dir=sprite_dir)


def validate_map(map_path: Path) -> None:
    game = _map_game(map_path)
    world = game.world
    if world.exitcell is None:
        raise ValueError(f"map has no exit: {map_path}")

    start = (0, 0)
    exit_x, exit_y = world.exitcell
    if world.wall_at(*start) or not (0 <= exit_x < world.width() and 0 <= exit_y < world.height()):
        raise ValueError(f"invalid player start or exit: {map_path}")

    reachable = {start}
    frontier = [start]
    while frontier:
        x, y = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                target = (x + dx, y + dy)
                tx, ty = target
                if not (0 <= tx < world.width() and 0 <= ty < world.height()):
                    continue
                if world.wall_at(tx, ty) or target in reachable:
                    continue
                reachable.add(target)
                frontier.append(target)
    if world.exitcell not in reachable:
        raise ValueError(f"exit is unreachable from player start: {map_path}")


def _spawn_positions(game: Game, count: int, seed: Optional[int]) -> list[tuple[int, int]]:
    world = game.world
    start = (0, 0)
    exit_position = world.exitcell
    candidates = []
    for x in range(world.width()):
        for y in range(world.height()):
            position = (x, y)
            if world.wall_at(x, y) or position == start or position == exit_position:
                continue
            if max(abs(x - start[0]), abs(y - start[1])) < MIN_MONSTER_START_DISTANCE:
                continue
            candidates.append(position)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"map has only {len(candidates)} valid spawn cells for {count} monsters")
    return candidates[:count]


def _build_game(
    variant_number: int,
    seed: Optional[int],
    weights: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[Game, BombermanAgent]:
    if seed is not None:
        random.seed(seed)

    map_path = _PROJECT1 / "map.txt"
    game = _map_game(map_path)

    if variant_number == 2:
        game.add_monster(StupidMonster("stupid", "S", 3, 9))
    elif variant_number == 3:
        game.add_monster(SelfPreservingMonster("selfpreserving", "S", 3, 9, 1))
    elif variant_number == 4:
        game.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, SMART_MONSTER_RANGE))
    elif variant_number == 5:
        game.add_monster(StupidMonster("stupid", "S", 3, 5))
        game.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, 1))
    elif variant_number != 1:
        raise ValueError("headless evaluation supports variants 1 through 5")

    if weights is None:
        agent = BombermanAgent("me", "C", 0, 0)
    else:
        agent = BombermanAgent("me", "C", 0, 0, weights=weights)
    game.add_character(agent)
    return game, agent


def build_custom_game(
    map_path: Path,
    dumb_monsters: int,
    smart_monsters: int,
    seed: Optional[int],
) -> tuple[Game, BombermanAgent, list[MonsterSpawn]]:
    if seed is not None:
        random.seed(seed)
    validate_map(map_path)
    game = _map_game(map_path)
    positions = _spawn_positions(game, dumb_monsters + smart_monsters, seed)
    spawns: list[MonsterSpawn] = []
    for index, (x, y) in enumerate(positions[:dumb_monsters]):
        game.add_monster(StupidMonster(f"dumb-{index}", "S", x, y))
        spawns.append(MonsterSpawn("DUMB", x, y))
    for index, (x, y) in enumerate(positions[dumb_monsters:]):
        game.add_monster(SelfPreservingMonster(f"smart-{index}", "A", x, y, SMART_MONSTER_RANGE))
        spawns.append(MonsterSpawn("SMART", x, y))
    agent = BombermanAgent("me", "C", 0, 0)
    game.add_character(agent)
    return game, agent, spawns


def _trace_tick(world, agent, tick: int) -> TickTrace:
    """Copy the current world and blackboard diagnostics into a trace frame."""
    position = (agent.x, agent.y) if agent in _characters(world) else None
    monsters = sorted((monster.x, monster.y) for monster in _monsters(world))
    board = getattr(agent, "blackboard", None)
    threat = set(board.get("monster_threat_cells") or set()) if board else set()
    safe = list(board.get("safe_actions") or []) if board else []
    selected = board.get("selected_action") if board else None
    planned_path = list(board.get("planned_path") or []) if board else []
    debug = board.get("debug_info") or {} if board else {}
    breakdowns = board.get("evaluation_breakdowns") or [] if board else []
    candidate_safety = board.get("candidate_safety") or [] if board else []
    model = board.get("world_model") if board else None
    reachable = monster_immediate_reachable_cells(model) if model else set()
    future_reachable = set(board.get("monster_t2_threat_cells") or set()) if board else set()
    current_threatened = bool(model and model.self_position in reachable)
    return TickTrace(
        tick=tick,
        character_position=position,
        monster_positions=monsters,
        threat_cells=threat,
        safe_actions=safe,
        selected_action=selected,
        active_behavior=debug.get("active_behavior"),
        selected_reason=debug.get("selected_reason"),
        planned_path=planned_path,
        events=[event.tpe for event in world.events],
        bomb_positions=sorted((bomb.x, bomb.y) for bomb in _bombs(world)),
        bomb_timers={f"{bomb.x},{bomb.y}": bomb.timer for bomb in _bombs(world)},
        explosion_positions=sorted((explosion.x, explosion.y) for explosion in _explosions(world)),
        evaluation_breakdowns=[asdict(breakdown) for breakdown in breakdowns],
        candidate_safety=candidate_safety,
        monster_immediate_reachable_cells=reachable,
        monster_t2_reachable_cells=future_reachable,
        current_cell_threatened_before_move=current_threatened,
    )


def run_trial(
    variant_number: int,
    seed: Optional[int] = None,
    trace: bool = False,
    display: bool = False,
    weights: Optional[dict[str, dict[str, float]]] = None,
) -> TrialResult:
    """Run one Project 1 variant, optionally rendering it with Pygame."""
    game, agent = _build_game(variant_number, seed, weights)
    world = game.world
    traces: list[TickTrace] = []
    ticks = 0

    if display:
        pygame.display.set_caption(f"Bomberman Project 1 | Variant {variant_number}")
        game.display_gui()

    while world.time > 0 and _characters(world):
        world.next()
        ticks += 1

        event_types = {event.tpe for event in world.events}
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise EvaluationAborted()
            game.display_gui()

        if trace:
            # Decisions are made after the world update, matching Game.go().
            world.next_decisions()
            traces.append(_trace_tick(world, agent, ticks))
        else:
            world.next_decisions()

        if display:
            pygame.time.wait(50)

        if Event.CHARACTER_FOUND_EXIT in event_types:
            return TrialResult(variant_number, True, "EXIT", ticks, seed, traces)
        if Event.CHARACTER_KILLED_BY_MONSTER in event_types:
            return TrialResult(variant_number, False, "MONSTER_DEATH", ticks, seed, traces)
        if Event.BOMB_HIT_CHARACTER in event_types:
            return TrialResult(variant_number, False, "EXPLOSION_DEATH", ticks, seed, traces)

    if world.time <= 0:
        reason = "TIMEOUT"
    elif not _characters(world):
        reason = "NO_CHARACTER"
    else:
        reason = "UNKNOWN"
    return TrialResult(variant_number, False, reason, ticks, seed, traces)


def run_custom_trial(
    map_path: Path,
    dumb_monsters: int,
    smart_monsters: int,
    seed: Optional[int],
    trace: bool = False,
    display: bool = False,
) -> TrialResult:
    game, agent, spawns = build_custom_game(map_path, dumb_monsters, smart_monsters, seed)
    world = game.world
    traces: list[TickTrace] = []
    ticks = 0
    if display:
        pygame.display.set_caption(f"Bomberman Stress Test | {map_path.stem}")
        game.display_gui()

    while world.time > 0 and _characters(world):
        world.next()
        ticks += 1
        event_types = {event.tpe for event in world.events}
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise EvaluationAborted()
            game.display_gui()
        world.next_decisions()
        if trace:
            traces.append(_trace_tick(world, agent, ticks))
        if display:
            pygame.time.wait(50)
        if Event.CHARACTER_FOUND_EXIT in event_types:
            reason, success = "EXIT", True
        elif Event.CHARACTER_KILLED_BY_MONSTER in event_types:
            reason, success = "MONSTER_DEATH", False
        elif Event.BOMB_HIT_CHARACTER in event_types:
            reason, success = "EXPLOSION_DEATH", False
        else:
            reason = None
            success = False
        if reason is not None:
            return TrialResult(0, success, reason, ticks, seed, traces, map_path.name, spawns)

    reason = "TIMEOUT" if world.time <= 0 else "NO_CHARACTER" if not _characters(world) else "UNKNOWN"
    return TrialResult(0, False, reason, ticks, seed, traces, map_path.name, spawns)


def trial_result_dict(result: TrialResult) -> dict[str, Any]:
    """Convert a trial result into JSON-compatible data."""
    data = asdict(result)
    for trace in data["trace"]:
        trace["threat_cells"] = [list(cell) for cell in trace["threat_cells"]]
        trace["monster_immediate_reachable_cells"] = [
            list(cell) for cell in trace["monster_immediate_reachable_cells"]
        ]
        trace["monster_t2_reachable_cells"] = [
            list(cell) for cell in trace["monster_t2_reachable_cells"]
        ]
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("variant", type=int, nargs="?", default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    print(run_trial(args.variant, args.seed, args.trace))
