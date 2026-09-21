from dataclasses import dataclass
from typing import Iterable, Set

from .actions import AgentAction
from .navigation import find_path, measure_mobility
from .safety import monster_reachable_layers
from .world_model import Position, WorldModel


@dataclass(frozen=True)
class EvaluationProfile:
    """Weights used by one behavior when scoring an action."""
    exit_progress_weight: float
    mobility_weight: float
    escape_options_weight: float
    monster_threat_weight: float
    bomb_threat_weight: float
    explosion_threat_weight: float
    trap_risk_weight: float
    future_monster_risk_weight: float
    future_escape_options_weight: float
    future_trap_risk_weight: float
    wait_penalty: float


NORMAL_NAVIGATION = EvaluationProfile(
    exit_progress_weight=12.0,
    mobility_weight=2.0,
    escape_options_weight=3.0,
    monster_threat_weight=30.0,
    bomb_threat_weight=24.0,
    explosion_threat_weight=100.0,
    trap_risk_weight=18.0,
    future_monster_risk_weight=12.0,
    future_escape_options_weight=2.0,
    future_trap_risk_weight=16.0,
    wait_penalty=20.0,
)

EMERGENCY_ESCAPE = EvaluationProfile(
    exit_progress_weight=1.0,
    mobility_weight=8.0,
    escape_options_weight=10.0,
    monster_threat_weight=45.0,
    bomb_threat_weight=35.0,
    explosion_threat_weight=150.0,
    trap_risk_weight=35.0,
    future_monster_risk_weight=25.0,
    future_escape_options_weight=5.0,
    future_trap_risk_weight=30.0,
    wait_penalty=8.0,
)

FALLBACK = EvaluationProfile(
    exit_progress_weight=2.0,
    mobility_weight=6.0,
    escape_options_weight=8.0,
    monster_threat_weight=40.0,
    bomb_threat_weight=30.0,
    explosion_threat_weight=120.0,
    trap_risk_weight=30.0,
    future_monster_risk_weight=20.0,
    future_escape_options_weight=4.0,
    future_trap_risk_weight=25.0,
    wait_penalty=6.0,
)


@dataclass(frozen=True)
class EvaluationBreakdown:
    """Feature values and final score for one candidate action."""
    action: AgentAction
    destination: Position
    exit_progress: float
    mobility: float
    escape_options: float
    monster_threat: float
    bomb_threat: float
    explosion_threat: float
    trap_risk: float
    future_monster_risk: float
    future_escape_options: float
    future_trap_risk: float
    wait_penalty: float
    lethal: bool
    total: float


def monster_immediate_cells(model: WorldModel) -> Set[Position]:
    """Return cells a monster could occupy during the next update."""
    cells: Set[Position] = set()
    for monster in model.monster_positions():
        cells.add(monster)
        x, y = monster
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                position = (x + dx, y + dy)
                if model.in_bounds(position) and not model.is_wall(position):
                    cells.add(position)
    return cells


def bomb_blast_cells(model: WorldModel, bomb: Position) -> Set[Position]:
    """Return the framework's cardinal blast cells for a bomb."""
    cells = {bomb}
    x, y = bomb
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, max(0, model.explosion_range) + 1):
            position = (x + dx * distance, y + dy * distance)
            if not model.in_bounds(position) or model.is_wall(position):
                break
            cells.add(position)
            if position in model.bombs and position != bomb:
                break
    return cells


def immediate_lethal_positions(model: WorldModel) -> Set[Position]:
    """Return positions treated as immediately lethal by the evaluator."""
    lethal = set(model.explosions)
    lethal.update(monster_immediate_cells(model))
    for bomb, timer in model.bomb_timers.items():
        if timer <= 1:
            lethal.update(bomb_blast_cells(model, bomb))
    return lethal


def measure_exit_progress(model: WorldModel, position: Position) -> float:
    """Return old A* distance minus new A* distance to the exit."""
    if model.exit_position is None or model.self_position is None:
        return 0.0
    old_path = find_path(model, model.self_position, model.exit_position)
    new_path = find_path(model, position, model.exit_position)
    if not new_path:
        return -float(max(1, model.width * model.height))
    old_distance = len(old_path) - 1 if old_path else 0
    new_distance = len(new_path) - 1
    return float(old_distance - new_distance)


def measure_threat(model: WorldModel, position: Position) -> float:
    """Measure current monster, bomb, and explosion threat."""
    threat = 0.0
    px, py = position

    for monster in model.monster_positions():
        distance = max(abs(monster[0] - px), abs(monster[1] - py))
        if distance == 0:
            threat += 100.0
        elif position in monster_immediate_cells(model):
            threat += 60.0
        else:
            threat += 12.0 / distance

    for explosion in model.explosions:
        if position == explosion:
            threat += 100.0

    for bomb, timer in model.bomb_timers.items():
        blast = bomb_blast_cells(model, bomb)
        if position in blast:
            if timer <= 1:
                threat += 80.0
            else:
                threat += 20.0 / max(1, timer)
        elif position == bomb:
            threat += 30.0

    return threat


def measure_bomb_threat(model: WorldModel, position: Position) -> float:
    score = 0.0
    for bomb, timer in model.bomb_timers.items():
        if position in bomb_blast_cells(model, bomb):
            score += 50.0 / max(1, timer)
        elif position == bomb:
            score += 40.0
    return score


def measure_explosion_threat(model: WorldModel, position: Position) -> float:
    return 100.0 if position in model.explosions else 0.0


def measure_escape_options(model: WorldModel, position: Position, lethal: Set[Position]) -> int:
    """Count neighboring cells that are not lethal or occupied by bombs."""
    escape_options = 0

    for neighbor in model.neighbors(position):
        if neighbor not in lethal and neighbor not in model.bombs:
            escape_options += 1

    return escape_options


def measure_trap_risk(model: WorldModel, position: Position, lethal: Set[Position]) -> float:
    mobility = measure_mobility(model, position, num_steps=2, forbidden_cells=lethal)
    escape_options = measure_escape_options(model, position, lethal)
    if escape_options == 0:
        return 100.0
    if mobility <= 1:
        return 30.0
    if mobility <= 3:
        return 12.0
    return 0.0


def measure_future_monster_risk(model: WorldModel, position: Position) -> float:
    """Return a penalty flag when a position is reachable at t+2."""
    _, second_step = monster_reachable_layers(model)
    if position in second_step:
        return 1.0
    return 0.0


def measure_future_escape_options(model: WorldModel, position: Position) -> int:
    """Count neighbors outside t+2 threat, bomb, and explosion cells."""
    _, second_step = monster_reachable_layers(model)

    escape_options = 0

    for neighbor in model.neighbors(position):
        if neighbor not in second_step:
            if neighbor not in model.bombs and neighbor not in model.explosions:
                escape_options += 1

    return escape_options


def measure_future_trap_risk(model: WorldModel, position: Position) -> float:
    """Penalize positions with very few future-safe escape choices."""
    options = measure_future_escape_options(model, position)
    if options == 0:
        return 1.0
    if options == 1:
        return 0.5
    return 0.0


def evaluate_position(
    model: WorldModel,
    position: Position,
    profile: EvaluationProfile = NORMAL_NAVIGATION,
) -> dict[str, float | bool]:
    lethal_cells = immediate_lethal_positions(model)
    lethal = position in lethal_cells
    mobility = float(measure_mobility(model, position, num_steps=2, forbidden_cells=lethal_cells))
    escape_options = float(measure_escape_options(model, position, lethal_cells))
    monster_threat = measure_threat(model, position)
    bomb_threat = measure_bomb_threat(model, position)
    explosion_threat = measure_explosion_threat(model, position)
    trap_risk = measure_trap_risk(model, position, lethal_cells)
    future_monster_risk = measure_future_monster_risk(model, position)
    future_escape_options = float(measure_future_escape_options(model, position))
    future_trap_risk = measure_future_trap_risk(model, position)
    return {
        "exit_progress": measure_exit_progress(model, position),
        "mobility": mobility,
        "escape_options": escape_options,
        "monster_threat": monster_threat,
        "bomb_threat": bomb_threat,
        "explosion_threat": explosion_threat,
        "trap_risk": trap_risk,
        "future_monster_risk": future_monster_risk,
        "future_escape_options": future_escape_options,
        "future_trap_risk": future_trap_risk,
        "lethal": lethal,
    }


def evaluate_action(
    model: WorldModel,
    action: AgentAction,
    profile: EvaluationProfile = NORMAL_NAVIGATION,
) -> EvaluationBreakdown:
    if model.self_position is None:
        destination = (0, 0)
        features = {
            "exit_progress": -100.0,
            "mobility": 0.0,
            "escape_options": 0.0,
            "monster_threat": 0.0,
            "bomb_threat": 0.0,
            "explosion_threat": 0.0,
            "trap_risk": 100.0,
            "future_monster_risk": 0.0,
            "future_escape_options": 0.0,
            "future_trap_risk": 100.0,
            "lethal": True,
        }
    else:
        sx, sy = model.self_position
        destination = (sx + action.dx, sy + action.dy)
        if not model.in_bounds(destination) or model.is_wall(destination):
            features = {
                "exit_progress": -100.0,
                "mobility": 0.0,
                "escape_options": 0.0,
                "monster_threat": 0.0,
                "bomb_threat": 0.0,
                "explosion_threat": 0.0,
                "trap_risk": 100.0,
                "future_monster_risk": 0.0,
                "future_escape_options": 0.0,
                "future_trap_risk": 100.0,
                "lethal": True,
            }
        else:
            features = evaluate_position(model, destination, profile)

    wait_penalty = 0.0
    if action.dx == 0 and action.dy == 0:
        wait_penalty = profile.wait_penalty

    exit_score = profile.exit_progress_weight * float(features["exit_progress"])
    mobility_score = profile.mobility_weight * float(features["mobility"])
    escape_score = profile.escape_options_weight * float(features["escape_options"])
    monster_penalty = profile.monster_threat_weight * float(features["monster_threat"])
    bomb_penalty = profile.bomb_threat_weight * float(features["bomb_threat"])
    explosion_penalty = profile.explosion_threat_weight * float(features["explosion_threat"])
    trap_penalty = profile.trap_risk_weight * float(features["trap_risk"])
    future_monster_penalty = profile.future_monster_risk_weight * float(features["future_monster_risk"])
    future_escape_score = profile.future_escape_options_weight * float(features["future_escape_options"])
    future_trap_penalty = profile.future_trap_risk_weight * float(features["future_trap_risk"])

    total = (
        exit_score
        + mobility_score
        + escape_score
        - monster_penalty
        - bomb_penalty
        - explosion_penalty
        - trap_penalty
        - future_monster_penalty
        + future_escape_score
        - future_trap_penalty
        - wait_penalty
    )
    if bool(features["lethal"]):
        total = -float("inf")

    return EvaluationBreakdown(
        action=action,
        destination=destination,
        exit_progress=float(features["exit_progress"]),
        mobility=float(features["mobility"]),
        escape_options=float(features["escape_options"]),
        monster_threat=float(features["monster_threat"]),
        bomb_threat=float(features["bomb_threat"]),
        explosion_threat=float(features["explosion_threat"]),
        trap_risk=float(features["trap_risk"]),
        future_monster_risk=float(features["future_monster_risk"]),
        future_escape_options=float(features["future_escape_options"]),
        future_trap_risk=float(features["future_trap_risk"]),
        wait_penalty=wait_penalty,
        lethal=bool(features["lethal"]),
        total=total,
    )


def rank_actions(
    model: WorldModel,
    actions: Iterable[AgentAction],
    profile: EvaluationProfile = NORMAL_NAVIGATION,
) -> list[EvaluationBreakdown]:
    """Return candidate actions from best to worst using stable tie-breaking."""
    scored = []
    for action in actions:
        scored.append(evaluate_action(model, action, profile))

    def ranking_key(result: EvaluationBreakdown) -> tuple[bool, float, int, int, int]:
        movement_size = abs(result.action.dx) + abs(result.action.dy)
        return (
            not result.lethal,
            result.total,
            movement_size,
            result.action.dy,
            result.action.dx,
        )

    return sorted(scored, key=ranking_key, reverse=True)
