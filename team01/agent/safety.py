# Safety-related utility functions for the Bomberman agent.
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Set, Tuple

from .actions import AgentAction
from .navigation import measure_mobility

Position = Tuple[int, int]

# Data structure representing the immediate safety assessment result for an action mostly for debugging purposes.
@dataclass(frozen=True)
class ImmediateSafetyResult:
    """Explain whether an action survives the next framework update."""
    action: AgentAction
    destination: Optional[Position]
    eligible: bool
    rejection_reason: Optional[str]
    current_cell_threatened: bool
    destination_threatened: bool
    monster_reachable_cells: Set[Position]


def monster_immediate_reachable_cells(model) -> Set[Position]:
    """Return cells any monster can occupy during the next monster update.

    The framework permits movement to any in-bounds non-wall cell in the
    monster's 8-neighborhood. Monsters may also remain in place.
    """
    reachable: Set[Position] = set()
    for monster in model.monster_positions():
        reachable.add(monster)
        x, y = monster
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                position = (x + dx, y + dy)
                if model.in_bounds(position) and not model.is_wall(position):
                    reachable.add(position)
    return reachable


def monster_reachable_layers(model) -> tuple[Set[Position], Set[Position]]:
    """Return separate conservative monster reachability layers for t+1/t+2.

    The t+1 layer is used for hard safety. The t+2 layer is only a heuristic
    risk signal, so callers must not merge it into the hard rejection set.
    """
    cached = getattr(model, "_monster_reachable_layers", None)
    if cached is not None:
        return cached
    current = set(model.monster_positions())
    first_step = set()
    for x, y in current:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                position = (x + dx, y + dy)
                if model.in_bounds(position) and not model.is_wall(position):
                    first_step.add(position)

    second_step: Set[Position] = set()
    for x, y in first_step:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                position = (x + dx, y + dy)
                if model.in_bounds(position) and not model.is_wall(position):
                    second_step.add(position)
    layers = (first_step, second_step)
    model._monster_reachable_layers = layers
    return layers


def short_horizon_survivability(model, position: Position, depth: int = 2) -> int:
    """Count safe positions reachable over the next few movement steps."""
    _, second_step = monster_reachable_layers(model)
    threat_layers = [second_step]
    frontier = set(second_step)

    # Extend the conservative monster threat envelope for the player search.
    for _ in range(depth - 1):
        next_frontier: Set[Position] = set()
        for cell in frontier:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if model.in_bounds(candidate) and not model.is_wall(candidate):
                        next_frontier.add(candidate)
        frontier = next_frontier
        threat_layers.append(frontier)

    player_frontier = {position}
    for threat in threat_layers:
        next_positions: Set[Position] = set()
        for current in player_frontier:
            candidates = [current] + model.neighbors(current)
            for candidate in candidates:
                if candidate in threat:
                    continue
                if candidate in model.bombs or candidate in model.explosions:
                    continue
                next_positions.add(candidate)

        player_frontier = next_positions
        if not player_frontier:
            return 0

    return len(player_frontier)


def assess_immediate_safety(model, action: AgentAction) -> ImmediateSafetyResult:
    """Check survival through the next monster and character update.

    Monsters move before the player's queued action is applied. That is why
    both the current cell and the requested destination are checked.
    """
    if model.self_position is None:
        return ImmediateSafetyResult(action, None, False, "NO_PLAYER", False, False, set())

    start = model.self_position
    destination = (start[0] + action.dx, start[1] + action.dy)
    reachable = monster_immediate_reachable_cells(model)

    if not model.in_bounds(destination):
        return ImmediateSafetyResult(action, destination, False, "OUT_OF_BOUNDS", start in reachable, False, reachable)
    if model.is_wall(destination):
        return ImmediateSafetyResult(action, destination, False, "WALL", start in reachable, destination in reachable, reachable)
    if destination in model.bombs:
        return ImmediateSafetyResult(action, destination, False, "BOMB_CELL", start in reachable, destination in reachable, reachable)
    if start in model.explosions:
        return ImmediateSafetyResult(action, destination, False, "CURRENT_EXPLOSION", True, destination in reachable, reachable)
    if destination in model.explosions:
        return ImmediateSafetyResult(action, destination, False, "DESTINATION_EXPLOSION", start in reachable, True, reachable)

    imminent_blast = set()
    for bomb, timer in model.bomb_timers.items():
        if timer <= 1:
            from .evaluation import bomb_blast_cells
            imminent_blast.update(bomb_blast_cells(model, bomb))
    if start in imminent_blast:
        return ImmediateSafetyResult(action, destination, False, "IMMINENT_BLAST_CURRENT", start in reachable, destination in reachable, reachable)
    if destination in imminent_blast:
        return ImmediateSafetyResult(action, destination, False, "IMMINENT_BLAST_DESTINATION", start in reachable, True, reachable)

    if start in reachable:
        # Current position is reachable by the monster.
        return ImmediateSafetyResult(action, destination, False, "MONSTER_CAN_REACH_CURRENT_CELL", True, destination in reachable, reachable)
    if destination in reachable:
        # Destination is reachable by the monster.
        return ImmediateSafetyResult(action, destination, False, "MONSTER_CAN_REACH_DESTINATION", False, True, reachable)
    return ImmediateSafetyResult(action, destination, True, None, False, False, reachable)


def find_executable_exit_action(model) -> Optional[AgentAction]:
    """Return a legal, next-update-safe move that immediately wins at the exit."""
    if model.self_position is None or model.exit_position is None:
        return None

    sx, sy = model.self_position
    ex, ey = model.exit_position
    action = AgentAction(ex - sx, ey - sy, False)
    if action not in legal_candidate_actions(model):
        return None
    if not assess_immediate_safety(model, action).eligible:
        return None
    return action


def safety_result_dict(result: ImmediateSafetyResult) -> dict:
    data = asdict(result)
    data["monster_reachable_cells"] = [list(cell) for cell in sorted(result.monster_reachable_cells)]
    return data


def monster_reachable_cells(model, monster: Position, horizon: int = 1) -> Set[Position]:
    if horizon < 1:
        return set()

    # Compute all cells that the monster can reach within the given horizon (bfs search)
    cells: Set[Position] = set()
    frontier = {monster}
    cells.add(monster)
    for _ in range(horizon):
        next_frontier: Set[Position] = set()
        for x, y in frontier:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    target = (x + dx, y + dy)
                    if not model.in_bounds(target) or target in model.walls:
                        continue
                    next_frontier.add(target)
        cells.update(next_frontier)
        frontier = next_frontier
    return cells


def monster_threat_cells(model, horizon: int = 2) -> Set[Position]:
    danger: Set[Position] = set()
    for monster in model.monster_positions():
        danger.update(monster_reachable_cells(model, monster, horizon=horizon))
    return danger


def immediate_danger_positions(model, danger_radius: int = 1) -> Set[Position]:
    danger: Set[Position] = set()

    for monster in model.monster_positions():
        mx, my = monster
        for dx in range(-danger_radius, danger_radius + 1):
            for dy in range(-danger_radius, danger_radius + 1):
                pos = (mx + dx, my + dy)
                if model.in_bounds(pos):
                    danger.add(pos)

    for bomb in model.bomb_positions():
        x, y = bomb
        danger.add((x, y))

    for explosion in model.explosions:
        x, y = explosion
        danger.add((x, y))

    return danger


def legal_candidate_actions(model) -> List[AgentAction]:
    """Return legal movement actions, including waiting in place."""
    if model.self_position is None:
        return []
    actions: List[AgentAction] = []
    sx, sy = model.self_position
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            target = (sx + dx, sy + dy)
            if not model.in_bounds(target):
                continue
            if target in model.walls:
                continue
            if target in model.bombs:
                continue
            actions.append(AgentAction(dx, dy, False))
    return actions


def filter_safe_actions(model, candidate_actions: Iterable[AgentAction], danger_cells: Set[Position]) -> List[AgentAction]:
    """Keep candidates whose destinations are outside the supplied danger set."""
    safe: List[AgentAction] = []
    if model.self_position is None:
        return safe

    sx, sy = model.self_position
    for action in candidate_actions:
        target = (sx + action.dx, sy + action.dy)
        if target in danger_cells or target in model.explosions:
            continue
        safe.append(action)
    return safe

def measure_danger(model, position: Position, danger_radius: int = 1) -> int:
    danger_level = 0
    px, py = position

    MONSTER_DANGER_WEIGHT = 30
    BOMB_DANGER_WEIGHT = 10
    EXPLOSION_DANGER_WEIGHT = 10

    for monster in model.monster_positions():
        mx, my = monster
        step_distance = max(abs(mx - px), abs(my - py))
        if abs(mx - px) <= danger_radius and abs(my - py) <= danger_radius:
            danger_level += MONSTER_DANGER_WEIGHT / max(1, step_distance)

    for bomb in model.bomb_positions():
        bx, by = bomb
        step_distance = max(abs(bx - px), abs(by - py))
        if abs(bx - px) <= danger_radius and abs(by - py) <= danger_radius:
            danger_level += BOMB_DANGER_WEIGHT / max(1, step_distance)

    for explosion in model.explosions:
        ex, ey = explosion
        step_distance = max(abs(ex - px), abs(ey - py))
        if abs(ex - px) <= danger_radius and abs(ey - py) <= danger_radius:
            danger_level += EXPLOSION_DANGER_WEIGHT / max(1, step_distance)

    #TODO: Consider adding mobility factor to danger assessment
    return danger_level