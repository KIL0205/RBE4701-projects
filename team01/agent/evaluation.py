from dataclasses import dataclass
from collections import deque
import heapq
import math
from typing import Iterable, Mapping, Set

from Bomberman.events import Event
from .actions import AgentAction
from .navigation import find_exit_path, find_path, measure_mobility
from .safety import monster_reachable_cells, monster_reachable_layers
from .world_model import Position, WorldModel

Q_FEATURE_VERSION = 3
Q_FEATURE_NAMES = (
    "q_bias",
    "win_next_update",
    "loss_next_update",
    "explosion_threat",
    "open_route_progress_gain",
    "breach_site_approach_gain",
    "bomb_wall_route_gain",
    "monster_in_clear_blast_ray",
    "bomb_escape_margin",
    "active_bomb_escape_margin",
    "unproductive_bomb",
    "safe_successor_fraction",
    "escape_route_diversity",
    "monster_clearance",
    "monster_fuse_envelope_overlap",
    "urgency_scaled_objective_progress",
)


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

QLEARNING = EvaluationProfile(
    exit_progress_weight=1.0,
    mobility_weight=1.0,
    escape_options_weight=1.0,
    monster_threat_weight=1.0,
    bomb_threat_weight=1.0,
    explosion_threat_weight=1.0,
    trap_risk_weight=1.0,
    future_monster_risk_weight=1.0,
    future_escape_options_weight=1.0,
    future_trap_risk_weight=1.0,
    wait_penalty=1.0,
)

DEFAULT_PROFILES = {
    "normal": NORMAL_NAVIGATION,
    "emergency": EMERGENCY_ESCAPE,
    "fallback": FALLBACK,
}


def profiles_from_mapping(
    weights: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, EvaluationProfile]:
    """Build evaluation profiles, overriding only supplied in-memory weights."""
    profiles = dict(DEFAULT_PROFILES)
    if weights is None:
        return profiles
    for profile_name, values in weights.items():
        if profile_name not in profiles:
            raise ValueError(f"unknown evaluation profile: {profile_name}")
        profile_values = {field: float(value) for field, value in values.items()}
        profiles[profile_name] = EvaluationProfile(
            **{**profiles[profile_name].__dict__, **profile_values}
        )
    return profiles


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


def measure_exit_transition(
    current_model: WorldModel,
    predicted_model: WorldModel,
) -> tuple[float, float]:
    """Return spatial and wall progress from a current-to-predicted transition."""
    if (
        current_model.exit_position is None
        or predicted_model.exit_position is None
        or current_model.self_position is None
        or predicted_model.self_position is None
    ):
        return 0.0, 0.0

    before = find_exit_path(
        current_model,
        current_model.self_position,
        current_model.exit_position,
    )
    after = find_exit_path(
        predicted_model,
        predicted_model.self_position,
        predicted_model.exit_position,
    )
    if before is None or after is None:
        return 0.0, 0.0

    return float(before[0] - after[0]), float(before[1] - after[1])


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
    loop_radius = measure_max_loop_radius(model)

    size_multiplier = 1/(loop_radius + 1)
    retval = 0.25
    if options == 0:
        retval = 1.0
    if options == 1:
        retval = 0.5
    return retval * size_multiplier


def measure_max_loop_radius(model: WorldModel) -> int:
    """Returns the radius of the largest loop in the map"""
    max_loop_len = 0
    for group in model.wallgroups:
        prev = True
        swaps = 0
        for wall in group:
            # TODO: May actually be unordered, not L -> R on screen order. Due to set notation. could be ordered by x coord afterwards.
            still_wall = model.is_wall(wall)
            if prev != still_wall:
                swaps += 1
                prev = still_wall
            # This implementation only checks for the first loop in a group.
            if swaps == 1:
                loop_start = group.index(wall)
            if swaps == 3:
                loop_len = group.index(wall) - loop_start - 1
                if loop_len > max_loop_len:
                    max_loop_len = loop_len
                break
    return max_loop_len


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

# Q feature normalization and bounds
_Q_FEATURE_BOUNDS = {
    "q_bias": (1.0, 1.0),
    "win_next_update": (0.0, 1.0),
    "loss_next_update": (0.0, 1.0),
    "explosion_threat": (0.0, 1.0),
    "open_route_progress_gain": (-1.0, 1.0),
    "breach_site_approach_gain": (-1.0, 1.0),
    "bomb_wall_route_gain": (0.0, 1.0),
    "monster_in_clear_blast_ray": (0.0, 1.0),
    "bomb_escape_margin": (-1.0, 1.0),
    "active_bomb_escape_margin": (-1.0, 1.0),
    "unproductive_bomb": (0.0, 1.0),
    "safe_successor_fraction": (0.0, 1.0),
    "escape_route_diversity": (0.0, 1.0),
    "monster_clearance": (0.0, 1.0),
    "monster_fuse_envelope_overlap": (0.0, 1.0),
    "urgency_scaled_objective_progress": (-1.0, 1.0),
}

# Normalize Q features based on predefined bounds
def normalize_q_features(
    features: Mapping[str, float],
    model: WorldModel | None = None,
) -> dict[str, float]:
    """Validate and bound the active Q feature schema."""
    del model
    if set(features) != set(Q_FEATURE_NAMES):
        missing = set(Q_FEATURE_NAMES) - set(features)
        stale = set(features) - set(Q_FEATURE_NAMES)
        raise ValueError(f"Q feature schema mismatch: missing={missing}, extra={stale}")

    normalized = {}
    for name in Q_FEATURE_NAMES:
        value = float(features[name])
        if not math.isfinite(value):
            raise ValueError(f"Q feature {name} is not finite: {value}")
        lower, upper = _Q_FEATURE_BOUNDS[name]
        normalized[name] = max(lower, min(upper, value))
    return normalized

# Compute move distances from a starting position, considering blocked positions.
def _move_distances(
    model: WorldModel,
    start: Position,
    blocked: Set[Position] | None = None,
) -> dict[Position, int]:
    forbidden = blocked or set()
    distances = {start: 0}
    frontier = deque([start])
    while frontier:
        current = frontier.popleft()
        for neighbor in model.neighbors(current):
            if neighbor in distances or neighbor in forbidden:
                continue
            distances[neighbor] = distances[current] + 1
            frontier.append(neighbor)
    return distances

# Use A* search to find an optimal route considering removed walls and bombs.
def _route_profile(
    model: WorldModel,
    removed_walls: Iterable[Position] = (),
    removed_bombs: Iterable[Position] = (),
) -> tuple[int, int, list[Position]] | None:
    """Find a route minimizing destructible walls, then moves."""
    start = model.self_position
    goal = model.exit_position
    if start is None or goal is None or not model.in_bounds(start) or not model.in_bounds(goal):
        return None
    cleared = set(removed_walls)
    cleared_bombs = set(removed_bombs)
    best = {start: (0, 0)}
    previous: dict[Position, Position | None] = {start: None}
    frontier = [(0, 0, start[0], start[1])]
    while frontier:
        walls_crossed, steps, x, y = heapq.heappop(frontier)
        current = (x, y)
        if best.get(current) != (walls_crossed, steps):
            continue
        if current == goal:
            path = []
            cursor: Position | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = previous[cursor]
            path.reverse()
            return walls_crossed, steps, path
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if (
                    not model.in_bounds(neighbor)
                    or neighbor in model.bombs and neighbor not in cleared_bombs
                ):
                    continue
                wall_cost = int(model.is_wall(neighbor) and neighbor not in cleared)
                candidate = (walls_crossed + wall_cost, steps + 1)
                if neighbor not in best or candidate < best[neighbor]:
                    best[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(frontier, (*candidate, neighbor[0], neighbor[1]))
    return None

# Compute a score for a given route profile, balancing walls crossed and steps taken.
def _route_score(profile: tuple[int, int, list[Position]], model: WorldModel) -> float:
    wall_count, steps, _ = profile
    return wall_count + steps / max(2, model.width * model.height + 1)

# Compute the gain in route efficiency after removing certain walls.
def _route_gain(
    model: WorldModel,
    removed_walls: Iterable[Position],
    before: tuple[int, int, list[Position]] | None = None,
    removed_bombs: Iterable[Position] = (),
) -> float:
    route_before = before if before is not None else _route_profile(model)
    if route_before is None:
        return 0.0
    route_after = _route_profile(model, removed_walls, removed_bombs)
    if route_after is None:
        return 0.0
    denominator = max(1.0, route_before[0] + 1.0)
    return max(0.0, min(1.0, (_route_score(route_before, model) - _route_score(route_after, model)) / denominator))

# Compute the actual blast geometry from a given origin, considering walls and entities.
def _blast_geometry(
    model: WorldModel,
    origin: Position,
    *,
    ignore_entities: bool = False,
    ignored_character_names: Set[str] | None = None,
) -> tuple[Set[Position], Set[Position]]:
    """Return actual cardinal blast cells and the first destructible wall per ray."""
    blast = {origin}
    hit_walls: Set[Position] = set()
    ignored = ignored_character_names or set()
    characters = {position for name, position in model.characters.items() if name not in ignored}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, max(0, model.explosion_range) + 1):
            position = (origin[0] + dx * distance, origin[1] + dy * distance)
            if not model.in_bounds(position):
                break
            if position == model.exit_position or position in model.bombs:
                break
            blast.add(position)
            if model.is_wall(position):
                hit_walls.add(position)
                break
            if not ignore_entities and (
                position in model.monsters or position in characters
            ):
                break
    return blast, hit_walls

# Identify useful breach sites along the current route, considering explosion range and walls.
def _useful_breach_sites(
    model: WorldModel,
    route_before: tuple[int, int, list[Position]] | None,
) -> Set[Position]:
    if route_before is None or route_before[0] == 0 or model.exit_position is None:
        return set()
    route_walls = {position for position in route_before[2] if model.is_wall(position)}
    sites: Set[Position] = set()
    for wall in route_walls:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for distance in range(1, max(0, model.explosion_range) + 1):
                site = (wall[0] - dx * distance, wall[1] - dy * distance)
                if not model.in_bounds(site):
                    continue
                if model.is_wall(site) or site in model.bombs or site == model.exit_position:
                    continue
                _, hit_walls = _blast_geometry(model, site, ignore_entities=True)
                route_hits = hit_walls & route_walls
                if route_hits and _route_gain(
                    model,
                    route_hits,
                    route_before,
                    removed_bombs={site},
                ) > 0.0:
                    sites.add(site)
    return sites


@dataclass(frozen=True)
class QFeatureContext:
    current_world: object
    current_model: WorldModel
    route_before: tuple[int, int, list[Position]] | None
    current_open_distance: int | None
    breach_sites: frozenset[Position]
    breach_site_distances: dict[Position, int]
    time_limit: int

# Prepare the context for Q-feature computation, precomputing route and breach site information.
def prepare_q_feature_context(
    current_world,
    current_model: WorldModel,
    time_limit: int,
) -> QFeatureContext:
    """Precompute state-only route data once for every candidate in this state."""
    start = current_model.self_position
    goal = current_model.exit_position
    route_before = _route_profile(current_model) if start is not None and goal else None
    current_distances = _move_distances(current_model, start) if start is not None else {}
    current_open_distance = current_distances.get(goal) if goal is not None else None
    sites = (
        _useful_breach_sites(current_model, route_before)
        if current_open_distance is None
        else set()
    )
    breach_distances = (
        _move_distances(current_model, start)
        if start is not None and sites
        else {}
    )
    return QFeatureContext(
        current_world=current_world,
        current_model=current_model,
        route_before=route_before,
        current_open_distance=current_open_distance,
        breach_sites=frozenset(sites),
        breach_site_distances=breach_distances,
        time_limit=max(1, int(time_limit)),
    )


# Compute the distance to the nearest site from the start position.
def _nearest_site_distance(
    model: WorldModel,
    start: Position | None,
    sites: Set[Position] | frozenset[Position],
    distances: dict[Position, int] | None = None,
) -> int | None:
    if start is None or not sites:
        return None
    if distances is None:
        distances = _move_distances(model, start)
    return min((distances[site] for site in sites if site in distances), default=None)


def _distance_to_safe_cell(
    model: WorldModel,
    start: Position | None,
    danger: Set[Position],
    blocked: Set[Position] | None = None,
    max_steps: int | None = None,
) -> int | None:
    if start is None:
        return None
    forbidden = set(blocked or ())
    forbidden.update(model.monsters)
    forbidden.update(model.explosions)
    forbidden.discard(start)
    distances = _move_distances(model, start, forbidden)
    safe = [
        distance
        for position, distance in distances.items()
        if position not in danger and (max_steps is None or distance <= max_steps)
    ]
    return min(safe, default=None)


def _bomb_escape_margin(
    model: WorldModel,
    origin: Position | None,
    agent_name: str,
) -> float:
    if origin is None:
        return -1.0
    blast, _ = _blast_geometry(
        model,
        origin,
        ignore_entities=True,
        ignored_character_names={agent_name},
    )
    available = max(0, int(model.bomb_time))
    blocked = set(model.walls) | set(model.bombs)
    blocked.discard(origin)
    required = _distance_to_safe_cell(model, origin, blast, blocked)
    if required is None:
        return -1.0
    return max(-1.0, min(1.0, (available - required) / max(1, available)))


def _active_bomb_escape_margin(
    model: WorldModel,
    excluded_bombs: Set[Position] | None = None,
    reference_position: Position | None = None,
) -> float:
    position = model.self_position
    if position is None or not model.bomb_timers:
        return 0.0
    excluded = excluded_bombs or set()
    threats = []
    for bomb, timer in model.bomb_timers.items():
        if bomb in excluded:
            continue
        blast, _ = _blast_geometry(model, bomb, ignore_entities=True)
        if position in blast or (reference_position is not None and reference_position in blast):
            threats.append((bomb, max(0, int(timer)), blast))
    if not threats:
        return 0.0
    margins = []
    for bomb, available, blast in threats:
        if position not in blast:
            margins.append(1.0)
            continue
        simultaneous_hazards = set(model.explosions)
        for other_bomb, other_timer in model.bomb_timers.items():
            if other_bomb in excluded:
                continue
            if max(0, int(other_timer)) <= available:
                other_blast, _ = _blast_geometry(model, other_bomb, ignore_entities=True)
                simultaneous_hazards.update(other_blast)
        required = _distance_to_safe_cell(
            model,
            position,
            simultaneous_hazards,
            set(model.walls) | set(model.bombs),
        )
        margin = -1.0 if required is None else (available - required) / max(1, available)
        margins.append(max(-1.0, min(1.0, margin)))
    return min(margins, default=0.0)


def _escape_route_diversity(
    model: WorldModel,
    start: Position | None,
    danger: Set[Position],
    horizon: int,
) -> float:
    if start is None or horizon < 1:
        return 0.0
    blocked = set(model.walls) | set(model.bombs) | set(model.monsters) | set(model.explosions)
    blocked.discard(start)
    viable = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            first = (start[0] + dx, start[1] + dy)
            if not model.in_bounds(first) or first in blocked:
                continue
            remaining = horizon - 1
            distances = _move_distances(model, first, blocked)
            if any(
                position not in danger and distance <= remaining
                for position, distance in distances.items()
            ):
                viable += 1
    return viable / 8.0


def _monster_fuse_envelope_overlap(
    model: WorldModel,
    blast: Set[Position],
    horizon: int,
) -> float:
    if not blast or not model.monsters:
        return 0.0
    overlaps = []
    for monster in model.monsters:
        reachable = monster_reachable_cells(model, monster, horizon)
        overlaps.append(len(reachable & blast) / len(blast))
    return max(overlaps, default=0.0)


def _event_feature(events: Iterable, feature: str, agent_name: str) -> float:
    for event in events:
        if (
            feature == "win"
            and event.tpe == Event.CHARACTER_FOUND_EXIT
            and event.character.name == agent_name
        ):
            return 1.0
        if feature == "loss":
            if (
                event.tpe == Event.CHARACTER_KILLED_BY_MONSTER
                and event.character.name == agent_name
            ):
                return 1.0
            if (
                event.tpe == Event.BOMB_HIT_CHARACTER
                and event.other is not None
                and event.other.name == agent_name
            ):
                return 1.0
    return 0.0


def normalize_q_features(features: Mapping[str, float]) -> dict[str, float]:
    """Validate and bound the active Q feature schema."""
    if set(features) != set(Q_FEATURE_NAMES):
        missing = set(Q_FEATURE_NAMES) - set(features)
        stale = set(features) - set(Q_FEATURE_NAMES)
        raise ValueError(f"Q feature schema mismatch: missing={missing}, extra={stale}")
    normalized = {}
    for name in Q_FEATURE_NAMES:
        value = float(features[name])
        if not math.isfinite(value):
            raise ValueError(f"Q feature {name} is not finite: {value}")
        lower, upper = _Q_FEATURE_BOUNDS[name]
        normalized[name] = max(lower, min(upper, value))
    return normalized


def evaluate_q_features(
    current_world,
    predicted_world,
    current_model: WorldModel,
    predicted_model: WorldModel,
    action: AgentAction,
    agent_name: str = "me",
    time_limit: int | None = None,
    predicted_events: Iterable = (),
    context: QFeatureContext | None = None,
) -> dict[str, float]:
    """Build the complete action-conditioned Project 2 Q feature vector."""
    start = current_model.self_position
    position = predicted_model.self_position
    goal = current_model.exit_position
    dimensions = max(1, max(current_model.width, current_model.height) - 1)
    if context is None or context.current_world is not current_world:
        context = prepare_q_feature_context(
            current_world,
            current_model,
            int(time_limit if time_limit is not None else current_world.time),
        )
    route_before = context.route_before
    current_open_distance = context.current_open_distance
    predicted_open_distance = (
        _move_distances(predicted_model, position).get(goal)
        if position is not None and goal is not None
        else None
    )

    open_progress = 0.0
    if current_open_distance is not None:
        if predicted_open_distance is None:
            open_progress = -1.0
        else:
            open_progress = (current_open_distance - predicted_open_distance) / dimensions

    breach_progress = 0.0
    if current_open_distance is None and not action.place_bomb:
        sites = context.breach_sites
        before_distance = _nearest_site_distance(
            current_model,
            start,
            sites,
            context.breach_site_distances,
        )
        after_distance = _nearest_site_distance(predicted_model, position, sites)
        if before_distance is not None:
            breach_progress = (
                -1.0
                if after_distance is None
                else (before_distance - after_distance) / dimensions
            )

    bomb_wall_gain = 0.0
    monster_in_ray = 0.0
    bomb_escape_margin = 0.0
    unproductive_bomb = 0.0
    monster_fuse_overlap = 0.0
    bomb_blast: Set[Position] = set()
    if action.place_bomb and start is not None:
        bomb_blast, hit_walls = _blast_geometry(
            predicted_model,
            start,
            ignored_character_names={agent_name},
        )
        bomb_wall_gain = _route_gain(
            predicted_model,
            hit_walls,
            removed_bombs={start},
        )
        monster_in_ray = float(any(monster in bomb_blast for monster in predicted_model.monsters))
        bomb_escape_margin = _bomb_escape_margin(predicted_model, start, agent_name)
        unproductive_bomb = float(bomb_wall_gain <= 1e-9 and monster_in_ray == 0.0)
        monster_fuse_overlap = _monster_fuse_envelope_overlap(
            predicted_model,
            bomb_blast,
            max(0, int(predicted_world.bomb_time)),
        )

    active_explosion_cells = set(predicted_model.explosions)
    explosion_threat = float(
        position is not None and position in active_explosion_cells
    )

    newly_placed_bombs = {start} if action.place_bomb and start is not None else set()
    active_margin = _active_bomb_escape_margin(
        predicted_model,
        newly_placed_bombs,
        reference_position=current_model.self_position,
    )
    active_bomb_positions = set(predicted_model.bomb_timers)
    active_bomb_positions.difference_update(newly_placed_bombs)
    active_danger: Set[Position] = set(active_explosion_cells)
    for bomb in active_bomb_positions:
        timer = predicted_model.bomb_timers[bomb]
        if timer <= max(0, int(predicted_world.bomb_time)):
            blast, _ = _blast_geometry(predicted_model, bomb, ignore_entities=True)
            active_danger.update(blast)

    safe_successors = 0.0
    if position is not None:
        successors = list(predicted_model.neighbors(position))
        if position not in predicted_model.bombs:
            successors.append(position)
        imminent = set(active_explosion_cells)
        imminent.update(monster_immediate_cells(predicted_model))
        for bomb in active_bomb_positions:
            if predicted_model.bomb_timers[bomb] <= 0:
                blast, _ = _blast_geometry(predicted_model, bomb, ignore_entities=True)
                imminent.update(blast)
        safe_successors = (
            sum(cell not in imminent for cell in successors) / len(successors)
            if successors
            else 0.0
        )

    escape_route_diversity = 0.0
    if action.place_bomb:
        escape_route_diversity = _escape_route_diversity(
            predicted_model,
            position,
            active_danger | bomb_blast,
            max(0, int(predicted_world.bomb_time)),
        )
    elif active_margin != 0.0:
        threatened_timers = [
            timer
            for bomb, timer in predicted_model.bomb_timers.items()
            if position is not None
            and position in _blast_geometry(predicted_model, bomb, ignore_entities=True)[0]
        ]
        if threatened_timers:
            escape_route_diversity = _escape_route_diversity(
                predicted_model,
                position,
                active_danger,
                max(0, min(threatened_timers)),
            )

    if position is None or not predicted_model.monsters:
        monster_clearance = 1.0
    else:
        nearest = min(
            max(abs(position[0] - monster[0]), abs(position[1] - monster[1]))
            for monster in predicted_model.monsters
        )
        monster_clearance = nearest / max(1, max(predicted_model.width, predicted_model.height) - 1)

    urgency = max(0.0, min(1.0, 1.0 - current_world.time / context.time_limit))
    objective_progress = open_progress if current_open_distance is not None else breach_progress
    if _event_feature(predicted_events, "win", agent_name):
        objective_progress = 1.0

    win_next_update = _event_feature(predicted_events, "win", agent_name)
    loss_event = _event_feature(predicted_events, "loss", agent_name)
    player_alive = any(
        character.name == agent_name
        for group in predicted_world.characters.values()
        for character in group
    )
    loss_next_update = float(loss_event == 1.0 or (not player_alive and win_next_update == 0.0))

    features = {
        "q_bias": 1.0,
        "win_next_update": win_next_update,
        "loss_next_update": loss_next_update,
        "explosion_threat": explosion_threat,
        "open_route_progress_gain": open_progress,
        "breach_site_approach_gain": breach_progress,
        "bomb_wall_route_gain": bomb_wall_gain,
        "monster_in_clear_blast_ray": monster_in_ray,
        "bomb_escape_margin": bomb_escape_margin,
        "active_bomb_escape_margin": active_margin,
        "unproductive_bomb": unproductive_bomb,
        "safe_successor_fraction": safe_successors,
        "escape_route_diversity": escape_route_diversity,
        "monster_clearance": monster_clearance,
        "monster_fuse_envelope_overlap": monster_fuse_overlap,
        "urgency_scaled_objective_progress": objective_progress * urgency,
    }
    return normalize_q_features(features)


def q_feature_names() -> tuple[str, ...]:
    """Return feature names from the current Q feature definition."""
    return Q_FEATURE_NAMES


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
