import heapq
from typing import List, Set, Tuple

from .world_model import WorldModel

Position = Tuple[int, int]


def _heuristic(a: Position, b: Position) -> float:
    """Estimate remaining path cost using diagonal grid distance."""
    ax, ay = a
    bx, by = b
    # return max(abs(bx - ax), abs(by - ay))
    # euclidean distance
    return ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5


def find_path(
    model: WorldModel,
    start: Position,
    goal: Position,
    forbidden_cells: Set[Position] | None = None,
) -> List[Position]:
    """Return an A* path including both the start and goal positions."""
    forbidden = forbidden_cells or set()
    if not model.in_bounds(start) or not model.in_bounds(goal):
        return []
    if model.is_wall(start) or model.is_wall(goal):
        return []
    if start == goal:
        return [start]

    came_from = {start: None}
    cost_so_far = {start: 0}
    frontier = []
    heapq.heappush(frontier, (_heuristic(start, goal), start))
    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in model.neighbors(current):
            if neighbor in forbidden and neighbor != goal:
                continue
            new_cost = cost_so_far[current] + _heuristic(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    return []

def measure_mobility(
    model: WorldModel,
    position: Position,
    num_steps: int,
    forbidden_cells: Set[Position] | None = None,
) -> int:
    """Count distinct positions reachable within ``num_steps`` moves."""
    forbidden = forbidden_cells or set()
    visited = {position}
    frontier = [(position, 0)]
    while frontier:
        current, steps = frontier.pop(0)
        if steps < num_steps:
            for neighbor in model.neighbors(current):
                if neighbor in forbidden:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, steps + 1))
    return len(visited) - 1  # exclude the starting position