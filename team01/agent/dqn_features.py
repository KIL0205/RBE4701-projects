"""DQN schema: the 16 stable QAgent features plus one DQN-only input."""

from __future__ import annotations

import math

from Bomberman.sensed_world import SensedWorld

from .actions import AgentAction
from .evaluation import QFeatureContext, _blast_geometry, q_feature_names
from .q_learning import q_feature_vector
from .world_model import WorldModel


DQN_FEATURE_VERSION = 2
DQN_FEATURE_NAMES = ("direct_objective_progress",)


def dqn_feature_names() -> tuple[str, ...]:
    """Return the complete DQN schema: stable Q inputs followed by DQN inputs."""
    return q_feature_names() + DQN_FEATURE_NAMES


def _objective_site(
    model: WorldModel,
    context: QFeatureContext,
) -> tuple[int, int] | None:
    """Choose the closest reachable useful site for the first route wall."""
    if context.current_open_distance is not None:
        return model.exit_position
    if context.route_before is None:
        return None

    route_walls = [
        position
        for position in context.route_before[2]
        if model.is_wall(position)
    ]
    for wall in route_walls:
        candidates = []
        for site in context.breach_sites:
            path_distance = context.breach_site_distances.get(site)
            if path_distance is None:
                continue
            _, hit_walls = _blast_geometry(
                model,
                site,
                ignore_entities=True,
            )
            if wall in hit_walls:
                candidates.append(
                    (
                        math.dist(site, wall),
                        path_distance,
                        site[1],
                        site[0],
                        site,
                    )
                )
        if candidates:
            return min(candidates)[-1]
    return None


def dqn_feature_vector(
    agent,
    wrld: SensedWorld,
    action: AgentAction,
) -> tuple[float, ...]:
    """Return shared Q features plus normalized direct objective progress."""
    shared_features = q_feature_vector(agent, wrld, action)
    context = agent._q_feature_context
    if context is None or context.current_world is not wrld:
        raise RuntimeError("DQN feature context was not prepared for this world")

    model = context.current_model
    position = model.self_position
    objective = _objective_site(model, context)
    if position is None or objective is None:
        progress = 0.0
    else:
        current_distance = math.dist(position, objective)
        successor = (position[0] + action.dx, position[1] + action.dy)
        successor_distance = math.dist(successor, objective)
        scale = max(1, max(model.width, model.height) - 1)
        progress = max(
            -1.0,
            min(1.0, (current_distance - successor_distance) / scale),
        )

    return shared_features + (progress,)
