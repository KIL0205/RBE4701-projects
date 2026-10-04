import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Bomberman"))

from team01.agent.actions import AgentAction
from team01.agent.evaluation import (
    EMERGENCY_ESCAPE,
    NORMAL_NAVIGATION,
    evaluate_action,
    measure_escape_options,
    measure_exit_transition,
    measure_mobility,
    q_feature_names,
    rank_actions,
)
from team01.agent.navigation import find_exit_path, find_path, measure_mobility
from team01.agent.world_model import WorldModel
from Bomberman.real_world import RealWorld
from Bomberman.sensed_world import SensedWorld
from team01.agent.q_learning import QAgent


def test_open_area_has_more_mobility_than_dead_end():
    open_model = WorldModel(5, 5)
    dead_end_model = WorldModel(5, 5)
    dead_end_model.walls = {
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
        (1, 2), (3, 2),
    }
    assert measure_mobility(open_model, (2, 2), 2) > measure_mobility(dead_end_model, (2, 2), 2)


def test_monster_threat_rejects_adjacent_destination():
    model = WorldModel(5, 5)
    model.self_position = (0, 0)
    model.monsters.add((2, 1))

    adjacent = evaluate_action(model, AgentAction(1, 0), EMERGENCY_ESCAPE)
    distant = evaluate_action(model, AgentAction(0, 1), EMERGENCY_ESCAPE)

    assert adjacent.lethal is True
    assert distant.monster_threat < adjacent.monster_threat


def test_safe_exit_progress_beats_safe_wait():
    model = WorldModel(5, 5, exit_position=(4, 4), self_position=(1, 1))
    toward_exit = evaluate_action(model, AgentAction(1, 1), NORMAL_NAVIGATION)
    wait = evaluate_action(model, AgentAction(0, 0), NORMAL_NAVIGATION)

    assert toward_exit.exit_progress > wait.exit_progress
    assert toward_exit.total > wait.total


def test_explosion_danger_beats_exit_progress():
    model = WorldModel(5, 5, exit_position=(4, 4), self_position=(1, 1))
    model.explosions.add((2, 2))

    dangerous = evaluate_action(model, AgentAction(1, 1), NORMAL_NAVIGATION)
    safe = evaluate_action(model, AgentAction(1, 0), NORMAL_NAVIGATION)

    assert dangerous.lethal is True
    assert safe.lethal is False
    assert safe.total > dangerous.total


def test_rank_actions_has_deterministic_order():
    model = WorldModel(3, 3, exit_position=(2, 2), self_position=(1, 1))
    actions = [AgentAction(0, 0), AgentAction(1, 1), AgentAction(1, 0)]

    first = rank_actions(model, actions)
    second = rank_actions(model, actions)

    assert first == second
    assert first[0].action == AgentAction(1, 1)


def test_exit_progress_on_open_and_wall_blocked_routes():
    current = WorldModel(4, 1, exit_position=(3, 0), self_position=(0, 0))
    predicted = WorldModel(4, 1, exit_position=(3, 0), self_position=(1, 0))
    progress, wall_progress = measure_exit_transition(current, predicted)
    assert progress > 0
    assert wall_progress == 0

    blocked = WorldModel(4, 1, exit_position=(3, 0), self_position=(0, 0))
    blocked.walls.add((2, 0))
    blocked_next = WorldModel(4, 1, exit_position=(3, 0), self_position=(1, 0))
    blocked_next.walls.add((2, 0))
    progress, wall_progress = measure_exit_transition(blocked, blocked_next)
    assert progress > 0
    assert progress > -float(blocked.width * blocked.height)
    assert wall_progress == 0
    assert find_exit_path(blocked, (0, 0), (3, 0)) == (3, 1)


def test_exit_wall_progress_and_moving_away():
    current = WorldModel(4, 1, exit_position=(3, 0), self_position=(0, 0))
    current.walls.add((1, 0))
    predicted = WorldModel(4, 1, exit_position=(3, 0), self_position=(0, 0))
    progress, wall_progress = measure_exit_transition(current, predicted)
    assert wall_progress > 0
    assert progress == 0

    toward_exit = WorldModel(4, 1, exit_position=(3, 0), self_position=(2, 0))
    moving_away = WorldModel(4, 1, exit_position=(3, 0), self_position=(1, 0))
    progress, _ = measure_exit_transition(toward_exit, moving_away)
    assert progress < 0


def test_exit_progress_without_exit_is_neutral_and_normal_path_still_blocks_walls():
    no_exit = WorldModel(4, 1, self_position=(0, 0))
    predicted = WorldModel(4, 1, self_position=(1, 0))
    assert measure_exit_transition(no_exit, predicted) == (0.0, 0.0)

    blocked = WorldModel(4, 1)
    blocked.walls.add((2, 0))
    assert find_path(blocked, (0, 0), (3, 0)) == []


def test_qagent_features_compare_candidates_without_mutating_sensed_world():
    world = RealWorld.from_params(10, 5, 10, 10, 2, 4)
    for y in range(5):
        world.add_wall(7, y)
    world.add_exit(9, 2)
    agent = QAgent("me", "C", 0, 0)
    world.add_character(agent)
    sensed = SensedWorld.from_world(world)

    me = sensed.me(agent)
    before = (me.x, me.y, me.dx, me.dy, me.maybe_place_bomb)
    wait_features = agent.features(sensed, AgentAction(0, 0))
    move_features = agent.features(sensed, AgentAction(1, 0))
    after_me = sensed.me(agent)
    after = (after_me.x, after_me.y, after_me.dx, after_me.dy, after_me.maybe_place_bomb)

    assert move_features["breach_site_approach_gain"] > 0.0
    assert move_features["open_route_progress_gain"] == 0.0
    assert set(move_features) == set(q_feature_names())
    assert before == after
