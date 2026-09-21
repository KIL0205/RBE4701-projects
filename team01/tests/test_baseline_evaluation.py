import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team01.agent.actions import AgentAction
from team01.agent.evaluation import (
    EMERGENCY_ESCAPE,
    NORMAL_NAVIGATION,
    evaluate_action,
    measure_escape_options,
    measure_mobility,
    rank_actions,
)
from team01.agent.navigation import measure_mobility
from team01.agent.world_model import WorldModel


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
