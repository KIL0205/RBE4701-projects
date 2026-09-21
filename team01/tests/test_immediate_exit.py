import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team01.agent.actions import AgentAction
from team01.agent.safety import find_executable_exit_action
from team01.agent.world_model import WorldModel

# Bomberman Agent was more worried about running away from the monster than escaping and winning the game immediately.
# This was testing the behavioral fix for the immediate exit override logic.
def test_adjacent_exit_wins_over_future_monster_risk():
    model = WorldModel(7, 7, exit_position=(3, 3), self_position=(2, 2))
    model.monsters.add((5, 5))

    assert find_executable_exit_action(model) == AgentAction(1, 1, False)


def test_adjacent_exit_wins_even_when_t2_reachability_contains_exit():
    model = WorldModel(7, 7, exit_position=(3, 3), self_position=(2, 2))
    model.monsters.add((1, 4))

    # The exit is conservatively reachable at t+2, but the queued move is
    # safe through the next monster update and wins immediately on arrival.
    assert find_executable_exit_action(model) == AgentAction(1, 1, False)


def test_current_cell_death_blocks_exit_override():
    model = WorldModel(7, 7, exit_position=(3, 3), self_position=(2, 2))
    model.monsters.add((1, 2))

    assert find_executable_exit_action(model) is None


def test_non_adjacent_exit_does_not_trigger_override():
    model = WorldModel(7, 7, exit_position=(4, 4), self_position=(2, 2))

    assert find_executable_exit_action(model) is None


def test_blocked_exit_destination_does_not_trigger_override():
    model = WorldModel(7, 7, exit_position=(3, 3), self_position=(2, 2))
    model.walls.add((3, 3))

    assert find_executable_exit_action(model) is None
