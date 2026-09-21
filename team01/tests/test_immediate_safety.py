import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team01.agent.actions import AgentAction
from team01.agent.safety import (
    assess_immediate_safety,
    monster_immediate_reachable_cells,
)
from team01.agent.world_model import WorldModel


def test_current_cell_threat_is_not_saved_by_queued_escape():
    model = WorldModel(5, 5, self_position=(2, 2))
    model.monsters.add((2, 1))

    result = assess_immediate_safety(model, AgentAction(1, 0))

    assert result.eligible is False
    assert result.current_cell_threatened is True
    assert result.rejection_reason == "MONSTER_CAN_REACH_CURRENT_CELL"
    assert result.destination == (3, 2)


def test_destination_reached_during_monster_update_is_rejected():
    model = WorldModel(5, 5, self_position=(0, 0))
    model.monsters.add((2, 1))

    result = assess_immediate_safety(model, AgentAction(1, 1))

    assert result.eligible is False
    assert result.current_cell_threatened is False
    assert result.destination_threatened is True
    assert result.rejection_reason == "MONSTER_CAN_REACH_DESTINATION"


def test_safe_escape_survives_current_and_destination_checks():
    model = WorldModel(6, 6, self_position=(0, 0))
    model.monsters.add((4, 4))

    result = assess_immediate_safety(model, AgentAction(1, 0))

    assert result.eligible is True
    assert result.rejection_reason is None
    assert result.current_cell_threatened is False
    assert result.destination_threatened is False


def test_walls_limit_monster_reachable_cells():
    model = WorldModel(5, 5)
    model.walls.add((2, 2))
    model.monsters.add((1, 1))

    reachable = monster_immediate_reachable_cells(model)

    assert (2, 2) not in reachable
    assert (1, 1) in reachable


def test_multiple_monsters_are_unioned():
    model = WorldModel(7, 7)
    model.monsters.update({(1, 1), (5, 5)})

    reachable = monster_immediate_reachable_cells(model)

    assert (0, 0) in reachable
    assert (6, 6) in reachable
