import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team01.agent.black_board import BlackBoard
from team01.agent.behavior_tree import SelectorNode, Status
from team01.agent.actions import AgentAction
from team01.agent.navigation import find_path
from team01.agent.safety import filter_safe_actions, legal_candidate_actions, monster_threat_cells
from team01.agent.world_model import WorldModel


def test_blackboard_roundtrip():
    board = BlackBoard()
    board.set("answer", 42)
    assert board.get("answer") == 42
    assert "answer" in board.keys()


def test_behavior_tree_selector_uses_reactive_priority():
    class AlwaysFail:
        def tick(self):
            return Status.FAILURE

    class AlwaysSuccess:
        def tick(self):
            return Status.SUCCESS

    selector = SelectorNode("root")
    selector.add_child(AlwaysFail())
    selector.add_child(AlwaysSuccess())

    assert selector.tick() == Status.SUCCESS


def test_path_find_straight_line():
    model = WorldModel(5, 5)
    path = find_path(model, (0, 0), (4, 0))
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_path_find_avoids_walls():
    model = WorldModel(5, 5)
    for x in range(5):
        model.walls.add((x, 2))
    model.walls.add((1, 0))
    path = find_path(model, (0, 0), (4, 0))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 0)
    assert not any(pos in model.walls for pos in path)


def test_path_returns_empty_when_no_route_exists():
    model = WorldModel(3, 3)
    model.walls = {(0, 1), (1, 1), (2, 1)}
    path = find_path(model, (0, 0), (2, 2))
    assert path == []


def test_path_handles_start_equals_goal():
    model = WorldModel(3, 3)
    assert find_path(model, (1, 1), (1, 1)) == [(1, 1)]


def test_forbidden_cell_route_works():
    model = WorldModel(4, 4)
    forbidden = {(1, 1), (2, 1)}
    path = find_path(model, (0, 0), (3, 0), forbidden_cells=forbidden)
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_monster_threat_and_safe_action_filter():
    model = WorldModel(5, 5)
    model.self_position = (2, 2)
    model.monsters.add((2, 1))
    danger = monster_threat_cells(model, horizon=1)
    assert (2, 1) in danger
    assert (2, 2) in danger
    assert (2, 2) in monster_threat_cells(model, horizon=2)

    actions = legal_candidate_actions(model)
    safe = filter_safe_actions(model, actions, danger)
    assert AgentAction(0, 0, False) not in safe
    assert AgentAction(0, -1, False) not in safe
    assert AgentAction(0, 1, False) in safe
