import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Bomberman"))

from Bomberman.real_world import RealWorld
from Bomberman.sensed_world import SensedWorld
from team01.agent.actions import AgentAction
from team01.agent.q_learning import QAgent
from team01.agent.evaluation import q_feature_names


def _game_with_agent():
    world = RealWorld.from_params(5, 5, 20, 4, 2, 2)
    world.add_exit(4, 4)
    agent = QAgent("me", "C", 0, 0, alpha=0.1, gamma=0.0, epsilon=0.0)
    world.add_character(agent)
    return world, agent, SensedWorld.from_world(world)


def test_bomb_candidate_uses_per_character_availability():
    world, agent, _ = _game_with_agent()
    other = QAgent("other", "O", 4, 0)
    world.add_character(other)
    world.add_bomb(3, 0, other)
    sensed = SensedWorld.from_world(world)
    bomb_action = AgentAction(0, 0, True)
    captured_actions = []
    agent.choose_action = lambda wrld, actions: captured_actions.extend(actions) or AgentAction(0, 0)

    agent.do(sensed)

    assert bomb_action in captured_actions

    blocked_world, blocked_agent, _ = _game_with_agent()
    blocked_world.add_bomb(3, 0, blocked_agent)
    blocked_sensed = SensedWorld.from_world(blocked_world)
    blocked_actions = []
    blocked_agent.choose_action = lambda wrld, actions: blocked_actions.extend(actions) or AgentAction(0, 0)

    blocked_agent.do(blocked_sensed)

    assert bomb_action not in blocked_actions


def test_bomb_action_prediction_selection_and_reward_continuity():
    world, agent, sensed = _game_with_agent()
    bomb_action = AgentAction(0, 0, True)
    source_me = sensed.me(agent)
    source_before = (source_me.x, source_me.y, source_me.dx, source_me.dy, source_me.maybe_place_bomb)

    predicted, _ = agent.predict_next_state(sensed, bomb_action)
    assert len(predicted.bombs) == 1
    predicted_bomb = next(iter(predicted.bombs.values()))
    assert (predicted_bomb.x, predicted_bomb.y) == (0, 0)
    assert predicted_bomb.owner.name == agent.name
    source_me_after = sensed.me(agent)
    assert source_before == (
        source_me_after.x,
        source_me_after.y,
        source_me_after.dx,
        source_me_after.dy,
        source_me_after.maybe_place_bomb,
    )

    features = agent.features(sensed, bomb_action)
    assert set(features) == set(q_feature_names())

    _, scored_actions = agent.max_q_value(sensed, [AgentAction(0, 0), bomb_action])
    assert bomb_action in [action for action, _ in scored_actions]

    with patch(
        "team01.agent.q_learning.random.choice",
        side_effect=lambda candidates: bomb_action if bomb_action in candidates else candidates[0],
    ):
        agent.do(sensed)
    assert agent.prev_action == bomb_action
    assert agent.maybe_place_bomb is True

    previous_bomb_threat = agent.prev_features["q_bias"]
    world.next()
    assert len(world.bombs) == 1

    agent.max_q_value = lambda wrld, actions: (0.0, [])
    agent.choose_action = lambda wrld, actions: next(action for action in actions if not action.place_bomb)
    agent.do(SensedWorld.from_world(world))

    expected_update = agent.alpha * agent.R_COST_OF_LIVING * previous_bomb_threat
    assert agent.weights["q_bias"] == expected_update