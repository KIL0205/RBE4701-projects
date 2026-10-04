import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Bomberman"))

from Bomberman.entity import MonsterEntity
from Bomberman.real_world import RealWorld
from Bomberman.sensed_world import SensedWorld
from team01.agent.actions import AgentAction
from team01.agent.evaluation import Q_FEATURE_NAMES, evaluate_q_features
from team01.agent.q_learning import QAgent
from team01.agent.world_model import WorldModel


def _world(
    width=5,
    height=5,
    start=(1, 1),
    exit_position=(4, 4),
    walls=(),
    bomb_time=10,
    explosion_range=4,
):
    world = RealWorld.from_params(width, height, 100, bomb_time, 2, explosion_range)
    if exit_position is not None:
        world.add_exit(*exit_position)
    for wall in walls:
        world.add_wall(*wall)
    agent = QAgent("me", "C", *start)
    world.add_character(agent)
    return world, agent


def _features(agent, world, action):
    sensed = SensedWorld.from_world(world)
    return agent.features(sensed, action)


def test_new_schema_names_bias_and_ranges():
    world, agent = _world()
    features = _features(agent, world, AgentAction(0, 0))

    assert len(Q_FEATURE_NAMES) == 16
    assert set(features) == set(Q_FEATURE_NAMES)
    assert features["q_bias"] == 1.0
    assert all(math.isfinite(value) for value in features.values())
    assert all(-1.0 <= value <= 1.0 for key, value in features.items() if key != "q_bias")
    assert features["explosion_threat"] == 0.0


def test_one_step_exit_and_death_events_are_terminal_features():
    winning_world, winning_agent = _world(width=3, height=3, start=(1, 1), exit_position=(2, 2))
    won = _features(winning_agent, winning_world, AgentAction(1, 1))
    assert won["win_next_update"] == 1.0
    assert won["loss_next_update"] == 0.0

    deadly_world, deadly_agent = _world(width=5, height=3, start=(0, 1), exit_position=(4, 1))
    deadly_world.add_bomb(2, 1, deadly_agent)
    deadly_world.bombs[deadly_world.index(2, 1)].timer = 0
    lost = _features(deadly_agent, deadly_world, AgentAction(1, 0))
    assert lost["loss_next_update"] == 1.0


def test_open_route_progress_is_action_conditioned_and_blocked_route_uses_breach_site():
    world, agent = _world(start=(2, 2), exit_position=(4, 4))
    toward = _features(agent, world, AgentAction(1, 1))
    away = _features(agent, world, AgentAction(-1, -1))
    assert toward["open_route_progress_gain"] > 0.0
    assert away["open_route_progress_gain"] < 0.0

    blocked, blocked_agent = _world(
        width=7,
        height=1,
        start=(0, 0),
        exit_position=(6, 0),
        walls={(5, 0)},
    )
    moving_to_breach = _features(blocked_agent, blocked, AgentAction(1, 0))
    waiting = _features(blocked_agent, blocked, AgentAction(0, 0))
    assert moving_to_breach["open_route_progress_gain"] == 0.0
    assert moving_to_breach["breach_site_approach_gain"] > waiting["breach_site_approach_gain"]


def test_bomb_route_gain_and_unproductive_bomb_use_actual_blast_blockers():
    blocked, agent = _world(
        width=5,
        height=1,
        start=(0, 0),
        exit_position=(4, 0),
        walls={(2, 0)},
        explosion_range=4,
    )
    bomb = _features(agent, blocked, AgentAction(0, 0, True))
    move = _features(agent, blocked, AgentAction(0, 0))
    assert bomb["bomb_wall_route_gain"] > 0.0
    assert bomb["unproductive_bomb"] == 0.0
    assert bomb["monster_in_clear_blast_ray"] == 0.0
    assert move["bomb_wall_route_gain"] == 0.0
    assert move["unproductive_bomb"] == 0.0

    pointless, pointless_agent = _world(width=5, height=5, start=(2, 2), exit_position=(4, 4))
    pointless_features = _features(pointless_agent, pointless, AgentAction(0, 0, True))
    assert pointless_features["unproductive_bomb"] == 1.0

    clear_ray, clear_ray_agent = _world(width=6, height=1, start=(0, 0), exit_position=(5, 0))
    clear_ray.add_monster(MonsterEntity("target", "S", 3, 0))
    ray_features = _features(clear_ray_agent, clear_ray, AgentAction(0, 0, True))
    assert ray_features["monster_in_clear_blast_ray"] == 1.0

    blocked_ray, blocked_ray_agent = _world(
        width=6,
        height=1,
        start=(0, 0),
        exit_position=(5, 0),
        walls={(2, 0)},
    )
    blocked_ray.add_monster(MonsterEntity("target", "S", 3, 0))
    blocked_ray_features = _features(blocked_ray_agent, blocked_ray, AgentAction(0, 0, True))
    assert blocked_ray_features["monster_in_clear_blast_ray"] == 0.0


def test_bomb_escape_and_active_bomb_escape_margins_reflect_timing():
    open_world, open_agent = _world(width=3, height=3, start=(1, 1), exit_position=(2, 2))
    safe_bomb = _features(open_agent, open_world, AgentAction(0, 0, True))
    assert safe_bomb["bomb_escape_margin"] > 0.0
    assert safe_bomb["active_bomb_escape_margin"] == 0.0

    corridor, corridor_agent = _world(
        width=5,
        height=1,
        start=(0, 0),
        exit_position=(4, 0),
        bomb_time=3,
        explosion_range=4,
    )
    trapped = _features(corridor_agent, corridor, AgentAction(0, 0, True))
    assert trapped["bomb_escape_margin"] < 0.0

    existing, existing_agent = _world(width=5, height=3, start=(0, 1), exit_position=(4, 2))
    existing.add_bomb(2, 1, existing_agent)
    existing.bombs[existing.index(2, 1)].timer = 5
    wait = _features(existing_agent, existing, AgentAction(0, 0))
    move_safe = _features(existing_agent, existing, AgentAction(0, 1))
    assert move_safe["active_bomb_escape_margin"] > wait["active_bomb_escape_margin"]


def test_safe_successors_and_escape_route_diversity_distinguish_options():
    open_world, open_agent = _world(width=3, height=3, start=(1, 1), exit_position=(2, 2))
    open_features = _features(open_agent, open_world, AgentAction(0, 0, True))
    assert open_features["safe_successor_fraction"] > 0.0
    assert open_features["escape_route_diversity"] > 0.0

    corridor, corridor_agent = _world(
        width=3,
        height=3,
        start=(1, 1),
        exit_position=(2, 2),
        walls={(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
    )
    corridor_features = _features(corridor_agent, corridor, AgentAction(0, 0, True))
    assert corridor_features["escape_route_diversity"] < open_features["escape_route_diversity"]


def test_explosion_timer_zero_expires_before_next_successor_move():
    active, active_agent = _world(width=5, height=5, start=(2, 2), exit_position=(4, 4))
    active.add_bomb(1, 1, active_agent)
    active.add_explosion(3, 2, active.bombs[active.index(1, 1)])
    active.explosions[active.index(3, 2)].timer = 2

    expiring, expiring_agent = _world(width=5, height=5, start=(2, 2), exit_position=(4, 4))
    expiring.add_bomb(1, 1, expiring_agent)
    expiring.add_explosion(3, 2, expiring.bombs[expiring.index(1, 1)])
    expiring.explosions[expiring.index(3, 2)].timer = 0

    active_features = _features(active_agent, active, AgentAction(0, 0))
    expiring_features = _features(expiring_agent, expiring, AgentAction(0, 0))

    assert expiring_features["safe_successor_fraction"] > active_features["safe_successor_fraction"]


def _add_timed_explosion(world, agent, position, timer):
    world.add_bomb(0, 0, agent)
    source_bomb = world.bombs[world.index(0, 0)]
    world.add_explosion(*position, source_bomb)
    world.explosions[world.index(*position)].timer = timer


def test_explosion_threat_is_action_conditioned_and_matches_immediate_loss():
    world, agent = _world(start=(2, 2), exit_position=(4, 4))
    _add_timed_explosion(world, agent, (3, 2), timer=1)

    wait = _features(agent, world, AgentAction(0, 0))
    move_into_explosion = _features(agent, world, AgentAction(1, 0))

    assert wait["explosion_threat"] == 0.0
    assert move_into_explosion["explosion_threat"] == 1.0
    assert move_into_explosion["loss_next_update"] == 1.0


def test_explosion_threat_decreases_when_moving_away():
    world, agent = _world(start=(2, 2), exit_position=(4, 4))
    _add_timed_explosion(world, agent, (2, 2), timer=2)

    stay = _features(agent, world, AgentAction(0, 0))
    move_away = _features(agent, world, AgentAction(0, -1))

    assert stay["explosion_threat"] == 1.0
    assert move_away["explosion_threat"] == 0.0


def test_explosion_threat_ignores_explosion_that_expires_before_candidate_move():
    world, agent = _world(start=(2, 2), exit_position=(4, 4))
    _add_timed_explosion(world, agent, (3, 2), timer=0)

    move_into_expiring_cell = _features(agent, world, AgentAction(1, 0))

    assert move_into_expiring_cell["explosion_threat"] == 0.0
    assert move_into_expiring_cell["loss_next_update"] == 0.0


def test_explosion_threat_is_bounded_in_representative_variant_states():
    variant_monsters = (
        (1, ()),
        (3, ((4, 0),)),
        (5, ((4, 0), (0, 4))),
    )
    actions = [
        AgentAction(dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    ] + [AgentAction(0, 0, True)]

    for variant, monster_positions in variant_monsters:
        world, agent = _world(start=(2, 2), exit_position=(4, 4))
        for index, position in enumerate(monster_positions):
            world.add_monster(MonsterEntity(f"monster-{index}", "S", *position))
        _add_timed_explosion(world, agent, (3, 2), timer=2)

        for action in actions:
            feature = _features(agent, world, action)["explosion_threat"]
            assert 0.0 <= feature <= 1.0, f"variant {variant}, action {action}"


def test_explosion_threat_appears_in_dynamic_q_contribution_diagnostic(capsys):
    world, agent = _world(start=(2, 2), exit_position=(4, 4))
    _add_timed_explosion(world, agent, (2, 2), timer=2)
    sensed = SensedWorld.from_world(world)
    action = AgentAction(0, 0, True)
    features = agent.features(sensed, action)
    agent.weights = {"explosion_threat": -80.0}
    contributions = agent.q_contributions(features)

    agent._print_q_contributions(
        sensed,
        [(action, features, contributions, sum(contributions.values()))],
        action,
        "max Q",
    )

    output = capsys.readouterr().out
    assert "explosion_threat" in output
    assert "-80.000" in output


def test_monster_clearance_and_fuse_envelope_overlap_are_bounded_and_action_conditioned():
    world, agent = _world(width=7, height=7, start=(1, 1), exit_position=(6, 6))
    world.add_monster(MonsterEntity("target", "S", 4, 1))
    close = _features(agent, world, AgentAction(1, 0))
    far = _features(agent, world, AgentAction(0, 1))
    bomb = _features(agent, world, AgentAction(0, 0, True))

    assert close["monster_clearance"] < far["monster_clearance"]
    assert 0.0 < bomb["monster_fuse_envelope_overlap"] <= 1.0
    assert close["monster_fuse_envelope_overlap"] == 0.0


def test_urgency_scaled_progress_increases_as_time_runs_out():
    world, agent = _world(width=5, height=5, start=(1, 1), exit_position=(4, 4))
    agent.episode_time_limit = 100
    abundant = _features(agent, world, AgentAction(1, 1))
    world.time = 10
    urgent = _features(agent, world, AgentAction(1, 1))

    assert urgent["urgency_scaled_objective_progress"] > abundant["urgency_scaled_objective_progress"]
    assert -1.0 <= urgent["urgency_scaled_objective_progress"] <= 1.0


def test_all_recommended_features_remain_finite_and_bounded_on_variants_1_3_5(monkeypatch):
    from team01.project2 import training

    training._configure_runtime(False)
    for challenge_index in (0, 2, 4):
        challenge = training.progression()[challenge_index]
        game, agent = training._build_variant_game(
            challenge,
            17,
            {},
            training._create_qagent,
        )
        sensed = SensedWorld.from_world(game.world)
        for action in (
            AgentAction(0, 0),
            AgentAction(1, 0),
            AgentAction(0, 0, True),
        ):
            features = agent.features(sensed, action)
            assert set(features) == set(Q_FEATURE_NAMES)
            assert all(math.isfinite(value) for value in features.values())
            assert all(-1.0 <= value <= 1.0 for key, value in features.items() if key != "q_bias")
            assert features["q_bias"] == 1.0