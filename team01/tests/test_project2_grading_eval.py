import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "Bomberman"))

from team01.agent.actions import AgentAction
from team01.agent.evaluation import q_feature_names
from team01.agent.q_learning import QAgent
from team01.eval.project2_grading_eval import (
    GradingTrialResult,
    evaluate_variant,
    generate_seed,
    is_solved,
    load_frozen_agent,
    parse_args,
    run_evaluation,
    run_trial,
)
from team01.eval.train_q_learning import EpisodeResult


def _save_known_weights(path: Path) -> dict[str, float]:
    agent = QAgent("me", "C", 0, 0)
    agent.weights = {
        name: index / 10.0 for index, name in enumerate(sorted(q_feature_names()))
    }
    agent.save_weights(str(path))
    return dict(agent.weights)


def _episode_result(outcome: str = "WON") -> EpisodeResult:
    return EpisodeResult(
        episode=1,
        outcome=outcome,
        ticks=12,
        total_reward=12.5 if outcome == "WON" else -1.0,
        bombs_placed=1,
        non_finite_q_fallbacks=0,
        non_finite_q_values=0,
        mean_abs_td_error=0.0,
        td_update_count=0,
        max_abs_q=3.25,
        feature_activation_counts={},
    )


def _fake_builder(created_agents=None):
    def build(_variant, _seed, weights):
        agent = QAgent("me", "C", 0, 0)
        agent.weights = dict(weights)
        agent.set_learning(no_training=True, epsilon=0.0)
        if created_agents is not None:
            created_agents.append(agent)
        return SimpleNamespace(world=object()), agent

    return build


def _fake_episode_runner(outcome: str = "WON"):
    def run(_world, _agent, *_args, **kwargs):
        assert kwargs["heartbeat_interval"] == 0
        assert kwargs["on_tick"] is None
        return _episode_result(outcome)

    return run


def test_loads_known_learned_weights_and_freezes_agent(tmp_path):
    checkpoint = tmp_path / "learned.json"
    expected = _save_known_weights(checkpoint)

    agent = load_frozen_agent(checkpoint)

    assert agent.weights == expected
    assert agent.training is False
    assert agent.epsilon == 0.0
    assert agent.td_update_count == 0


def test_missing_weights_fail_without_fresh_policy(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="will not fall back"):
        load_frozen_agent(missing)


def test_trial_uses_loaded_weights_without_training_or_file_mutation(tmp_path):
    checkpoint = tmp_path / "learned.json"
    canonical = _save_known_weights(checkpoint)
    before = checkpoint.read_bytes()
    created_agents = []

    result = run_trial(
        5,
        seed=123456,
        weights=canonical,
        display=False,
        game_builder=_fake_builder(created_agents),
        episode_runner=_fake_episode_runner("TIMEOUT"),
    )

    assert result.variant == 5
    assert result.seed == 123456
    assert result.success is False
    assert result.reason == "TIMEOUT"
    assert result.td_update_count == 0
    assert created_agents[0].weights == canonical
    assert created_agents[0].weights == dict(canonical)
    assert checkpoint.read_bytes() == before


def test_each_trial_gets_clean_agent_state_with_same_canonical_weights():
    canonical = {name: 1.0 for name in q_feature_names()}
    created_agents = []

    first = run_trial(
        1,
        seed=1,
        weights=canonical,
        game_builder=_fake_builder(created_agents),
        episode_runner=_fake_episode_runner("WON"),
    )
    second = run_trial(
        1,
        seed=2,
        weights=canonical,
        game_builder=_fake_builder(created_agents),
        episode_runner=_fake_episode_runner("LOST"),
    )

    assert first.seed != second.seed
    assert len(created_agents) == 2
    assert created_agents[0] is not created_agents[1]
    for agent in created_agents:
        assert agent.weights == canonical
        assert agent.training is False
        assert agent.epsilon == 0.0
        assert agent.prev_model is None
        assert agent.prev_action is None
        assert agent.prev_features is None
        assert agent.td_update_count == 0
        assert agent.episode_reward == 0.0
        assert agent.bombs_placed == 0


def test_frozen_mode_never_enters_epsilon_exploration_or_terminal_td(monkeypatch):
    agent = QAgent("me", "C", 0, 0)
    agent.weights = {"q_bias": 1.0}
    agent.set_learning(no_training=True, epsilon=0.0)
    action = AgentAction(1, 0)

    monkeypatch.setattr(
        agent,
        "max_q_value",
        lambda _world, actions: (1.0, [(actions[0], 1.0)]),
    )
    monkeypatch.setattr(
        random,
        "random",
        lambda: (_ for _ in ()).throw(AssertionError("epsilon branch called")),
    )
    selected = agent.choose_action(None, [action])
    assert selected == action

    agent.prev_features = {"q_bias": 1.0}
    before = dict(agent.weights)
    assert agent.finish_episode(agent.R_WIN) is False
    assert agent.weights == before
    assert agent.td_update_count == 0


def test_solved_threshold_is_inclusive():
    assert is_solved(25, 50) is True
    assert is_solved(24, 50) is False
    with pytest.raises(ValueError):
        is_solved(0, 0)


def test_generate_seeds_are_unique_and_unsigned_32_bit():
    used = set()
    seeds = [generate_seed(used) for _ in range(500)]
    assert len(set(seeds)) == len(seeds) == len(used)
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_trial_recording_and_first_failure():
    outcomes = iter(("WON", "WON", "LOST", "TIMEOUT"))

    def trial_runner(variant, *, seed, **_kwargs):
        outcome = next(outcomes)
        return GradingTrialResult(
            variant=variant,
            seed=seed,
            success=outcome == "WON",
            reason=outcome,
            ticks=seed % 100,
            reward=1.0,
            bombs=0,
            max_abs_q=2.0,
            td_update_count=0,
        )

    summary = evaluate_variant(
        3,
        4,
        set(),
        {"q_bias": 1.0},
        display=False,
        trial_runner=trial_runner,
    )

    assert [trial["run"] for trial in summary["trials"]] == [1, 2, 3, 4]
    assert all(trial["variant"] == 3 for trial in summary["trials"])
    assert all("seed" in trial for trial in summary["trials"])
    assert summary["first_failure"] == {
        "run": 3,
        "seed": summary["trials"][2]["seed"],
        "reason": "LOST",
        "ticks": summary["trials"][2]["ticks"],
    }
    assert summary["failure_reasons"] == {"LOST": 1, "TIMEOUT": 1}


def test_scoring_and_json_results_use_project1_rubric(tmp_path):
    output = tmp_path / "project2-results.json"

    def trial_runner(variant, *, seed, **_kwargs):
        failed = variant in {3, 5}
        reason = "LOST" if failed else "WON"
        return GradingTrialResult(
            variant=variant,
            seed=seed,
            success=not failed,
            reason=reason,
            ticks=8,
            reward=1.0,
            bombs=0,
            max_abs_q=1.0,
            td_update_count=0,
        )

    result = run_evaluation(
        {"q_bias": 2.0},
        runs=4,
        display=False,
        weights_path=tmp_path / "weights.json",
        output_path=output,
        trial_runner=trial_runner,
    )

    assert result["total_points"] == 60
    assert result["maximum_points"] == 120
    assert result["variants_solved"] == 3
    assert result["overall_wins"] == 12
    assert result["overall_trials"] == 20
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["learning_enabled"] is False
    assert saved["epsilon"] == 0.0
    assert saved["variants"]["3"]["solved"] is False
    assert saved["variants"]["3"]["first_failure"]["run"] == 1


def test_replay_cli_requires_variant_and_seed_together():
    args = parse_args(["--variant", "5", "--seed", "123", "--no-display"])
    assert args.variant == 5
    assert args.seed == 123
    assert args.display is False
    with pytest.raises(SystemExit):
        parse_args(["--variant", "5"])
    with pytest.raises(SystemExit):
        parse_args(["--seed", "123"])
