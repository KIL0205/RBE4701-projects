import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "Bomberman"))

from team01.agent.actions import AgentAction
from team01.agent.behavior_tree import QLearningActionNode
from team01.agent.evaluation import q_feature_names
from team01.agent.q_learning import QAgent, QBTAgent
from team01.eval.project2_grading_eval import (
    GradingTrialResult,
    _default_checkpoint_path,
    _load_grading_agent,
    parse_args as parse_grading_args,
    run_evaluation,
)
from team01.eval.train_q_learning import EpisodeResult
from team01.project2 import training
from team01.project2.agent_backends import (
    AGENT_CHOICES,
    DEFAULT_AGENT_STATE_PATHS,
    DEFAULT_AGENT,
)


def _write_weights(path: Path, agent_type: str = "qbt") -> dict[str, float]:
    agent = QBTAgent("me", "C", 0, 0) if agent_type == "qbt" else QAgent("me", "C", 0, 0)
    agent.weights = {
        name: index / 10.0 for index, name in enumerate(sorted(q_feature_names()))
    }
    agent.save_weights(str(path))
    return dict(agent.weights)


@pytest.mark.parametrize("agent_type", AGENT_CHOICES)
def test_training_and_grading_parsers_accept_each_agent(agent_type):
    assert training.parse_args(["--agent", agent_type, "--no-display"]).agent_type == agent_type
    assert parse_grading_args(["--agent", agent_type, "--no-display"]).agent == agent_type


def test_training_and_grading_default_to_qbt_and_resolve_separate_paths():
    train_args = training.parse_args(["--no-display"])
    grading_args = parse_grading_args(["--no-display"])

    assert DEFAULT_AGENT == "qbt"
    assert train_args.agent_type == grading_args.agent == "qbt"
    assert train_args.weights == DEFAULT_AGENT_STATE_PATHS["qbt"]
    assert _default_checkpoint_path("qbt") == DEFAULT_AGENT_STATE_PATHS["qbt"]
    assert DEFAULT_AGENT_STATE_PATHS["q"] != DEFAULT_AGENT_STATE_PATHS["qbt"]


def test_grading_api_resolves_default_state_path_from_selected_backend(tmp_path):
    def trial_runner(variant, *, seed, **_kwargs):
        return GradingTrialResult(
            variant=variant,
            seed=seed,
            success=True,
            reason="WON",
            ticks=1,
            reward=0.0,
            bombs=0,
            max_abs_q=0.0,
            td_update_count=0,
        )

    result = run_evaluation(
        {},
        runs=1,
        display=False,
        output_path=tmp_path / "grading.json",
        trial_runner=trial_runner,
        agent_type="q",
    )

    assert result["weights_path"] == str(DEFAULT_AGENT_STATE_PATHS["q"].resolve())


@pytest.mark.parametrize(
    "parse,argv",
    [
        (training.parse_args, ["--agent", "invalid"]),
        (parse_grading_args, ["--agent", "invalid"]),
    ],
)
def test_agent_parsers_reject_invalid_values(parse, argv):
    with pytest.raises(SystemExit):
        parse(argv)


def test_training_factory_keeps_q_and_qbt_backends_distinct():
    q_agent = training._create_qagent()
    qbt_agent = training._create_qbt_agent()

    assert type(q_agent) is QAgent
    assert type(qbt_agent) is QBTAgent
    assert q_agent is not qbt_agent
    assert not hasattr(q_agent, "root")
    assert type(qbt_agent.root).__name__ == "QLearningRootController"
    assert qbt_agent.root.root.name == "root"

    def walk(node):
        yield node
        for child in getattr(node, "children", ()):
            yield from walk(child)

    assert any(
        isinstance(node, QLearningActionNode)
        for node in walk(qbt_agent.root.root)
    )


def test_q_and_qbt_have_independent_weights_and_persistence(tmp_path):
    q_agent = training._create_qagent()
    qbt_agent = training._create_qbt_agent()
    q_agent.weights = {"q_bias": 1.0}
    qbt_agent.weights = {"q_bias": 2.0}

    q_agent.weights["q_bias"] = 3.0
    assert qbt_agent.weights == {"q_bias": 2.0}
    qbt_agent.weights["q_bias"] = 4.0
    assert q_agent.weights == {"q_bias": 3.0}

    q_path = tmp_path / "q.json"
    qbt_path = tmp_path / "qbt.json"
    q_agent.save_weights(str(q_path))
    qbt_agent.save_weights(str(qbt_path))

    loaded_q = QAgent("me", "C", 0, 0)
    loaded_qbt = QBTAgent("me", "C", 0, 0)
    loaded_q.load_weights(str(q_path))
    loaded_qbt.load_weights(str(qbt_path))
    assert loaded_q.weights["q_bias"] == 3.0
    assert loaded_qbt.weights["q_bias"] == 4.0
    assert q_path != qbt_path
    q_state = json.loads(q_path.read_text(encoding="utf-8"))
    qbt_state = json.loads(qbt_path.read_text(encoding="utf-8"))
    assert q_state["feature_version"] == qbt_state["feature_version"] == 3
    assert q_state["features"] == qbt_state["features"]


@pytest.mark.parametrize(
    ("agent_type", "agent_class"),
    [("q", QAgent), ("qbt", QBTAgent)],
)
def test_parallel_q_worker_reconstructs_selected_class_and_defaults(
    monkeypatch,
    agent_type,
    agent_class,
):
    parameters = training._default_learning_parameters(agent_type)
    assert parameters == (
        agent_class("me", "C", 0, 0).alpha,
        agent_class("me", "C", 0, 0).gamma,
        agent_class("me", "C", 0, 0).epsilon,
    )

    constructed = []

    def game_factory(_challenge, _seed, weights, agent_factory):
        agent = agent_factory()
        agent.weights = dict(weights)
        constructed.append(agent)
        return SimpleNamespace(world=object()), agent

    def episode_runner(_game, agent, episode, *_args):
        agent.prev_features = {"q_bias": 1.0}
        agent.prev_action = AgentAction(1, 0)
        agent.finish_episode(agent.R_WIN)
        return EpisodeResult(
            episode=episode,
            outcome="WON",
            ticks=1,
            total_reward=agent.R_WIN,
            bombs_placed=0,
            non_finite_q_fallbacks=0,
            non_finite_q_values=0,
            mean_abs_td_error=agent.total_abs_td_error / agent.td_update_count,
            td_update_count=agent.td_update_count,
            max_abs_q=agent.max_abs_q,
            feature_activation_counts={},
        )

    monkeypatch.setattr(training, "_build_variant_game", game_factory)
    monkeypatch.setattr(training, "_run_parallel_episode", episode_runner)
    challenge = training.progression()[0]
    task = training._make_round_tasks(
        challenge=challenge,
        trial_numbers=(1,),
        batch_number=1,
        first_trial_index=1,
        trial_count=1,
        round_number=1,
        base_seed=123,
        starting_weights={name: 0.0 for name in q_feature_names()},
        guis=0,
        q_contributions_diagnostic=False,
        agent_type=agent_type,
    )[0]

    result = training._parallel_training_worker(task)

    assert result.error is None
    assert result.outcome == "WON"
    assert type(constructed[0]) is agent_class
    assert constructed[0].alpha == parameters[0]
    assert constructed[0].td_update_count == 1


def test_dqn_factory_uses_the_shared_q_reward_contract():
    from team01.agent.deep_q_learning import DeepQAgent

    dqn_agent = training._create_dqn_agent()

    assert type(dqn_agent) is DeepQAgent
    assert dqn_agent.R_NEAR_MONSTER == QAgent.R_NEAR_MONSTER


def test_dqn_grading_loads_a_frozen_checkpoint_without_mutating_it(tmp_path):
    from team01.agent.deep_q_learning import DeepQAgent

    checkpoint = tmp_path / "dqn.pt"
    DeepQAgent("me", "C", 0, 0).save_checkpoint(checkpoint)
    before = checkpoint.read_bytes()

    backend = _load_grading_agent("dqn", checkpoint)
    agent = backend.agent_factory()

    assert backend.agent_type == "dqn"
    assert agent.training is False
    assert agent.epsilon == 0.0
    assert agent.learning_enabled is False
    assert agent.collect_experience is False
    assert agent.policy_network.training is False
    assert len(agent.replay_buffer) == 0
    assert checkpoint.read_bytes() == before


def test_dqn_training_dispatch_saves_its_separate_checkpoint(tmp_path):
    def game_factory(_challenge, _seed, _weights, agent_factory):
        return SimpleNamespace(world=object()), agent_factory()

    def episode_runner(_game, _agent, episode, *_args):
        return EpisodeResult(
            episode=episode,
            outcome="WON",
            ticks=1,
            total_reward=1.0,
            bombs_placed=0,
            non_finite_q_fallbacks=0,
            non_finite_q_values=0,
            mean_abs_td_error=0.0,
            td_update_count=0,
            max_abs_q=0.0,
            feature_activation_counts={},
        )

    checkpoint = tmp_path / "training-dqn.pt"
    summary = training.run_progressive_training(
        trials=1,
        survive=1,
        seed=1234,
        weights_path=checkpoint,
        display=False,
        fresh=True,
        max_batches=1,
        workers=1,
        guis=0,
        eval_trials=1,
        history_path=None,
        game_factory=game_factory,
        episode_runner=episode_runner,
        agent_type="dqn",
        start_variant=2,
        dqn_updates_per_batch=0,
    )

    assert summary.completed_trials == 1
    assert summary.weights_saved
    assert checkpoint.is_file()


def test_dqn_parallel_rollout_uses_frozen_spawned_worker():
    from team01.agent.deep_q_learning import DeepQAgent

    agent = DeepQAgent("me", "C", 0, 0, device="cpu")
    results = training._run_dqn_rollout_batch(
        agent=agent,
        assignments=((training.progression()[0], 123),),
        workers=2,
        guis=0,
        batch_number=1,
        epsilon=0.0,
        time_limit=1,
    )

    assert len(results) == 1
    assert results[0].outcome == "TIMEOUT"
    assert results[0].error is None
    assert results[0].policy_unchanged
    assert results[0].gradients_absent


def test_qbt_training_updates_and_persists_its_own_state(tmp_path):
    path = tmp_path / "qbt.json"
    agent = training._create_qbt_agent()
    agent.prev_features = {"q_bias": 1.0}
    agent.prev_action = AgentAction(1, 0)

    assert agent.finish_episode(agent.R_WIN) is True
    assert agent.weights["q_bias"] > 0
    agent.save_weights(str(path))

    loaded = QBTAgent("me", "C", 0, 0)
    loaded.load_weights(str(path))
    assert loaded.weights == agent.weights
    assert json.loads(path.read_text(encoding="utf-8"))["feature_version"] == 3


def test_frozen_qbt_does_not_update_learned_state(tmp_path):
    path = tmp_path / "qbt.json"
    expected = _write_weights(path)
    before = path.read_bytes()
    backend = _load_grading_agent("qbt", path)
    agent = backend.agent_factory()
    agent.prev_features = {"q_bias": 1.0}
    agent.prev_action = AgentAction(1, 0)

    assert type(agent) is QBTAgent
    assert type(agent.root).__name__ == "QLearningRootController"
    assert agent.training is False
    assert agent.epsilon == 0.0
    assert agent.finish_episode(agent.R_WIN) is False
    assert agent.weights == expected
    assert path.read_bytes() == before


def test_qbt_training_uses_shared_curriculum_and_start_variant(tmp_path):
    instances = []
    trained_variants = []

    def game_factory(challenge, _seed, weights, agent_factory):
        trained_variants.append(challenge.number)
        agent = agent_factory()
        agent.weights = dict(weights)
        instances.append(agent)
        return SimpleNamespace(world=object()), agent

    def episode_runner(_game, agent, episode, *_args):
        agent.prev_features = {"q_bias": 1.0}
        agent.prev_action = AgentAction(1, 0)
        agent.finish_episode(agent.R_WIN)
        return EpisodeResult(
            episode=episode,
            outcome="WON",
            ticks=1,
            total_reward=agent.R_WIN,
            bombs_placed=0,
            non_finite_q_fallbacks=0,
            non_finite_q_values=0,
            mean_abs_td_error=agent.R_WIN,
            td_update_count=agent.td_update_count,
            max_abs_q=0.0,
            feature_activation_counts={},
        )

    weights_path = tmp_path / "training-qbt.json"
    summary = training.run_progressive_training(
        trials=1,
        survive=1,
        seed=1234,
        weights_path=weights_path,
        display=False,
        fresh=True,
        max_batches=1,
        workers=1,
        guis=0,
        eval_trials=1,
        history_path=None,
        game_factory=game_factory,
        episode_runner=episode_runner,
        agent_type="qbt",
        start_variant=2,
    )

    assert summary.completed_trials == 1
    assert instances
    assert trained_variants == [2]
    assert type(instances[0]) is QBTAgent
    assert instances[0].td_update_count == 1
    assert summary.current_stage.number == 3
    assert weights_path.is_file()


def test_qbt_uses_the_unchanged_stage_threshold_calculation():
    assert training._required_newest_wins(survive=15, trials=20, newest_episodes=10) == 8


def _frozen_qbt_trial_runner(variant, *, seed, weights, display, agent_factory):
    agent = agent_factory()
    before = dict(agent.weights)
    agent.prev_features = {"q_bias": 1.0}
    agent.prev_action = AgentAction(1, 0)
    assert type(agent) is QBTAgent
    assert agent.training is False
    assert agent.epsilon == 0.0
    assert agent.finish_episode(agent.R_WIN) is False
    assert agent.weights == before == weights
    assert display is False
    return GradingTrialResult(
        variant=variant,
        seed=seed,
        success=True,
        reason="WON",
        ticks=1,
        reward=1.0,
        bombs=0,
        max_abs_q=1.0,
        td_update_count=0,
    )


def _frozen_q_trial_runner(variant, *, seed, weights, display, agent_factory):
    agent = agent_factory()
    before = dict(agent.weights)
    agent.prev_features = {"q_bias": 1.0}
    agent.prev_action = AgentAction(1, 0)
    assert type(agent) is QAgent
    assert agent.training is False
    assert agent.epsilon == 0.0
    assert agent.finish_episode(agent.R_WIN) is False
    assert agent.weights == before == weights
    assert display is False
    return GradingTrialResult(
        variant=variant,
        seed=seed,
        success=True,
        reason="WON",
        ticks=1,
        reward=1.0,
        bombs=0,
        max_abs_q=1.0,
        td_update_count=0,
    )


def _frozen_dqn_trial_runner(variant, *, seed, weights, display, agent_factory):
    from team01.agent.deep_q_learning import DeepQAgent

    agent = agent_factory()
    assert type(agent) is DeepQAgent
    assert weights is None
    assert display is False
    assert agent.training is False
    assert agent.epsilon == 0.0
    assert agent.learning_enabled is False
    assert agent.collect_experience is False
    assert agent.policy_network.training is False
    assert len(agent.replay_buffer) == 0
    return GradingTrialResult(
        variant=variant,
        seed=seed,
        success=True,
        reason="WON",
        ticks=1,
        reward=1.0,
        bombs=0,
        max_abs_q=1.0,
        td_update_count=0,
    )


def _next_unique_seed(used):
    seed = len(used) + 100
    while seed in used:
        seed += 1
    used.add(seed)
    return seed


def test_qbt_grading_is_frozen_and_parallel_workers_reconstruct_backend(tmp_path):
    checkpoint = tmp_path / "qbt.json"
    weights = _write_weights(checkpoint)
    before = checkpoint.read_bytes()
    result = run_evaluation(
        weights,
        runs=1,
        display=False,
        weights_path=checkpoint,
        output_path=tmp_path / "results.json",
        trial_runner=_frozen_qbt_trial_runner,
        agent_factory=_load_grading_agent("qbt", checkpoint).agent_factory,
        agent_type="qbt",
        workers=2,
        seed_generator=_next_unique_seed,
    )

    assert result["agent_type"] == "qbt"
    assert result["overall_trials"] == 5
    assert result["overall_wins"] == 5
    assert checkpoint.read_bytes() == before


def test_q_grading_parallel_workers_reconstruct_basic_q(tmp_path):
    checkpoint = tmp_path / "q.json"
    weights = _write_weights(checkpoint, "q")
    before = checkpoint.read_bytes()
    result = run_evaluation(
        weights,
        runs=1,
        display=False,
        weights_path=checkpoint,
        output_path=tmp_path / "q-results.json",
        trial_runner=_frozen_q_trial_runner,
        agent_factory=_load_grading_agent("q", checkpoint).agent_factory,
        agent_type="q",
        workers=2,
        seed_generator=_next_unique_seed,
    )

    assert result["agent_type"] == "q"
    assert result["overall_trials"] == 5
    assert result["overall_wins"] == 5
    assert checkpoint.read_bytes() == before


def test_dqn_grading_parallel_workers_reconstruct_frozen_network(tmp_path):
    from team01.agent.deep_q_learning import DeepQAgent

    checkpoint = tmp_path / "dqn.pt"
    DeepQAgent("me", "C", 0, 0).save_checkpoint(checkpoint)
    before = checkpoint.read_bytes()
    backend = _load_grading_agent("dqn", checkpoint)
    result = run_evaluation(
        None,
        runs=1,
        display=False,
        weights_path=checkpoint,
        output_path=tmp_path / "dqn-results.json",
        trial_runner=_frozen_dqn_trial_runner,
        agent_factory=backend.agent_factory,
        agent_type="dqn",
        workers=2,
        canonical_snapshot=backend.canonical_snapshot,
        seed_generator=_next_unique_seed,
    )

    assert result["agent_type"] == "dqn"
    assert result["overall_trials"] == 5
    assert result["overall_wins"] == 5
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("agent_type", ("q", "qbt", "dqn"))
def test_serial_grading_api_constructs_selected_backend_without_factory(
    agent_type,
    tmp_path,
):
    if agent_type == "dqn":
        from team01.agent.deep_q_learning import DeepQAgent

        checkpoint = tmp_path / "dqn.pt"
        DeepQAgent("me", "C", 0, 0).save_checkpoint(checkpoint)
        weights = None
        trial_runner = _frozen_dqn_trial_runner
    else:
        checkpoint = tmp_path / f"{agent_type}.json"
        weights = _write_weights(checkpoint, agent_type)
        trial_runner = (
            _frozen_qbt_trial_runner
            if agent_type == "qbt"
            else _frozen_q_trial_runner
        )

    before = checkpoint.read_bytes()
    result = run_evaluation(
        weights,
        runs=1,
        display=False,
        weights_path=checkpoint,
        output_path=tmp_path / f"{agent_type}-serial-results.json",
        trial_runner=trial_runner,
        agent_type=agent_type,
        workers=1,
        seed_generator=_next_unique_seed,
    )

    assert result["agent_type"] == agent_type
    assert result["overall_trials"] == 5
    assert result["overall_wins"] == 5
    assert checkpoint.read_bytes() == before
