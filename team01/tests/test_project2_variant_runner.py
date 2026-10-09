import hashlib
import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "Bomberman"))

from team01.agent.q_learning import QAgent, QBTAgent
from team01.project2 import variant_runner
from team01.project2.agent_backends import (
    AGENT_CHOICES,
    DEFAULT_AGENT,
    DEFAULT_AGENT_STATE_PATHS,
)


VARIANT_MODULES = tuple(
    __import__(f"team01.project2.variant{number}", fromlist=["main"])
    for number in range(1, 6)
)
VARIANT_MODULE_CASES = tuple(enumerate(VARIANT_MODULES, start=1))


class _StubGame:
    def __init__(self, variant_number):
        self.variant_number = variant_number
        self.scenario_first_random = random.random()
        self.agent = None
        self.started = False

    def add_character(self, agent):
        self.agent = agent

    def go(self, _wait):
        self.started = True
        self.first_game_random = random.random()


def _loader_consumes_random(agent_type, weights_path):
    random.random()
    return variant_runner._load_backend(agent_type, weights_path)


def _save_agent_state(agent_type, path):
    if agent_type == "dqn":
        from team01.agent.deep_q_learning import DeepQAgent

        DeepQAgent("me", "C", 0, 0).save_checkpoint(path)
        return

    agent_class = QBTAgent if agent_type == "qbt" else QAgent
    agent_class("me", "C", 0, 0).save_weights(str(path))


@pytest.mark.parametrize("variant_number,module", VARIANT_MODULE_CASES)
def test_each_variant_cli_defaults_to_qbt_and_accepts_supported_agents(
    variant_number, module
):
    assert DEFAULT_AGENT == "qbt"
    assert module.main.__module__ == f"team01.project2.variant{variant_number}"
    assert variant_runner.parse_variant_args(variant_number, []).agent == "qbt"
    for agent_type in AGENT_CHOICES:
        assert (
            variant_runner.parse_variant_args(
                variant_number, ["--agent", agent_type]
            ).agent
            == agent_type
        )


@pytest.mark.parametrize("variant_number", range(1, 6))
def test_variant_cli_rejects_unknown_agents(variant_number):
    with pytest.raises(SystemExit):
        variant_runner.parse_variant_args(variant_number, ["--agent", "invalid"])


def test_explicit_seed_is_preserved_and_printed(tmp_path, capsys):
    state_path = tmp_path / "state"
    _save_agent_state("q", state_path)
    saved_state = state_path.read_bytes()
    games = []

    result = variant_runner.run_variant(
        1,
        agent_type="q",
        seed=12345,
        weights_path=state_path,
        display=False,
        game_factory=lambda number: games.append(_StubGame(number)) or games[-1],
    )

    assert result["seed"] == 12345
    output = capsys.readouterr().out
    assert "Variant: V1 - Alone in the world" in output
    assert "Agent: Q-Learning" in output
    assert "Seed: 12345" in output
    assert games[0].started
    assert state_path.read_bytes() == saved_state


def test_automatic_seed_uses_project1_random_0_to_1000_pattern(monkeypatch):
    monkeypatch.setattr(random, "randint", lambda lower, upper: (lower, upper)[1])

    assert variant_runner.resolve_seed(None) == 1000
    assert variant_runner.resolve_seed(4) == 4


@pytest.mark.parametrize("agent_type", AGENT_CHOICES)
def test_runner_resolves_default_state_path_for_selected_agent(
    agent_type, tmp_path, monkeypatch
):
    state_path = tmp_path / f"{agent_type}.state"
    _save_agent_state(agent_type, state_path)
    monkeypatch.setitem(DEFAULT_AGENT_STATE_PATHS, agent_type, state_path)

    result = variant_runner.run_variant(
        1,
        agent_type=agent_type,
        seed=8,
        display=False,
        game_factory=_StubGame,
    )

    assert result["weights_path"] == state_path


@pytest.mark.parametrize("agent_type", AGENT_CHOICES)
def test_same_seed_builds_same_scenario_independent_of_selected_agent(
    agent_type, tmp_path, capsys
):
    state_path = tmp_path / "state"
    _save_agent_state(agent_type, state_path)
    games = []
    result = variant_runner.run_variant(
        4,
        agent_type=agent_type,
        seed=24680,
        weights_path=state_path,
        display=False,
        game_factory=lambda number: games.append(_StubGame(number)) or games[-1],
        backend_loader=_loader_consumes_random,
    )

    assert result["seed"] == 24680
    assert games[0].scenario_first_random == random.Random(24680).random()
    assert games[0].first_game_random == random.Random(24680).random()
    assert games[0].variant_number == 4
    assert variant_runner.VARIANTS[4].monsters == (
        variant_runner.VariantMonster("smart", "selfpreserving", "S", 3, 9, 1),
    )
    assert "Agent:" in capsys.readouterr().out


@pytest.mark.parametrize("agent_type", ("q", "qbt"))
def test_q_and_qbt_load_frozen_state_without_writing_it(agent_type, tmp_path):
    state_path = tmp_path / f"{agent_type}.json"
    agent = QBTAgent("me", "C", 0, 0) if agent_type == "qbt" else QAgent("me", "C", 0, 0)
    agent.save_weights(str(state_path))
    before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    games = []

    result = variant_runner.run_variant(
        2,
        agent_type=agent_type,
        seed=8765,
        weights_path=state_path,
        display=False,
        game_factory=lambda number: games.append(_StubGame(number)) or games[-1],
    )

    loaded = games[0].agent
    assert result["agent_type"] == agent_type
    assert loaded.training is False
    assert loaded.epsilon == 0.0
    assert (type(loaded) is QBTAgent) is (agent_type == "qbt")
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == before


def test_dqn_loads_frozen_checkpoint_without_writing_it(tmp_path):
    from team01.agent.deep_q_learning import DeepQAgent

    checkpoint = tmp_path / "dqn.pt"
    DeepQAgent("me", "C", 0, 0).save_checkpoint(checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    games = []

    result = variant_runner.run_variant(
        1,
        agent_type="dqn",
        seed=9876,
        weights_path=checkpoint,
        display=False,
        game_factory=lambda number: games.append(_StubGame(number)) or games[-1],
    )

    agent = games[0].agent
    assert result["agent_type"] == "dqn"
    assert agent.training is False
    assert agent.epsilon == 0.0
    assert agent.learning_enabled is False
    assert agent.collect_experience is False
    assert agent.policy_network.training is False
    assert len(agent.replay_buffer) == 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("variant_number", range(1, 6))
def test_each_variant_defaults_to_its_canonical_state_file(variant_number):
    args = variant_runner.parse_variant_args(variant_number, [])

    assert args.weights is None
    assert DEFAULT_AGENT_STATE_PATHS["q"] != DEFAULT_AGENT_STATE_PATHS["qbt"]
    assert DEFAULT_AGENT_STATE_PATHS["dqn"].name == "dqn_checkpoint.pt"


def test_missing_agent_state_is_reported_before_starting_game(tmp_path):
    with pytest.raises(FileNotFoundError, match="learned state not found"):
        variant_runner.run_variant(
            1,
            seed=1,
            weights_path=tmp_path / "missing.state",
            display=False,
            game_factory=lambda _number: pytest.fail("game must not start"),
        )
