"""Deep Q-Learning agent and checkpoint handling for Project 2."""

from __future__ import annotations

import math
import os
import pickle
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from Bomberman.entity import CharacterEntity
from Bomberman.events import Event
from Bomberman.sensed_world import SensedWorld

from .actions import AgentAction
from .dqn_network import DQNNetwork, ReplayBuffer, Transition
from .dqn_features import (
    DQN_FEATURE_VERSION,
    dqn_feature_names,
    dqn_feature_vector,
)
from .q_learning import QAgent
from .safety import legal_candidate_actions
from .world_model import WorldModel


DQN_LEARNING_RATE = 1e-4
DQN_REPLAY_CAPACITY = 50_000
DQN_REPLAY_WARMUP = 1_000
DQN_BATCH_SIZE = 64
DQN_MAX_GRAD_NORM = 10.0
DQN_TARGET_SYNC_INTERVAL = 500
DQN_CHECKPOINT_VERSION = 1
DEFAULT_DQN_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "project2" / "dqn_checkpoint.pt"
)


def _copy_to_cpu(value):
    """Clone tensors while copying nested optimizer state for snapshots."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return value


def _snapshot_values_equal(first, second) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _snapshot_values_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(
            _snapshot_values_equal(left, right)
            for left, right in zip(first, second)
        )
    return first == second


def checkpoint_snapshots_equal(first: dict, second: dict) -> bool:
    """Compare immutable CPU checkpoint snapshots, including tensor state."""
    return _snapshot_values_equal(first, second)


def _state_dict_is_finite(state_dict: dict) -> bool:
    return all(
        torch.isfinite(value).all().item()
        for value in state_dict.values()
        if isinstance(value, torch.Tensor)
    )


def _nested_values_are_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_nested_values_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_values_are_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


@dataclass(frozen=True)
class DQNTrainingResult:
    """Diagnostics from one completed replay optimizer step."""

    loss: float
    mean_abs_td_error: float
    max_abs_td_error: float
    mean_q: float
    max_abs_q: float
    gradient_norm: float
    target_synced: bool


class DeepQAgent(CharacterEntity):
    """Choose actions with a DQN and expose training state to the trainer.

    The policy scores action-conditioned features; a separate frozen target
    supplies Bellman bootstrap values. The curriculum trainer owns replay and
    optimizer updates, while parallel rollout agents only collect experience.
    """

    R_WIN = 1000.0
    R_LOSE = -1000.0
    R_COST_OF_LIVING = -1.0
    R_NEAR_MONSTER = QAgent.R_NEAR_MONSTER
    R_STEP = QAgent.R_STEP
    R_KILL_MONSTER = QAgent.R_KILL_MONSTER
    R_BREAK_WALL = QAgent.R_BREAK_WALL
    R_PLACE_BOMB = QAgent.R_PLACE_BOMB

    def __init__(
        self,
        name,
        avatar,
        x,
        y,
        epsilon: float = 0.05,
        device: Optional[torch.device | str] = None,
        gamma: float = 0.9,
        learning_rate: float = DQN_LEARNING_RATE,
        replay_capacity: int = DQN_REPLAY_CAPACITY,
        replay_warmup: int = DQN_REPLAY_WARMUP,
        batch_size: int = DQN_BATCH_SIZE,
        max_grad_norm: float = DQN_MAX_GRAD_NORM,
        target_sync_interval: int = DQN_TARGET_SYNC_INTERVAL,
    ):
        super().__init__(name, avatar, x, y)

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if replay_warmup < batch_size:
            raise ValueError("replay_warmup must be at least batch_size")
        if max_grad_norm <= 0 or not math.isfinite(max_grad_norm):
            raise ValueError("max_grad_norm must be positive and finite")
        if target_sync_interval < 1:
            raise ValueError("target_sync_interval must be at least 1")
        if learning_rate <= 0 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.feature_names = dqn_feature_names()
        self.policy_network = DQNNetwork(len(self.feature_names)).to(self.device)
        self.target_network = DQNNetwork(len(self.feature_names)).to(self.device)
        # Only the policy is optimized; the target stays frozen between scheduled syncs.
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.gamma = gamma
        self.learning_rate = learning_rate
        # Replay belongs to the canonical learner and receives worker-produced transitions.
        self.replay_buffer = ReplayBuffer(
            replay_capacity,
            feature_size=len(self.feature_names),
        )
        self.replay_warmup = replay_warmup
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_sync_interval = target_sync_interval
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
        )

        self.epsilon = epsilon
        self.training = True
        self.learning_enabled = True
        self.collect_experience = True
        self.episode_time_limit = None
        self._q_feature_context = None
        self.pending_features: Optional[tuple[float, ...]] = None
        self.pending_model: Optional[WorldModel] = None
        self.pending_action: Optional[AgentAction] = None
        self.episode_transitions: list[Transition] = []
        self.episode_action_count = 0
        self.episode_reward = 0.0
        self._episode_finished = False
        self.last_candidate_q_values: list[tuple[AgentAction, float]] = []
        self.max_abs_q = 0.0
        self.q_evaluation_count = 0
        self.non_finite_q_values = 0
        self.bombs_placed = 0
        self.optimizer_steps = 0
        self.target_sync_count = 0
        self._central_update_in_progress = False
        self.episodes = 0
        self.last_loss: Optional[float] = None
        self.last_mean_abs_td_error: Optional[float] = None
        self.last_max_abs_td_error: Optional[float] = None
        self.last_mean_q: Optional[float] = None
        self.last_max_abs_q: Optional[float] = None
        self.last_gradient_norm: Optional[float] = None

    def set_learning_rate(self, learning_rate: float) -> None:
        """Override the Adam learning rate while keeping all optimizer state."""
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or learning_rate <= 0
            or not math.isfinite(learning_rate)
        ):
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _hyperparameters(self) -> dict:
        """Describe the current training configuration and network shape."""
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "replay_capacity": self.replay_buffer.capacity,
            "replay_warmup": self.replay_warmup,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "target_sync_interval": self.target_sync_interval,
            "input_size": len(self.feature_names),
            "hidden_sizes": (
                self.policy_network.layer1.out_features,
                self.policy_network.layer2.out_features,
            ),
            "output_size": self.policy_network.output.out_features,
        }

    def get_checkpoint_snapshot(self) -> dict:
        """Return cloned CPU trainable state for frozen-policy verification."""
        return {
            "policy_state_dict": _copy_to_cpu(self.policy_network.state_dict()),
            "target_state_dict": _copy_to_cpu(self.target_network.state_dict()),
            "optimizer_state_dict": _copy_to_cpu(self.optimizer.state_dict()),
            "optimizer_steps": self.optimizer_steps,
            "target_sync_count": self.target_sync_count,
            "episodes": self.episodes,
        }

    def save_checkpoint(self, path: Path | str = DEFAULT_DQN_CHECKPOINT_PATH) -> None:
        """Atomically save policy, target, Adam state, and counters, not replay."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "checkpoint_version": DQN_CHECKPOINT_VERSION,
            "dqn_feature_version": DQN_FEATURE_VERSION,
            "dqn_feature_names": self.feature_names,
            "input_size": len(self.feature_names),
            "policy_state_dict": self.policy_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "target_sync_count": self.target_sync_count,
            "episodes": self.episodes,
            "hyperparameters": self._hyperparameters(),
        }

        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_path.parent,
        )
        os.close(temporary_descriptor)
        temporary_path = Path(temporary_name)
        try:
            # Keep the temporary file beside its destination so replacement is atomic.
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load_checkpoint(self, path: Path | str = DEFAULT_DQN_CHECKPOINT_PATH) -> None:
        """Validate then restore learned state, leaving replay and runtime mode fresh.

        Candidate networks and optimizer state are checked before replacing the
        live learner, so an invalid checkpoint cannot reset a working policy.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DQN checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            raise ValueError(f"Could not read DQN checkpoint: {error}") from error

        if not isinstance(checkpoint, dict):
            raise ValueError("DQN checkpoint must be a dictionary")

        required_fields = {
            "checkpoint_version",
            "dqn_feature_version",
            "dqn_feature_names",
            "input_size",
            "policy_state_dict",
            "target_state_dict",
            "optimizer_state_dict",
            "optimizer_steps",
            "target_sync_count",
            "episodes",
            "hyperparameters",
        }
        missing_fields = required_fields - checkpoint.keys()
        if missing_fields:
            schema_fields = {
                "dqn_feature_version",
                "dqn_feature_names",
                "input_size",
            }
            if missing_fields & schema_fields:
                raise ValueError(
                    "DQN checkpoint feature schema mismatch: missing "
                    + ", ".join(sorted(missing_fields & schema_fields))
                )
            raise ValueError(
                "DQN checkpoint is missing required field(s): "
                + ", ".join(sorted(missing_fields))
            )

        checkpoint_version = checkpoint["checkpoint_version"]
        if (
            not isinstance(checkpoint_version, int)
            or isinstance(checkpoint_version, bool)
            or checkpoint_version != DQN_CHECKPOINT_VERSION
        ):
            raise ValueError(
                f"Unsupported DQN checkpoint version "
                f"{checkpoint_version}; expected "
                f"{DQN_CHECKPOINT_VERSION}"
            )
        feature_version = checkpoint["dqn_feature_version"]
        if (
            not isinstance(feature_version, int)
            or isinstance(feature_version, bool)
            or feature_version != DQN_FEATURE_VERSION
        ):
            raise ValueError(
                f"DQN checkpoint feature schema mismatch: version "
                f"{feature_version} does not match current version "
                f"{DQN_FEATURE_VERSION}"
            )
        saved_names = checkpoint["dqn_feature_names"]
        if not isinstance(saved_names, (list, tuple)):
            raise ValueError("DQN checkpoint dqn_feature_names must be a list or tuple")
        if not all(isinstance(name, str) for name in saved_names):
            raise ValueError("DQN checkpoint dqn_feature_names must contain only strings")
        if tuple(saved_names) != self.feature_names:
            raise ValueError(
                "DQN checkpoint feature schema mismatch: feature names or "
                "ordering do not match the current schema"
            )
        saved_input_size = checkpoint["input_size"]
        if (
            not isinstance(saved_input_size, int)
            or isinstance(saved_input_size, bool)
            or saved_input_size != len(self.feature_names)
        ):
            raise ValueError(
                "DQN checkpoint feature schema mismatch: input_size "
                f"{saved_input_size!r} does not match {len(self.feature_names)}"
            )

        hyperparameters = checkpoint["hyperparameters"]
        if not isinstance(hyperparameters, dict):
            raise ValueError("DQN checkpoint hyperparameters must be a dictionary")
        expected_hyperparameters = {
            "learning_rate",
            "gamma",
            "replay_capacity",
            "replay_warmup",
            "batch_size",
            "max_grad_norm",
            "target_sync_interval",
            "input_size",
            "hidden_sizes",
            "output_size",
        }
        missing_hyperparameters = expected_hyperparameters - hyperparameters.keys()
        if missing_hyperparameters:
            raise ValueError(
                "DQN checkpoint hyperparameters are missing: "
                + ", ".join(sorted(missing_hyperparameters))
            )
        expected_architecture = {
            "input_size": len(self.feature_names),
            "hidden_sizes": (
                self.policy_network.layer1.out_features,
                self.policy_network.layer2.out_features,
            ),
            "output_size": self.policy_network.output.out_features,
        }
        for name, expected in expected_architecture.items():
            saved_value = hyperparameters[name]
            if name == "hidden_sizes" and isinstance(saved_value, (list, tuple)):
                if not all(
                    isinstance(size, int) and not isinstance(size, bool)
                    for size in saved_value
                ):
                    raise ValueError("DQN checkpoint hidden_sizes must contain integers")
                saved_value = tuple(saved_value)
            elif name != "hidden_sizes" and (
                not isinstance(saved_value, int) or isinstance(saved_value, bool)
            ):
                raise ValueError(
                    f"DQN checkpoint architecture {name} must be an integer"
                )
            if saved_value != expected:
                raise ValueError(
                    f"DQN checkpoint architecture {name}={saved_value!r} "
                    f"does not match current value {expected!r}"
                )

        learning_rate = hyperparameters["learning_rate"]
        gamma = hyperparameters["gamma"]
        replay_capacity = hyperparameters["replay_capacity"]
        replay_warmup = hyperparameters["replay_warmup"]
        batch_size = hyperparameters["batch_size"]
        max_grad_norm = hyperparameters["max_grad_norm"]
        target_sync_interval = hyperparameters["target_sync_interval"]
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
            or not math.isfinite(learning_rate)
        ):
            raise ValueError("DQN checkpoint learning_rate must be positive and finite")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or not math.isfinite(gamma):
            raise ValueError("DQN checkpoint gamma must be finite")
        if not isinstance(replay_capacity, int) or isinstance(replay_capacity, bool) or replay_capacity < 1:
            raise ValueError("DQN checkpoint replay_capacity must be a positive integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("DQN checkpoint batch_size must be a positive integer")
        if not isinstance(replay_warmup, int) or isinstance(replay_warmup, bool) or replay_warmup < batch_size:
            raise ValueError("DQN checkpoint replay_warmup must be at least batch_size")
        if (
            not isinstance(max_grad_norm, (int, float))
            or isinstance(max_grad_norm, bool)
            or max_grad_norm <= 0
            or not math.isfinite(max_grad_norm)
        ):
            raise ValueError("DQN checkpoint max_grad_norm must be positive and finite")
        if (
            not isinstance(target_sync_interval, int)
            or isinstance(target_sync_interval, bool)
            or target_sync_interval < 1
        ):
            raise ValueError("DQN checkpoint target_sync_interval must be a positive integer")

        counters = {}
        for name in ("optimizer_steps", "target_sync_count", "episodes"):
            value = checkpoint[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"DQN checkpoint {name} must be a nonnegative integer")
            counters[name] = value

        saved_policy = checkpoint["policy_state_dict"]
        saved_target = checkpoint["target_state_dict"]
        if not isinstance(saved_policy, dict) or not isinstance(saved_target, dict):
            raise ValueError("DQN checkpoint network state must be dictionaries")
        if not _state_dict_is_finite(saved_policy):
            raise ValueError("DQN checkpoint policy parameters contain non-finite values")
        if not _state_dict_is_finite(saved_target):
            raise ValueError("DQN checkpoint target parameters contain non-finite values")

        candidate_policy = DQNNetwork(len(self.feature_names)).to(self.device)
        candidate_target = DQNNetwork(len(self.feature_names)).to(self.device)
        try:
            candidate_policy.load_state_dict(saved_policy, strict=True)
            candidate_target.load_state_dict(saved_target, strict=True)
        except (RuntimeError, KeyError, TypeError) as error:
            raise ValueError(f"DQN checkpoint network architecture is incompatible: {error}") from error

        optimizer_state = checkpoint["optimizer_state_dict"]
        if not isinstance(optimizer_state, dict):
            raise ValueError("DQN checkpoint optimizer_state_dict must be a dictionary")
        if not {"state", "param_groups"} <= optimizer_state.keys():
            raise ValueError(
                "DQN checkpoint optimizer_state_dict is missing state or param_groups"
            )
        saved_groups = optimizer_state.get("param_groups")
        if (
            not isinstance(saved_groups, list)
            or len(saved_groups) != 1
            or not isinstance(saved_groups[0], dict)
            or len(saved_groups[0].get("params", ()))
            != sum(1 for _ in candidate_policy.parameters())
        ):
            raise ValueError("DQN checkpoint optimizer parameter groups are incompatible")

        candidate_optimizer = torch.optim.Adam(
            candidate_policy.parameters(),
            lr=float(learning_rate),
        )
        try:
            candidate_optimizer.load_state_dict(optimizer_state)
        except (KeyError, RuntimeError, ValueError, TypeError) as error:
            raise ValueError(f"DQN checkpoint optimizer state is incompatible: {error}") from error
        if not _nested_values_are_finite(candidate_optimizer.state_dict()):
            raise ValueError("DQN checkpoint optimizer state contains non-finite values")
        if candidate_optimizer.param_groups[0]["lr"] != float(learning_rate):
            raise ValueError(
                "DQN checkpoint optimizer learning rate does not match hyperparameters"
            )

        candidate_target.eval()
        for parameter in candidate_target.parameters():
            parameter.requires_grad_(False)

        self.policy_network = candidate_policy
        self.target_network = candidate_target
        self.optimizer = candidate_optimizer
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.replay_buffer = ReplayBuffer(
            replay_capacity,
            feature_size=len(self.feature_names),
        )
        self.replay_warmup = replay_warmup
        self.batch_size = batch_size
        self.max_grad_norm = float(max_grad_norm)
        self.target_sync_interval = target_sync_interval
        self.optimizer_steps = counters["optimizer_steps"]
        self.target_sync_count = counters["target_sync_count"]
        self.episodes = counters["episodes"]
        self.last_loss = None
        self.last_mean_abs_td_error = None
        self.last_max_abs_td_error = None
        self.last_mean_q = None
        self.last_max_abs_q = None
        self.last_gradient_norm = None
        self.last_candidate_q_values = []
        self.q_evaluation_count = 0
        self.max_abs_q = 0.0
        self.non_finite_q_values = 0
        self.policy_network.train(self.training)

    def features(self, wrld: SensedWorld, action: AgentAction) -> tuple[float, ...]:
        """Return the canonical ordered DQN state-action feature vector."""
        return dqn_feature_vector(self, wrld, action)

    def predict_next_state(
        self,
        wrld: SensedWorld,
        action: AgentAction,
    ) -> tuple[SensedWorld, list]:
        """Simulate one action using the same prediction path as QAgent."""
        predicted = SensedWorld.from_world(wrld)
        me = predicted.me(self)
        if me is None:
            return predicted, []
        me.move(action.dx, action.dy)
        me.maybe_place_bomb = action.place_bomb
        next_world, events = predicted.next()
        return next_world, events

    def q_value(self, wrld: SensedWorld, action: AgentAction) -> float:
        """Evaluate one action-conditioned feature vector with the policy net."""
        feature_values = self.features(wrld, action)
        feature_tensor = torch.tensor(
            feature_values,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            q_tensor = self.policy_network(feature_tensor)

        q_value = float(q_tensor.item())
        self.q_evaluation_count += 1
        if not math.isfinite(q_value):
            self.non_finite_q_values += 1
            raise ValueError(f"DQN network produced a non-finite Q-value: {q_value}")
        self.max_abs_q = max(self.max_abs_q, abs(q_value))
        return q_value

    def choose_action(
        self,
        wrld: SensedWorld,
        actions: list[AgentAction],
    ) -> Optional[AgentAction]:
        """Choose a random legal action or the first highest-valued action."""
        self.last_candidate_q_values = []
        if not actions:
            return None
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            return random.choice(actions)

        self.last_candidate_q_values = [
            (action, self.q_value(wrld, action))
            for action in actions
        ]
        return max(self.last_candidate_q_values, key=lambda item: item[1])[0]

    def set_learning(self, no_training: bool, *, epsilon: float = 0.1) -> None:
        """Enable normal learning or enter frozen, non-collecting evaluation."""
        self.training = not no_training
        self.learning_enabled = not no_training
        self.collect_experience = not no_training
        self.epsilon = 0.0 if no_training else epsilon
        self.policy_network.train(self.training)

    def set_rollout(self, *, epsilon: float, collect_experience: bool = True) -> None:
        """Use a frozen network for exploratory experience-collection rollout."""
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative and finite")
        self.training = False
        self.learning_enabled = False
        self.collect_experience = collect_experience
        self.epsilon = epsilon
        self.policy_network.eval()

    def reset_episode(self) -> None:
        """Clear pending experience and transient counters before a new episode."""
        self.episode_time_limit = None
        self._q_feature_context = None
        self.pending_features = None
        self.pending_model = None
        self.pending_action = None
        self.episode_transitions = []
        self.episode_action_count = 0
        self.episode_reward = 0.0
        self._episode_finished = False
        self.last_candidate_q_values = []
        self.max_abs_q = 0.0
        self.q_evaluation_count = 0
        self.non_finite_q_values = 0
        self.bombs_placed = 0

    def calc_reward(
        self,
        previous: WorldModel,
        current: WorldModel,
        action: AgentAction,
    ) -> float:
        """Use the same transition reward function as the linear QAgent."""
        return QAgent.calc_reward(self, previous, current, action)

    def legal_actions(self, wrld: SensedWorld) -> list[AgentAction]:
        """Return legal actions for this observed state, including bomb rules."""
        current_model = WorldModel.from_sensed_world(wrld)
        actions = legal_candidate_actions(current_model)
        me = wrld.me(self)
        bomb_action = AgentAction(0, 0, True)
        can_place_bomb = me is not None and not any(
            bomb.owner == me for bomb in wrld.bombs.values()
        )
        actions = [action for action in actions if not action.place_bomb or can_place_bomb]
        if can_place_bomb and bomb_action not in actions:
            actions.append(bomb_action)
        return actions

    def _finish_pending_transition(
        self,
        wrld: SensedWorld,
        actions: list[AgentAction],
    ) -> None:
        if (
            self.pending_features is None
            or self.pending_model is None
            or self.pending_action is None
        ):
            return

        next_model = WorldModel.from_sensed_world(wrld)
        reward = self.calc_reward(
            self.pending_model,
            next_model,
            self.pending_action,
        )
        next_action_features = tuple(
            self.features(wrld, action) for action in actions
        )
        if self.collect_experience:
            self.episode_transitions.append(
                Transition(
                    features=self.pending_features,
                    reward=reward,
                    next_action_features=next_action_features,
                    done=False,
                )
            )
        self.episode_reward += reward
        self.pending_features = None
        self.pending_model = None
        self.pending_action = None

    def finish_episode(self, reward: float) -> bool:
        """Finish the final action with a terminal reward and increment episodes."""
        if self._episode_finished:
            return False
        if self.pending_features is not None:
            if self.collect_experience:
                self.episode_transitions.append(
                    Transition(
                        features=self.pending_features,
                        reward=reward,
                        next_action_features=(),
                        done=True,
                    )
                )
            self.episode_reward += reward
        self.pending_features = None
        self.pending_model = None
        self.pending_action = None
        self._episode_finished = True
        self.episodes += 1
        return True

    def complete_rollout_episode(self, outcome: str) -> None:
        """Finalize timeout or other episode endings without an engine callback."""
        reward = self.R_WIN if outcome == "WON" else self.R_LOSE
        self.finish_episode(reward)

    def done(self, wrld: SensedWorld) -> None:
        """Capture terminal outcomes delivered by the Bomberman world."""
        for event in wrld.events:
            if (
                event.tpe == Event.CHARACTER_FOUND_EXIT
                and event.character.name == self.name
            ):
                self.finish_episode(self.R_WIN)
                return
            if (
                event.tpe == Event.CHARACTER_KILLED_BY_MONSTER
                and event.character.name == self.name
            ):
                self.finish_episode(self.R_LOSE)
                return
            if (
                event.tpe == Event.BOMB_HIT_CHARACTER
                and event.other is not None
                and event.other.name == self.name
            ):
                self.finish_episode(self.R_LOSE)
                return

    def store_transition(self, transition: Transition) -> None:
        """Add an experience to replay while this agent is in training mode."""
        if self.collect_experience:
            self.replay_buffer.add(transition)

    def sync_target_network(self) -> None:
        """Copy the policy parameters into the frozen target network."""
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
        self.target_sync_count += 1

    def _target_value(self, transition: Transition) -> float:
        """Calculate one detached Bellman target from a replay transition."""
        target = transition.reward
        if not transition.done and transition.next_action_features:
            next_features = torch.tensor(
                transition.next_action_features,
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                next_q_values = self.target_network(next_features).reshape(-1)
                if not torch.isfinite(next_q_values).all():
                    raise ValueError("Target network produced a non-finite Q-value")
                max_next_q = float(next_q_values.max().item())
            target += self.gamma * max_next_q

        if not math.isfinite(target):
            raise ValueError(f"DQN Bellman target is not finite: {target}")
        return target

    def train_batch(self) -> Optional[DQNTrainingResult]:
        """Perform at most one optimizer step from a sampled replay minibatch."""
        if not self.learning_enabled:
            return None

        if len(self.replay_buffer) < self.replay_warmup:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        current_features = torch.tensor(
            [transition.features for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        # The network emits [batch_size, 1]; reshape both sides to [batch_size].
        predicted_q = self.policy_network(current_features).reshape(-1)
        if not torch.isfinite(predicted_q).all():
            raise ValueError("Policy network produced a non-finite Q-value")

        target_values = [
            self._target_value(transition)
            for transition in transitions
        ]
        target_q = torch.tensor(
            target_values,
            dtype=torch.float32,
            device=self.device,
        )
        if not torch.isfinite(rewards).all() or not torch.isfinite(target_q).all():
            raise ValueError("DQN replay batch contains non-finite values")

        td_errors = target_q - predicted_q.detach()
        if not torch.isfinite(td_errors).all():
            raise ValueError("DQN TD error contains a non-finite value")

        self.optimizer.zero_grad()
        loss = self.loss_function(predicted_q, target_q)
        if not torch.isfinite(loss):
            raise ValueError("DQN loss is not finite")

        loss.backward()

        # Clipping bounds large gradients before Adam changes the policy.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=self.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad()
            raise ValueError("DQN gradient norm is not finite")

        self.optimizer.step()
        self.optimizer_steps += 1

        target_synced = self.optimizer_steps % self.target_sync_interval == 0
        if target_synced:
            self.sync_target_network()

        loss_value = float(loss.item())
        td_error_values = td_errors.abs()
        q_values = predicted_q.detach()
        result = DQNTrainingResult(
            loss=loss_value,
            mean_abs_td_error=float(td_error_values.mean().item()),
            max_abs_td_error=float(td_error_values.max().item()),
            mean_q=float(q_values.mean().item()),
            max_abs_q=float(q_values.abs().max().item()),
            gradient_norm=float(gradient_norm.item()),
            target_synced=target_synced,
        )
        if not all(
            math.isfinite(value)
            for value in (
                result.loss,
                result.mean_abs_td_error,
                result.max_abs_td_error,
                result.mean_q,
                result.max_abs_q,
                result.gradient_norm,
            )
        ):
            raise ValueError("DQN learning diagnostics contain a non-finite value")

        self.last_loss = result.loss
        self.last_mean_abs_td_error = result.mean_abs_td_error
        self.last_max_abs_td_error = result.max_abs_td_error
        self.last_mean_q = result.mean_q
        self.last_max_abs_q = result.max_abs_q
        self.last_gradient_norm = result.gradient_norm
        return result

    def get_diagnostics(self) -> dict:
        """Return scalar learning counters and current candidate Q-values."""
        return {
            "q_evaluation_count": self.q_evaluation_count,
            "max_abs_q": self.max_abs_q,
            "non_finite_q_values": self.non_finite_q_values,
            "last_candidate_q_values": tuple(self.last_candidate_q_values),
            "replay_size": len(self.replay_buffer),
            "optimizer_steps": self.optimizer_steps,
            "target_sync_count": self.target_sync_count,
            "last_loss": self.last_loss,
            "last_mean_abs_td_error": self.last_mean_abs_td_error,
            "last_max_abs_td_error": self.last_max_abs_td_error,
            "last_mean_q": self.last_mean_q,
            "last_max_abs_q": self.last_max_abs_q,
            "last_gradient_norm": self.last_gradient_norm,
            "learning_rate": self.learning_rate,
        }

    def do(self, wrld: SensedWorld) -> None:
        """Collect a completed transition, then select and queue one action."""
        current_model = WorldModel.from_sensed_world(wrld)
        actions = self.legal_actions(wrld)
        self._finish_pending_transition(wrld, actions)

        action = self.choose_action(wrld, actions)
        if action is None:
            return

        if self.collect_experience:
            self.pending_features = self.features(wrld, action)
            self.pending_model = current_model
            self.pending_action = action
        self.episode_action_count += 1
        self.move(action.dx, action.dy)
        if action.place_bomb:
            self.place_bomb()
            self.bombs_placed += 1
