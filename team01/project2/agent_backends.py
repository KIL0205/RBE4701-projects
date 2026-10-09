"""Shared Project 2 agent choices, labels, and learned-state paths."""

from pathlib import Path


AGENT_CHOICES = ("q", "dqn", "qbt")
DEFAULT_AGENT = "qbt"
Q_LEARNING_AGENTS = frozenset(("q", "qbt"))
AGENT_LABELS = {
    "q": "Q-Learning",
    "dqn": "Deep Q-Network",
    "qbt": "Behavior Tree w/ Q-Learning",
}

PROJECT2_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_Q_WEIGHTS_PATH = PROJECT2_DIRECTORY / "q_learning_weights.json"
DEFAULT_DQN_CHECKPOINT_PATH = PROJECT2_DIRECTORY / "dqn_checkpoint.pt"
DEFAULT_QBT_WEIGHTS_PATH = PROJECT2_DIRECTORY / "q_learning_bt_weights.json"
DEFAULT_AGENT_STATE_PATHS = {
    "q": DEFAULT_Q_WEIGHTS_PATH,
    "dqn": DEFAULT_DQN_CHECKPOINT_PATH,
    "qbt": DEFAULT_QBT_WEIGHTS_PATH,
}
