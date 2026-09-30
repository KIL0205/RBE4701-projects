import random
from typing import Dict, List, Callable
import json

from Bomberman.entity import CharacterEntity
from world_model import Position, WorldModel
from evaluation import evaluate_position, QLEARNING
import navigation

class QAgent(CharacterEntity):
    def __init__(self, name, avatar, x, y, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        super().__init__(name, avatar, x, y)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.weights: Dict[str, float] = {}

    def do(self):
        # TODO: Implement the action selection and execution logic for the agent's turn
        pass

    def features(model: WorldModel) -> Dict[str, float | bool]:
        features = evaluate_position(model, model.self_position, QLEARNING)
        """
        "exit_progress",
        "mobility",
        "escape_options",
        "monster_threat",
        "bomb_threat",
        "explosion_threat",
        "trap_risk",
        "future_monster_risk",
        "future_escape_options",
        "future_trap_risk",
        "lethal",
        """
        return features

        

    
    def get_q_value(self, features: Dict[str, float]) -> float:

        q_value = 0.0

        for name, value in features.items():
            weight = self.weights.get(name, 0.0)
            q_value += weight * value

        return q_value


    def choose_action(self, actions: List[str], feature_function: Callable) -> str:

        if not actions:
            return None

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(actions)

        # Exploitation
        best_action = None
        best_q = float("-inf")

        for action in actions:
            features = feature_function(action)
            q_value = self.get_q_value(features)

            if q_value > best_q:
                best_q = q_value
                best_action = action

        return best_action


    def update(self, features: Dict[str, float], reward: float, next_max_q: float):
        current_q = self.get_q_value(features)

        target = reward + self.gamma * next_max_q
        difference = target - current_q

        for name, value in features.items():
            old_weight = self.weights.get(name, 0.0)

            self.weights[name] = (old_weight + self.alpha * difference * value)

    def save_weights(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(self.weights, file)

    def load_weights(self, file_path: str):
        with open(file_path, "r") as file:
            self.weights = json.load(file)