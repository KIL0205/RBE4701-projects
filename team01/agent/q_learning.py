import random
from typing import Dict, List, Callable
import json

from Bomberman.entity import CharacterEntity
from world_model import Position, WorldModel
from evaluation import immediate_lethal_positions, monster_immediate_cells, bomb_blast_cells
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

    def features(model: WorldModel) -> Dict[str, float]:
        features = {"distanceToGoal": None,
                    "distanceToEnemy": None,
                    "inBombDanger": None,
                    "mobility": None
                    }
        
        sx, sy = WorldModel.self_position
        num_steps = 2

        # Distance to goal feature (could be changed to goal progress 1/x)
        if WorldModel.exit_position:
            gx, gy = WorldModel.exit_position
            features["distanceToGoal"] = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5

        # Total distance to monsters
        if WorldModel.monsters:
            mDist = 0
            for mx, my in WorldModel.monster_positions:
                mDist = mDist + ((mx - sx) ** 2 + (my - sy) ** 2) ** 0.5
            features["distanceToEnemy"] = mDist

        # In explosion range of bomb
        if WorldModel.bombs:
            for bx, by in WorldModel.bomb_positions:
                dx = abs(sx - bx)
                dy = abs(sy - by)
                if (dx == 0 and dy <= WorldModel.explosion_range) or (dx <= WorldModel.explosion_range and dy == 0) or (dx == dy == 0):
                    features["inBombDanger"] = 1
                else:
                    features["inBombDanger"] = 0

        # Mobility. Should always have the self_position, so if statement is probably redundant
        if WorldModel.self_position:
            forbidden_cells = immediate_lethal_positions(model)
            forbidden = forbidden_cells or set()
            visited = {WorldModel.self_position}
            frontier = [(WorldModel.self_position, 0)]
            while frontier:
                current, steps = frontier.pop(0)
                if steps < num_steps:
                    for neighbor in model.neighbors(current):
                        if neighbor in forbidden:
                            continue
                        if neighbor not in visited:
                            visited.add(neighbor)
                            frontier.append((neighbor, steps + 1))
            features["mobility"] = len(visited) - 1  # exclude the starting position

        # Loops for escaping monsters in future chambers [To be implemented when I can think clearly]
        # WorldModel.walls
        

    
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