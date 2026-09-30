import random
from typing import Dict, List, Callable
import json

from .behavior_tree import ActionNode, ConditionNode, SelectorNode, SequenceNode, Status
from .black_board import BlackBoard
from .blackboard_keys import BBKeys
from .world_model import WorldModel
from .actions import AgentAction

from .safety import (
    filter_safe_actions,
    find_executable_exit_action,
    legal_candidate_actions,
    monster_immediate_reachable_cells,
    monster_reachable_layers,
    monster_threat_cells
)

from Bomberman.entity import CharacterEntity


class QAgent(CharacterEntity):
    # rewards
    R_WIN = 1000
    R_LOSE = -1000
    R_COST_OF_LIVING = -1
    R_NEAR_MONSTER = -5
    R_KILL_MONSTER = 20
    R_BREAK_WALL = 5
    R_PLACE_BOMB = 2

    def __init__(self, name, avatar, x, y, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        super().__init__(name, avatar, x, y)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # keeping track of world stuff
        self.num_monsters = 0
        self.prev_world_state = None

        self.weights: Dict[str, float] = {}

    def do(self):
        # TODO: Implement the action selection and execution logic for the agent's turn
        pass

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
    
    def calc_reward(self, action: AgentAction, model: WorldModel) -> float:
        """ Calculates and returns a reward based on the last action of the character. """
        reward = 0.0
        prev_pos = [self.x - action.dx, self.y - action.dy]
        
        # TODO: catch errors
        # TODO: test
        
        # first, get new state of character
        pos = [self.x, self.y]
        
        # first, check the two end conditions
        # if monster in same position
        if pos in model.monster_positions():
            return self.R_LOSE # harsh penalty for dying
        
        # if in bomb blast
        blast = set()
        for bomb, timer in model.bomb_timers.items():
            if timer <= 1:
                from .evaluation import bomb_blast_cells
                blast.update(bomb_blast_cells(model, bomb))
        if pos in blast:
            return self.R_LOSE # harsh penalty for dying
        
        # if won
        exit = model.exit_position
        if pos in exit:
            return self.R_WIN # high reward for winning
        
        # cost of living
        reward -= self.R_COST_OF_LIVING # minor cost for staying alive
        
        # other costs:
        if pos in monster_threat_cells(model, 2):
            # within possible chasing distance of monster
            reward -= self.R_NEAR_MONSTER # medium cost for being in danger
            
        # calculate distance to exit, if prev_pos further than pos, then reward +5
        dist_to_exit = max(abs(pos[0] - exit[0]), abs(pos[1] - exit[1])) # best possible distance to the exit
        prev_dist_to_exit = max(abs(prev_pos[0] - exit[0]), abs(prev_pos[1] - exit[1]))
        
        if dist_to_exit < prev_dist_to_exit:
            reward += self.R_STEP # medium reward for getting closer to exit
        
        # if monster dead, +20
        if len(model.monsters) < len(self.prev_world_state.monsters):
            reward += self.R_KILL_MONSTER # high reward for killing a monster

        # if wall broken, +5
        if len(model.walls) < len(self.prev_world_state.walls): # TODO: can walls be broken?
            reward += self.R_BREAK_WALL # minor reward for breaking a wall

        # if bomb placed, +2
        if action.place_bomb:
            reward += self.R_PLACE_BOMB # minor reward to encourage using bombs
            
        self.prev_world_state = model # update prev world state
            
        return reward
        

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