import random
import math
import warnings
from typing import Dict, List, Tuple
import json

from Bomberman.entity import CharacterEntity
from Bomberman.events import Event
from Bomberman.sensed_world import SensedWorld
from team01.agent.black_board import BlackBoard
from team01.agent.blackboard_keys import BBKeys
from team01.agent.controller import QLearningRootController

from .actions import AgentAction
from .evaluation import (
    Q_FEATURE_VERSION,
    evaluate_q_features,
    prepare_q_feature_context,
    profiles_from_mapping,
    q_feature_names,
)
from .safety import find_executable_exit_action, legal_candidate_actions, monster_threat_cells, assess_immediate_safety
from .world_model import WorldModel

class QAgent(CharacterEntity):
    # rewards
    R_WIN = 1000.0
    R_LOSE = -1000.0
    R_COST_OF_LIVING = -0.1
    R_NEAR_MONSTER = 0.0
    R_STEP = 0.0
    R_KILL_MONSTER = 0.0
    R_BREAK_WALL = 0.0
    R_PLACE_BOMB = 0.0
    
    # class QCandidate(action, features, ):
    #     action: AgentAction
    #     features: Dict[str, float]
    #     eligible: bool
    #     rejection_reason: str

    def __init__(self, name, avatar, x, y, alpha: float = 0.0005, gamma: float = 0.9, epsilon: float = 0.05):
        super().__init__(name, avatar, x, y)
        
        ## behavior tree
        self.blackboard = BlackBoard()
        self.evaluation_profiles = profiles_from_mapping(None) ## NOTE: originally on bomberman agent. needed?
        self.root = QLearningRootController(self.blackboard, self.evaluation_profiles)
        ## -------------
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # keeping track of world stuff
        self.num_monsters = 0
        # prev_model tracks the state for the previous transition.
        self.prev_world_state = None

        self.prev_features = None
        self.prev_action = None
        self.prev_model = None

        self.training = True
        self.debug = False
        self.q_contributions_diagnostic = False
        self.diagnostic_tick = 0
        self.episode_time_limit = None
        self._q_feature_context = None
        self.last_score = 0.0
        self.episode_reward = 0.0
        self.bombs_placed = 0
        self.last_candidate_q_values = []
        self.non_finite_q_fallbacks = 0
        self.non_finite_q_values = 0
        self.total_abs_td_error = 0.0
        self.td_update_count = 0
        self.max_abs_q = 0.0
        self.feature_activation_counts: Dict[str, int] = {}

        self.weights: Dict[str, float] = {}
        self.current_feature_names = set()

    def do(self, wrld: SensedWorld):
        current_model = WorldModel.from_sensed_world(wrld)
        
        # NOTE:
        # basic approach to q-learning:
        #   Pick an initial state, s, at random
        #   while not at goal state do:
        #       pick an action a at random
        #       get s' and r
        #       update Q(s, a) using (r + gamma*max(a')[Q(s', a')])
        #   end while
        
        # update with behavior tree:
        #   - QAgent owns action selection & learning
        #   - behavior tree decides the context and allowed actions for each state
        
        # The Q-learning loop is better thought of as: 
        #   observe state (s), choose an action (a), observe reward (r) and next state (s'), 
        #   then update (Q(s,a)) using the best allowed action in (s').
        
        #   the state isn’t picked randomly each turn; in your agent, the world observation supplies it.
        
        # Update action set selection using tree ==============
        legal_actions = legal_candidate_actions(current_model)
        
        legal_actions_features = [self.features(wrld, action) for action in legal_actions]
        
        self.blackboard.erase() #sets all keys values to None

        self.blackboard.set(BBKeys.WORLD_MODEL, current_model)
        self.blackboard.set(BBKeys.SENSED_WORLD, wrld)
        self.blackboard.set(BBKeys.POSSIBLE_ACTIONS, legal_actions)
        self.blackboard.set(BBKeys.POSSIBLE_ACTIONS_FEATURES, legal_actions_features)
        self.root.tick()
        
        candidate_actions = {
            action
            for action, features in self.blackboard.get(BBKeys.Q_CANDIDATES).items()
            }
        
        candidates = self.blackboard.get(BBKeys.Q_CANDIDATES)
        
        # ==============
        
        me = wrld.me(self)
        bomb_action = AgentAction(0, 0, True)
        can_place_bomb = me is not None and not any(
            bomb.owner == me for bomb in wrld.bombs.values()
        )
        actions = [action for action in candidate_actions if not action.place_bomb or can_place_bomb]
        # if can_place_bomb and bomb_action not in actions: ## NOTE: no randomly adding bombs now
        #     legal_actions.append(bomb_action)

        transition_debug = None
        if (
            self.training
            and self.prev_features is not None
            and self.prev_model is not None
            and self.prev_action is not None
        ):
            reward = self.calc_reward(self.prev_model, current_model, self.prev_action)
            next_max_q, _ = self.max_q_value(wrld, actions)
            previous_q = self.get_q_value(self.prev_features)
            target = reward + self.gamma * next_max_q
            td_error = self.update(self.prev_features, reward, next_max_q)
            self._record_learning_diagnostics(self.prev_features, td_error)
            self.episode_reward += reward
            transition_debug = (self.prev_action, reward, previous_q, next_max_q, target, td_error)

        action = self.choose_action(wrld, actions)
        if action is None:
            return

        features = candidates[action]# self.features(wrld, action)

        self.prev_model = current_model
        self.prev_action = action
        self.prev_features = features

        self.move(action.dx, action.dy)

        if action.place_bomb:
            self.place_bomb()
            self.bombs_placed += 1

        if self.debug:
            candidate_q_values = self.last_candidate_q_values
            if not candidate_q_values:
                _, candidate_q_values = self.max_q_value(wrld, actions)
            candidates = ", ".join(
                f"{candidate}={q_value:.3f}"
                for candidate, q_value in candidate_q_values
            )
            if transition_debug is not None:
                previous_action, reward, previous_q, next_max_q, target, td_error = transition_debug
                print(
                    f"[Q] update action={previous_action} reward={reward:.2f} "
                    f"Q={previous_q:.3f} next={next_max_q:.3f} "
                    f"target={target:.3f} TD={td_error:.3f} weights={self.weights}"
                )
            print(f"[Q] candidates=[{candidates}] selected={action}")

    def features(self, wrld: SensedWorld, action: AgentAction) -> Dict[str, float]:
        if self.episode_time_limit is None:
            self.episode_time_limit = max(1, int(wrld.time))
        context = self._q_feature_context
        if context is not None and context.current_world is wrld:
            current_model = context.current_model
        else:
            current_model = WorldModel.from_sensed_world(wrld)
            context = prepare_q_feature_context(
                wrld,
                current_model,
                self.episode_time_limit,
            )
            self._q_feature_context = context
        predicted_world, predicted_events = self.predict_next_state(wrld, action)
        predicted_model = WorldModel.from_sensed_world(predicted_world)
        if predicted_model.self_position is None:
            me = wrld.me(self)
            if me is not None:
                predicted_model.self_position = (me.x + action.dx, me.y + action.dy)
        return evaluate_q_features(
            wrld,
            predicted_world,
            current_model,
            predicted_model,
            action,
            agent_name=self.name,
            time_limit=self.episode_time_limit,
            predicted_events=predicted_events,
            context=context,
        )

    def predict_next_state(self, wrld: SensedWorld, action: AgentAction) -> Tuple[SensedWorld, List]:
        predicted = SensedWorld.from_world(wrld)
        me = predicted.me(self)
        if me is None:
            return predicted, []
        me.move(action.dx, action.dy)
        me.maybe_place_bomb = action.place_bomb
        next_world, events = predicted.next()
        return next_world, events
    
    def get_q_value(self, features: Dict[str, float]) -> float:
        self.sync_weights(features)
        q_value = 0.0

        for name, value in features.items():
            weight = self.weights.get(name, 0.0)
            q_value += weight * value

        return q_value

    def sync_weights(self, features: Dict[str, float]):
        self.current_feature_names = set(features)
        for name in features:
            self.weights.setdefault(name, 0.0)


    def choose_action(self, wrld: SensedWorld, actions: List[AgentAction]) -> AgentAction:
        """Choose the best action based on Q-values."""       
        self.last_candidate_q_values = []
        if not actions:
            return None

        # exploration - if random choice is less than epsilon, pick random action
        if self.training and random.random() < self.epsilon:
            selected_action = random.choice(actions) # TODO: maybe update with curious exploration function?
            if self.q_contributions_diagnostic:
                random_state = random.getstate()
                try:
                    _, _, details = self._score_candidate_actions(
                        wrld,
                        actions,
                        update_state=False,
                        record_diagnostics=False,
                    )
                finally:
                    random.setstate(random_state)
                self._print_q_contributions(
                    wrld,
                    details,
                    selected_action,
                    "epsilon exploration",
                )
            return selected_action

        # exploitation - otherwise, pick best known action in current state
        best_q, scored_actions = self.max_q_value(wrld, actions)
        self.last_candidate_q_values = scored_actions
        finite_actions = [
            (action, q_value)
            for action, q_value in scored_actions
            if math.isfinite(q_value)
        ]
        if not finite_actions:
            self.non_finite_q_fallbacks += 1
            selected_action = random.choice(actions)
            if self.q_contributions_diagnostic:
                self._print_q_contributions(
                    wrld,
                    self._last_candidate_details,
                    selected_action,
                    "non-finite Q fallback",
                )
            return selected_action

        best_q = max(q_value for _, q_value in finite_actions)
        best_actions = [action for action, q_value in finite_actions if q_value == best_q]
        selected_action = random.choice(best_actions)
        if self.q_contributions_diagnostic:
            self._print_q_contributions(
                wrld,
                self._last_candidate_details,
                selected_action,
                "max Q tie-break" if len(best_actions) > 1 else "max Q",
            )
        return selected_action

    def q_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Return feature-weight products without modifying agent state."""
        return {
            name: self.weights.get(name, 0.0) * value
            for name, value in features.items()
        }

    @staticmethod
    def _format_action(action: AgentAction) -> str:
        if action.place_bomb:
            return "PLACE_BOMB"
        if action.dx == 0 and action.dy == 0:
            return "WAIT"
        directions = {
            (0, -1): "UP",
            (0, 1): "DOWN",
            (-1, 0): "LEFT",
            (1, 0): "RIGHT",
            (-1, -1): "UP_LEFT",
            (1, -1): "UP_RIGHT",
            (-1, 1): "DOWN_LEFT",
            (1, 1): "DOWN_RIGHT",
        }
        return f"MOVE_{directions.get((action.dx, action.dy), f'{action.dx}_{action.dy}')}"

    def _print_q_contributions(
        self,
        wrld: SensedWorld,
        details: List[Tuple[AgentAction, Dict[str, float], Dict[str, float], float]],
        selected_action: AgentAction,
        selection_reason: str,
    ) -> None:
        finite_details = [item for item in details if math.isfinite(item[3])]
        best_q = max((item[3] for item in finite_details), default=None)
        greedy_actions = (
            [item[0] for item in finite_details if item[3] == best_q]
            if best_q is not None
            else []
        )
        bomb_details = next((item for item in details if item[0].place_bomb), None)
        bomb_is_greedy = bomb_details is not None and bomb_details[0] in greedy_actions
        if not selected_action.place_bomb and not bomb_is_greedy:
            return

        me = wrld.me(self)
        position = (me.x, me.y) if me is not None else None
        monster_count = sum(len(group) for group in wrld.monsters.values())
        bomb_count = len(wrld.bombs)
        exit_position = wrld.exitcell
        greedy_names = [self._format_action(action) for action in greedy_actions]

        print("\nQ CONTRIBUTIONS", flush=True)
        print(f"Tick: {self.diagnostic_tick}", flush=True)
        print(f"Position: {position}", flush=True)
        print(f"Monsters alive: {monster_count}", flush=True)
        print(f"Active bombs: {bomb_count}", flush=True)
        print(f"Exit position: {exit_position}", flush=True)
        print("Candidate Q-values:", flush=True)
        for action, _, _, q_value in details:
            print(f"  {self._format_action(action):<14} Q={q_value:+.3f}", flush=True)

        for action, _, contributions, q_value in details:
            print(f"\n{self._format_action(action)}", flush=True)
            for name, value in contributions.items():
                print(f"  {name:<24} {value:+.3f}", flush=True)
            print(f"  {'-' * 30}\n  {'Q':<24} {q_value:+.3f}", flush=True)

        greedy_label = ", ".join(greedy_names) if greedy_names else "none (no finite Q)"
        print(f"Greedy action: {greedy_label}", flush=True)
        print(f"Selected: {self._format_action(selected_action)}", flush=True)
        print(f"Reason: {selection_reason}", flush=True)

        if selected_action.place_bomb and bomb_details is not None:
            non_bomb_values = [
                item[3]
                for item in finite_details
                if not item[0].place_bomb
            ]
            best_non_bomb_q = max(non_bomb_values) if non_bomb_values else None
            bomb_q = bomb_details[3]
            difference = (
                bomb_q - best_non_bomb_q
                if best_non_bomb_q is not None
                else None
            )
            print("\nBOMB DECISION", flush=True)
            print(f"Greedy: {greedy_label}", flush=True)
            print(f"Selected: {self._format_action(selected_action)}", flush=True)
            print(f"Bomb Q: {bomb_q:+.3f}", flush=True)
            if best_non_bomb_q is None:
                print("Best non-bomb Q: unavailable", flush=True)
                print("Difference: unavailable", flush=True)
            else:
                print(f"Best non-bomb Q: {best_non_bomb_q:+.3f}", flush=True)
                print(f"Difference: {difference:+.3f}", flush=True)

    def max_q_value(self, wrld: SensedWorld, actions: List[AgentAction]):
        best_q, scored_actions, details = self._score_candidate_actions(
            wrld,
            actions,
            update_state=True,
            record_diagnostics=True,
        )
        self._last_candidate_details = details
        return best_q, scored_actions

    def _score_candidate_actions(
        self,
        wrld: SensedWorld,
        actions: List[AgentAction],
        update_state: bool,
        record_diagnostics: bool,
    ):
        """Score each candidate action based on its Q-value."""
        if not actions:
            return 0.0, [], []
        scored_actions = []
        details = []
        # records = [] # list of candidate records, each containing an action & corresponding features
        for action in actions:
            features = self.features(wrld, action)
            
            # # Create a record for the current action and its features
            # action_record = {"action": action, "features": features}
            # records.append(QCandidate(
            #     action=action,
            #     features=features,
            #     eligible=True,
            #     rejection_reason=""
            # ))
            
            if update_state:
                q_value = self.get_q_value(features)
                if self.q_contributions_diagnostic:
                    contributions = self.q_contributions(features)
                    details.append((action, features, contributions, q_value))
            else:
                contributions = self.q_contributions(features)
                q_value = sum(contributions.values())
                details.append((action, features, contributions, q_value))
            if record_diagnostics:
                if not math.isfinite(q_value):
                    self.non_finite_q_values += 1
                else:
                    self.max_abs_q = max(self.max_abs_q, abs(q_value))
            scored_actions.append((action, q_value))
        
        # # post the record of each action's feature dict to the blackboard
        # self.blackboard.set(BBKeys.Q_CANDIDATES, records)
        
        best_q = max(q for _, q in scored_actions)
        return best_q, scored_actions, details

    # Not currently used. Standard one-step Q-learning calculates max Q(s', a')
    # directly from the newly observed state, so this additional predicted
    # lookahead is unnecessary for the first implementation.
    # def future_q_value(self, current_model, action):
    #     predicted_world, _ = self.predict_next_state(current_model, action)
    #     possible_actions = legal_candidate_actions(predicted_world)
    #     if not possible_actions:
    #         return 0.0
    #     action = self.choose_action(possible_actions, current_model=predicted_world)
    #     return max(self.get_q_value(self.features(predicted_world, a)) for a in possible_actions)
    
    def calc_reward(self, previous: WorldModel, current: WorldModel, action: AgentAction) -> float:
        """Calculates reward for the previous state-action and current state."""
        reward = 0.0
        previous_position = previous.self_position
        current_position = current.self_position

        # Terminal death/win rewards are handled separately through finish_episode() + done().
        # this wouldnt work correctly if the character is already absent from the observed WorldModel (it dies).
        # if current_position in current.monster_positions():
        #     return self.R_LOSE
        # blast = set()
        # for bomb, timer in current.bomb_timers.items():
        #     if timer <= 1:
        #         from .evaluation import bomb_blast_cells
        #         blast.update(bomb_blast_cells(current, bomb))
        # if current_position in blast:
        #     return self.R_LOSE
        # if current_position == current.exit_position:
        #     return self.R_WIN

        # cost of living
        reward += self.R_COST_OF_LIVING # minor cost for staying alive

        # other costs:
        if current_position is not None and current_position in monster_threat_cells(current, 2):
            # within possible chasing distance of monster
            reward += self.R_NEAR_MONSTER # medium cost for being in danger

        if (
            previous_position is not None
            and current_position is not None
            and current.exit_position is not None
        ):
            exit_position = current.exit_position
            dist_to_exit = max(abs(current_position[0] - exit_position[0]), abs(current_position[1] - exit_position[1]))
            prev_dist_to_exit = max(abs(previous_position[0] - exit_position[0]), abs(previous_position[1] - exit_position[1]))
            if dist_to_exit < prev_dist_to_exit:
                reward += self.R_STEP # medium reward for getting closer to exit

        # if monster dead, +20
        if len(current.monsters) < len(previous.monsters):
            reward += self.R_KILL_MONSTER # high reward for killing a monster

        # Currently disabled: the engine reports wall-hit events but does not
        # remove walls, so comparing wall counts cannot detect a break.
        # if len(current.walls) < len(previous.walls):
        #     reward += self.R_BREAK_WALL # minor reward for breaking a wall

        # Bomb placement is neutral; only its consequences affect reward.
        if action.place_bomb:
            reward += self.R_PLACE_BOMB

        return reward

    def finish_episode(self, reward: float):
        if not self.training or self.prev_features is None:
            return False

        previous_q = self.get_q_value(self.prev_features)
        td_error = self.update(self.prev_features, reward, 0.0)
        self._record_learning_diagnostics(self.prev_features, td_error)
        self.episode_reward += reward
        if self.debug:
            print(
                f"[Q] terminal action={self.prev_action} reward={reward:.2f} "
                f"Q={previous_q:.3f} next=0.000 target={reward:.3f} "
                f"TD={td_error:.3f} weights={self.weights}"
            )
        self.prev_features = None
        self.prev_action = None
        self.prev_model = None
        return True

    def done(self, wrld: SensedWorld):
        for event in wrld.events:
            if event.tpe == Event.CHARACTER_FOUND_EXIT and event.character.name == self.name:
                self.finish_episode(self.R_WIN)
                return
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER and event.character.name == self.name:
                self.finish_episode(self.R_LOSE)
                return
            if (
                event.tpe == Event.BOMB_HIT_CHARACTER
                and event.other is not None
                and event.other.name == self.name
            ):
                self.finish_episode(self.R_LOSE)
                return

        # TODO: Handle terminal callbacks without a matching win/death event.
        

    def update(self, features: Dict[str, float], reward: float, next_max_q: float) -> float:
        current_q = self.get_q_value(features)

        target = reward + self.gamma * next_max_q
        difference = target - current_q

        for name, value in features.items():
            old_weight = self.weights.get(name, 0.0)

            self.weights[name] = (old_weight + self.alpha * difference * value)

        return difference

    def _record_learning_diagnostics(self, features: Dict[str, float], td_error: float):
        self.total_abs_td_error += abs(td_error)
        self.td_update_count += 1
        for name, value in features.items():
            self.feature_activation_counts[name] = (
                self.feature_activation_counts.get(name, 0)
                + int(abs(value) > 1e-9)
            )

    def save_weights(self, file_path: str):
        feature_names = tuple(q_feature_names())
        self.sync_weights(dict.fromkeys(feature_names, 0.0))
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "feature_version": Q_FEATURE_VERSION,
                    "features": list(feature_names),
                    "weights": {name: self.weights[name] for name in feature_names},
                },
                file,
                indent=2,
            )

    def load_weights(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            saved = json.load(file)

        if not isinstance(saved, dict):
            raise ValueError("Q-learning weights must be a JSON object")

        if "feature_version" not in saved:
            raise ValueError(
                "Unversioned Q-learning weights cannot be safely reused; "
                "the required feature_version field is missing."
            )

        required_fields = {"feature_version", "features", "weights"}
        missing_fields = required_fields - saved.keys()
        if missing_fields:
            raise ValueError(
                "Q-learning weights are missing required field(s): "
                + ", ".join(sorted(missing_fields))
            )

        saved_version = saved["feature_version"]
        if saved_version != Q_FEATURE_VERSION:
            raise ValueError(
                f"Saved weights use feature version {saved_version}; current "
                f"implementation uses version {Q_FEATURE_VERSION}. "
                "Previous weights cannot be safely reused. Start with new weights "
                "or explicitly migrate them."
            )

        saved_features = saved["features"]
        if not isinstance(saved_features, list) or not all(
            isinstance(name, str) for name in saved_features
        ):
            raise ValueError("Q-learning 'features' must be a list of feature names")

        saved_weights = saved["weights"]
        if not isinstance(saved_weights, dict):
            raise ValueError("Q-learning 'weights' must be a dictionary")

        current_features = set(q_feature_names())
        saved_features = set(saved_features)

        if not all(isinstance(value, (int, float)) for value in saved_weights.values()):
            raise ValueError("Q-learning weights must be numeric")

        matching_features = saved_features & current_features
        if saved_features and not matching_features:
            raise ValueError("Saved weights contain no features used by the current implementation")

        missing_features = current_features - saved_features
        stale_features = saved_features - current_features
        if missing_features or stale_features:
            warnings.warn(
                f"Partial weight compatibility: initialized {len(missing_features)} "
                f"new features to zero and ignored {len(stale_features)} stale features.",
                UserWarning,
                stacklevel=2,
            )

        self.weights = {
            name: float(saved_weights.get(name, 0.0))
            for name in current_features
        }
        self.current_feature_names = current_features

    def set_learning(self, no_training: bool, *, epsilon: float = 0.1) -> None:
        if no_training:
            self.training = False
            self.epsilon = 0.0
        else:
            self.training = True
            self.epsilon = epsilon