import sys

from team01.agent import actions

sys.path.insert(0, '../../Bomberman')

from entity import CharacterEntity

from .behavior_tree import ActionNode, ConditionNode, QLearningActionNode, SelectorNode, SequenceNode, Status
from .black_board import BlackBoard
from .blackboard_keys import BBKeys
from .world_model import WorldModel
from .actions import AgentAction
from .safety import (
    assess_immediate_safety,
    filter_safe_actions,
    find_executable_exit_action,
    legal_candidate_actions,
    monster_immediate_reachable_cells,
    monster_reachable_layers,
    safety_result_dict,
    short_horizon_survivability,
)
from .navigation import find_path
from .evaluation import (
    bomb_blast_cells,
    evaluate_action,
    immediate_lethal_positions,
    profiles_from_mapping,
    rank_actions,
)


class IsImmediateDanger(ConditionNode):
    """Select the emergency branch when the current cell cannot survive."""
    def __init__(self, blackboard):
        super().__init__("is_immediate_danger", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "IsImmediateDanger"})
        current_result = assess_immediate_safety(model, AgentAction(0, 0, False))
        if not current_result.eligible:
            self.blackboard.set(BBKeys.DEBUG_INFO, {
                            "active_behavior": "IsImmediateDanger",
                            "danger_reason": current_result.rejection_reason,
                        })
            return Status.SUCCESS
        return Status.FAILURE

## TODO: UPDATE W/QLEARNING ================================================================
class Q_ChooseSafeMoveSet(QLearningActionNode):
    """[Q-Learning] Choose and publish to blackboard the immediately safe actions as candidates."""
    def __init__(self, blackboard, profile):
        super().__init__(" q_choose_safe_move_set", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Q_ChooseSafeMove"})
        
        # determine immediately safe actions (NOTE: no bomb placement)
        legal_actions = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS)
        legal_features = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS_FEATURES)
        legal = {action: features for action, features in zip(legal_actions, legal_features)}
        safe_candidates = {}
        
        def safe_action(action) -> bool:
            return (legal[action].get("loss_next_update") != 1.0 # safe if the action doesn't result in loss
                    and legal[action].get("explosion_threat") != 1.0 # safe if action doesn't result in being in active explosion
                    and ((action.place_bomb and legal[action].get("bomb_escape_margin") >= 0) # safe if placed bomb and action has bomb escape margin
                        or not action.place_bomb and legal[action].get("active_bomb_escape_margin") >= 0) # safe if didn't place bomb and action has active bomb escape margin
                    )
        
        safe_candidates = {
            action: features
            for action, features in legal.items()
            if safe_action(action)
        }
        
        if not safe_candidates:
            return Status.FAILURE

        self.blackboard.set(BBKeys.Q_CANDIDATES, safe_candidates)
        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Q_ChooseSafeMove"})
        
        return Status.SUCCESS

class ChooseSafeMove(ActionNode):
    """Choose the highest-scoring action after immediate safety filtering."""
    def __init__(self, blackboard, profile):
        super().__init__("choose_safe_move", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "ChooseSafeMove"})
        danger = monster_immediate_reachable_cells(model)
        _, future_danger = monster_reachable_layers(model)
        legal = legal_candidate_actions(model)
        safety_results = [assess_immediate_safety(model, action) for action in legal]
        eligible = [result.action for result in safety_results if result.eligible]
        safe = filter_safe_actions(model, eligible, danger)
        rankings = rank_actions(model, eligible, self.profile)

        self.blackboard.set(BBKeys.POSSIBLE_ACTIONS, legal)
        self.blackboard.set(BBKeys.SAFE_ACTIONS, safe)
        self.blackboard.set(BBKeys.MONSTER_THREAT_CELLS, danger)
        self.blackboard.set(BBKeys.DANGER_CELLS, danger)
        self.blackboard.set(BBKeys.MONSTER_T2_THREAT_CELLS, future_danger)
        self.blackboard.set(BBKeys.EVALUATION_BREAKDOWNS, rankings)
        self.blackboard.set(
            BBKeys.CANDIDATE_SAFETY,
            [safety_result_dict(result) for result in safety_results],
        )

        best = next((result for result in rankings if not result.lethal), None)
        if best is None:
            self.blackboard.set(BBKeys.SELECTED_ACTION, AgentAction(0, 0, False))
            return Status.SUCCESS

        self.blackboard.set(BBKeys.SELECTED_ACTION, best.action)
        return Status.SUCCESS

## QLEARNING VERSION ================================================================
class Q_IsDangerSoon(ConditionNode):
    """Detect an observable near-term hazard without multi-tick search."""
    def __init__(self, blackboard):
        super().__init__("is_danger_soon", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        lethal = immediate_lethal_positions(model)
        bomb_danger = set()
        for position, timer in model.bomb_timers.items():
            if timer <= 2:
                bomb_danger.update(bomb_blast_cells(model, position))
        danger = lethal | bomb_danger
        legal = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS)
        current_result = assess_immediate_safety(model, AgentAction(0, 0, False))
        has_eligible_action = False
        for action in legal:
            if assess_immediate_safety(model, action).eligible:
                has_eligible_action = True
                break
        
        in_danger = False
        
        for action in legal:
            action_pos = (model.self_position[0] + action.dx, model.self_position[1] + action.dy)
            if action_pos in danger:
                in_danger = True
                break

        if in_danger or not current_result.eligible or not has_eligible_action:
            self.blackboard.set(BBKeys.DEBUG_INFO, {
                "active_behavior": "IsDangerSoon",
                "danger_reason": "near_term_observable_hazard",
            })
            self.blackboard.set(BBKeys.DANGER_CELLS, danger)
            return Status.SUCCESS
        return Status.FAILURE


class IsDangerSoon(ConditionNode):
    """Detect an observable near-term hazard without multi-tick search."""
    def __init__(self, blackboard):
        super().__init__("is_danger_soon", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        lethal = immediate_lethal_positions(model)
        bomb_danger = set()
        for position, timer in model.bomb_timers.items():
            if timer <= 2:
                bomb_danger.update(bomb_blast_cells(model, position))
        danger = lethal | bomb_danger
        legal = legal_candidate_actions(model)
        current_result = assess_immediate_safety(model, AgentAction(0, 0, False))
        has_eligible_action = False
        for action in legal:
            if assess_immediate_safety(model, action).eligible:
                has_eligible_action = True
                break

        if not current_result.eligible or not has_eligible_action:
            self.blackboard.set(BBKeys.DEBUG_INFO, {
                "active_behavior": "IsDangerSoon",
                "danger_reason": "near_term_observable_hazard",
            })
            self.blackboard.set(BBKeys.DANGER_CELLS, danger)
            return Status.SUCCESS
        return Status.FAILURE


## QLEARNING VERSION ================================================================
class Q_AvoidThreatMoveSet(QLearningActionNode):
    """[Q-Learning] Choose and publish to blackboard actions that avoid future threats as candidates."""
    def __init__(self, blackboard, profile):
        super().__init__("avoid_threat", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE
        
        # choose candidate actions that best avoid future threats + bomb placement
        legal_actions = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS)
        legal_features = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS_FEATURES)
        legal = {action: features for action, features in zip(legal_actions, legal_features)}
        
        avoid_threat_candidates = {}
        
        def safe_in_future(action) -> bool:
            return (legal[action].get("loss_next_update") != 1.0  # safe if the action doesn't result in loss
                    and legal[action].get("explosion_threat") != 1.0  # safe if action doesn't result in being in active explosion
                    and ((action.place_bomb and legal[action].get("bomb_escape_margin") >= 0) # safe if placed bomb and action has bomb escape margin
                        or not action.place_bomb and legal[action].get("active_bomb_escape_margin") >= 0) # safe if didn't place bomb and action has active bomb escape margin
                    and legal[action].get("safe_successor_fraction") > 0.25 # 1 = all immediate neighbors are safe
                    and (not action.place_bomb or (action.place_bomb and legal[action].get("escape_route_diversity") >= 0.25)) # 1 = all immediate neighbors have viable route to safety
                    )
        
        avoid_threat_candidates = {
            action: features
            for action, features in legal.items()
            if safe_in_future(action)
        }
        
        if not avoid_threat_candidates:
            return Status.FAILURE

        self.blackboard.set(BBKeys.Q_CANDIDATES, avoid_threat_candidates)
        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Q_AvoidThreatMoveSet" })
        return Status.SUCCESS


class AvoidThreat(ActionNode):
    """Choose an eligible action using the emergency evaluation profile."""
    def __init__(self, blackboard, profile):
        super().__init__("avoid_threat", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "AvoidThreat"})
        legal = legal_candidate_actions(model)
        safety_results = [assess_immediate_safety(model, action) for action in legal]
        eligible = [result.action for result in safety_results if result.eligible]
        _, future_danger = monster_reachable_layers(model)
        rankings = rank_actions(model, eligible, self.profile)
        self.blackboard.set(BBKeys.EVALUATION_BREAKDOWNS, rankings)
        self.blackboard.set(BBKeys.MONSTER_T2_THREAT_CELLS, future_danger)
        self.blackboard.set(
            BBKeys.CANDIDATE_SAFETY,
            [safety_result_dict(result) for result in safety_results],
        )
        best = next((result for result in rankings if not result.lethal), None)
        if best is None:
            return Status.FAILURE
        self.blackboard.set(BBKeys.SELECTED_ACTION, best.action)
        return Status.SUCCESS

## QLEARNING VERSION ================================================================
class Q_NavigateToExitMoveSet(QLearningActionNode):
    """
    [Q-Learning] Chooses and posts to blackboard a set of best actions for navigation 
    using A* for route context, then avoid an unsafe or risky next step.
    """
    def __init__(self, blackboard, profile):
        super().__init__("navigate_to_exit", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.exit_position is None or model.self_position is None:
            return Status.FAILURE
        
        # NOTE: no bombs here
        legal_actions = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS)
        legal_features = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS_FEATURES)
        legal = {action: features for action, features in zip(legal_actions, legal_features)}
                
        move_to_exit_candidates = {}
        
        def safe_action(action) -> bool:
                    return (legal[action].get("loss_next_update") != 1.0 # safe if the action doesn't result in loss
                            and legal[action].get("explosion_threat") != 1.0 # safe if action doesn't result in being in active explosion
                            and ((action.place_bomb and legal[action].get("bomb_escape_margin") >= 0) # safe if placed bomb and action has bomb escape margin
                                or not action.place_bomb and legal[action].get("active_bomb_escape_margin") >= 0) # safe if didn't place bomb and action has active bomb escape margin
                            )
        
        move_to_exit_candidates = {
            action: features
            for action, features in legal.items()
            if safe_action(action)
        }
        
        if not move_to_exit_candidates: # if no safe applicable moves, return failure
            return Status.FAILURE

        
        self.blackboard.set(BBKeys.Q_CANDIDATES, move_to_exit_candidates)
        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Q_NavigateToExitMoveSet"})
        
        return Status.SUCCESS

class NavigateToExit(ActionNode):
    """Use A* for route context, then avoid an unsafe or risky next step."""
    def __init__(self, blackboard, profile):
        super().__init__("navigate_to_exit", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.exit_position is None or model.self_position is None:
            return Status.FAILURE

        start = model.self_position
        goal = model.exit_position
        danger = immediate_lethal_positions(model) | monster_immediate_reachable_cells(model)
        _, future_danger = monster_reachable_layers(model)
        self.blackboard.set(BBKeys.MONSTER_THREAT_CELLS, danger)
        self.blackboard.set(BBKeys.DANGER_CELLS, danger)
        self.blackboard.set(BBKeys.MONSTER_T2_THREAT_CELLS, future_danger)
        legal = legal_candidate_actions(model)
        safety_results = [assess_immediate_safety(model, action) for action in legal]
        self.blackboard.set(BBKeys.POSSIBLE_ACTIONS, legal)
        self.blackboard.set(
            BBKeys.SAFE_ACTIONS,
            [result.action for result in safety_results if result.eligible],
        )
        self.blackboard.set(
            BBKeys.CANDIDATE_SAFETY,
            [safety_result_dict(result) for result in safety_results],
        )
        path = find_path(model, start, goal, forbidden_cells=danger)
        self.blackboard.set(BBKeys.PLANNED_PATH, path)

        if not path or len(path) < 2:
            return Status.FAILURE

        next_pos = path[1]
        dx = next_pos[0] - start[0]
        dy = next_pos[1] - start[1]
        action = AgentAction(dx, dy, False)
        safety_results = [assess_immediate_safety(model, candidate) for candidate in legal]
        eligible = [result.action for result in safety_results if result.eligible]
        path_result = assess_immediate_safety(model, action)
        if not path_result.eligible:
            rankings = rank_actions(model, eligible, self.profile)
        else:
            path_score = evaluate_action(model, action, self.profile)
            path_survivability = short_horizon_survivability(model, path_score.destination)
            if (
                path_score.future_monster_risk == 0
                and path_score.future_trap_risk == 0
                and path_survivability > 0
            ):
                rankings = [path_score]
            else:
                safer_actions = []
                for candidate in eligible:
                    candidate_position = (
                        model.self_position[0] + candidate.dx,
                        model.self_position[1] + candidate.dy,
                    )
                    if short_horizon_survivability(model, candidate_position) > 0:
                        safer_actions.append(candidate)
                rankings = rank_actions(model, safer_actions or eligible, self.profile)
        self.blackboard.set(BBKeys.EVALUATION_BREAKDOWNS, rankings)
        best = next((result for result in rankings if not result.lethal), None)
        if best is None:
            return Status.FAILURE
        action = best.action
        self.blackboard.set(BBKeys.SELECTED_ACTION, action)
        self.blackboard.set(BBKeys.DEBUG_INFO, {
            "active_behavior": "NavigateToExit",
            "path_length": len(path) - 1,
        })
        return Status.SUCCESS

class IsExitBlocked(ConditionNode):
    def __init__(self, blackboard):
        super().__init__("is_exit_blocked", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.exit_position is None or model.self_position is None:
            return Status.FAILURE

        return Status.SUCCESS if not find_path(model, model.self_position, model.exit_position) else Status.FAILURE

## TODO: UPDATE W/QLEARNING ================================================================
class Q_ExplodeWallMoveSet(QLearningActionNode):
    """[Q-Learning] Chooses and posts to blackboard a set of actions including exploding walls."""
    def __init__(self, blackboard):
        super().__init__("explode_wall", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE
        
        # return all possible actions + bomb placement
        legal_actions = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS)
        legal_features = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS_FEATURES)
        legal = {action: features for action, features in zip(legal_actions, legal_features)}
        
        def safe_and_explode_wall_progress(action) -> bool:
            """returns true if the action is safe and leads to exploding a wall (either by moving toward a potential breach or placing a bomb)"""
            return (legal[action].get("loss_next_update") != 1.0 # safe if the action doesn't result in loss
                    and legal[action].get("explosion_threat") != 1.0 # safe if action doesn't result in being in active explosion
                    and ((action.place_bomb and legal[action].get("bomb_escape_margin") >= 0) # safe if placed bomb and action has bomb escape margin
                        or not action.place_bomb and legal[action].get("active_bomb_escape_margin") >= 0) # safe if didn't place bomb and action has active bomb escape margin
                    and (legal[action].get("breach_site_approach_gain") > 0 if not action.place_bomb else False
                        or legal[action].get("bomb_wall_route_gain") > 0 if action.place_bomb else False)
            )

        explode_wall_actions = {
            action: features
            for action, features in legal.items()
            if safe_and_explode_wall_progress(action)
        }
                
        if not explode_wall_actions: # if no explode wall actions available, return failure
            return Status.FAILURE
        
        self.blackboard.set(BBKeys.Q_CANDIDATES, explode_wall_actions)
        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Q_ExplodeWallMoveSet"})

        return Status.SUCCESS


class ExplodeWall(ActionNode):
    def __init__(self, blackboard):
        super().__init__("explode_wall", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "ExplodeWall"})
        # The current framework emits a wall-hit event but does not clearly remove the wall, need to check if the wall is actually removed
        return Status.FAILURE

## TODO: FINISH ================================================================
class Q_FindSafeFallbackMoveSet(QLearningActionNode):
    """[Q-Learning] Chooses and posts to blackboard a set of immediately safe actions when normal navigation fails."""
    def __init__(self, blackboard, profile):
        super().__init__("find_safe_fallback", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE
        
        # return a set of safe fallback actions + bomb placement
        legal_actions = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS)
        legal_features = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS_FEATURES)
        legal = {action: features for action, features in zip(legal_actions, legal_features)}
        fallback_action_candidates = {}
        
        def safe_action(action) -> bool:
            return (legal[action].get("loss_next_update") != 1.0 # safe if the action doesn't result in loss
                    and legal[action].get("explosion_threat") != 1.0 # safe if action doesn't result in being in active explosion
                    and ((action.place_bomb and legal[action].get("bomb_escape_margin") >= 0) # safe if placed bomb and action has bomb escape margin
                        or not action.place_bomb and legal[action].get("active_bomb_escape_margin") >= 0) # safe if didn't place bomb and action has active bomb escape margin
                    )
        
        fallback_action_candidates = {
            action: features
            for action, features in legal.items()
            if safe_action(action)
        }
        
        if not fallback_action_candidates:
            return Status.FAILURE

        self.blackboard.set(BBKeys.Q_CANDIDATES, fallback_action_candidates)
        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Q_FindSafeFallbackMoveSet"})
        
        return Status.SUCCESS


class FindSafeFallback(ActionNode):
    """Choose the best immediately safe action when normal navigation fails."""
    def __init__(self, blackboard, profile):
        super().__init__("find_safe_fallback", blackboard)
        self.profile = profile

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "FindSafeFallback"})
        legal = legal_candidate_actions(model)
        safety_results = [assess_immediate_safety(model, action) for action in legal]
        eligible = [result.action for result in safety_results if result.eligible]
        rankings = rank_actions(model, eligible, self.profile)
        self.blackboard.set(BBKeys.EVALUATION_BREAKDOWNS, rankings)
        self.blackboard.set(
            BBKeys.CANDIDATE_SAFETY,
            [safety_result_dict(result) for result in safety_results],
        )
        best = next((result for result in rankings if not result.lethal), None)
        if best is None:
            return Status.FAILURE
        self.blackboard.set(BBKeys.SELECTED_ACTION, best.action)
        return Status.SUCCESS

## QLEARNING EDITION ================================================================
class Q_Wait(ActionNode):
    """Queue a valid no-movement action as the final fallback."""
    def __init__(self, blackboard):
        super().__init__("wait", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE
        
        legal_actions = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS)
        legal_features = self.blackboard.get(BBKeys.POSSIBLE_ACTIONS_FEATURES)
        legal = {action: features for action, features in zip(legal_actions, legal_features)}
        
        wait_action = AgentAction(0, 0, False)
        
        # TODO: account for the fact that wait might not be a legal action (somehow)
        wait_candidate = { # just gives wait and features
            action: features
            for action, features in legal.items()
            if action == wait_action
        }

        self.blackboard.set(BBKeys.Q_CANDIDATES, wait_candidate)
        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Q_Wait"})
        
        return Status.SUCCESS

class Wait(ActionNode):
    """Queue a valid no-movement action as the final fallback."""
    def __init__(self, blackboard):
        super().__init__("wait", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Wait"})
        self.blackboard.set(BBKeys.SELECTED_ACTION, AgentAction(0, 0, False))
        return Status.SUCCESS

## TODO: UPDATE W/QLEARNING ACTION NODES ================================================================
## NOTE: we want the action nodes to rank actions based on Q-learning weights. reuses the behavior tree structure &
##       safety helpers, but uses Q-aware nodes for feature scoring and action selection.
##       HOWEVER, currently, behavior tree functions as a sort of pre-filter for actions.
class QLearningRootController:
    def __init__(self, blackboard, profiles):
        self.blackboard = blackboard
        self.root = SelectorNode("root", blackboard)

        #safety branch for handling immediate and future dangers
        safety_selector = SelectorNode("safety_selector", blackboard)
        imediate_danger_branch = SequenceNode("immediate_danger", blackboard)
        imediate_danger_branch.add_child(IsImmediateDanger(blackboard))
        imediate_danger_branch.add_child(Q_ChooseSafeMoveSet(blackboard, profiles["emergency"]))

        future_danger_branch = SequenceNode("future_danger", blackboard)
        future_danger_branch.add_child(Q_IsDangerSoon(blackboard))
        future_danger_branch.add_child(Q_AvoidThreatMoveSet(blackboard, profiles["emergency"]))

        safety_selector.add_child(imediate_danger_branch)
        safety_selector.add_child(future_danger_branch)

        #navigation branch
        navigation_seq = SequenceNode("navigation_seq", blackboard)
        navigation_seq.add_child(Q_NavigateToExitMoveSet(blackboard, profiles["normal"]))

        #fallback branch for handling situations when no other actions are possible
        fallback_selector = SelectorNode("fallback_selector", blackboard)
        explosion_sequence = SequenceNode("explosion_sequence", blackboard)
        explosion_sequence.add_child(IsExitBlocked(blackboard))
        explosion_sequence.add_child(Q_ExplodeWallMoveSet(blackboard))
        fallback_selector.add_child(explosion_sequence)
        fallback_selector.add_child(Q_FindSafeFallbackMoveSet(blackboard, profiles["fallback"]))
        fallback_selector.add_child(Q_Wait(blackboard))

        self.root.add_child(safety_selector)
        self.root.add_child(navigation_seq)
        self.root.add_child(fallback_selector)

    def tick(self):
        return self.root.tick()

class RootController:
    def __init__(self, blackboard, profiles):
        self.blackboard = blackboard
        self.root = SelectorNode("root", blackboard)

        #safety branch for handling immediate and future dangers
        safety_selector = SelectorNode("safety_selector", blackboard)
        imediate_danger_branch = SequenceNode("immediate_danger", blackboard)
        imediate_danger_branch.add_child(IsImmediateDanger(blackboard))
        imediate_danger_branch.add_child(ChooseSafeMove(blackboard, profiles["emergency"]))

        future_danger_branch = SequenceNode("future_danger", blackboard)
        future_danger_branch.add_child(IsDangerSoon(blackboard))
        future_danger_branch.add_child(AvoidThreat(blackboard, profiles["emergency"]))

        safety_selector.add_child(imediate_danger_branch)
        safety_selector.add_child(future_danger_branch)

        #navigation branch
        navigation_seq = SequenceNode("navigation_seq", blackboard)
        navigation_seq.add_child(NavigateToExit(blackboard, profiles["normal"]))

        #fallback branch for handling situations when no other actions are possible
        fallback_selector = SelectorNode("fallback_selector", blackboard)
        explosion_sequence = SequenceNode("explosion_sequence", blackboard)
        explosion_sequence.add_child(IsExitBlocked(blackboard))
        explosion_sequence.add_child(ExplodeWall(blackboard))
        fallback_selector.add_child(explosion_sequence)
        fallback_selector.add_child(FindSafeFallback(blackboard, profiles["fallback"]))
        fallback_selector.add_child(Wait(blackboard))

        self.root.add_child(safety_selector)
        self.root.add_child(navigation_seq)
        self.root.add_child(fallback_selector)

    def tick(self):
        return self.root.tick()


class BombermanAgent(CharacterEntity):
    def __init__(self, name, avatar, x, y, weights=None):
        super().__init__(name, avatar, x, y)
        self.blackboard = BlackBoard()
        self.evaluation_profiles = profiles_from_mapping(weights)
        self.root = RootController(self.blackboard, self.evaluation_profiles)

    def do(self, wrld):
        model = WorldModel.from_sensed_world(wrld)
        self.blackboard.erase() #sets all keys values to None
        self.blackboard.set(BBKeys.WORLD_MODEL, model)

        exit_action = find_executable_exit_action(model)
        if exit_action is not None:
            self.blackboard.set(BBKeys.SELECTED_ACTION, exit_action)
            self.blackboard.set(BBKeys.DEBUG_INFO, {
                "active_behavior": "ImmediateExit",
                "immediate_exit_available": True,
                "immediate_exit_action": {
                    "dx": exit_action.dx,
                    "dy": exit_action.dy,
                    "place_bomb": exit_action.place_bomb,
                },
                "immediate_exit_executable": True,
                "selected_reason": "IMMEDIATE_EXIT_WIN",
            })
        else:
            self.root.tick()

        action = self.blackboard.get(BBKeys.SELECTED_ACTION)

        if action is None:
            self.move(0, 0)
            return

        self.move(action.dx, action.dy)
        if action.place_bomb:
            self.place_bomb()
