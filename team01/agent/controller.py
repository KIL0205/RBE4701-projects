import sys

sys.path.insert(0, '../../bomberman')

from entity import CharacterEntity

from .behavior_tree import ActionNode, ConditionNode, SelectorNode, SequenceNode, Status
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
)
from .navigation import find_path
from .evaluation import (
    EMERGENCY_ESCAPE,
    FALLBACK,
    NORMAL_NAVIGATION,
    bomb_blast_cells,
    evaluate_action,
    immediate_lethal_positions,
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
        
class ChooseSafeMove(ActionNode):
    """Choose the highest-scoring action after immediate safety filtering."""
    def __init__(self, blackboard):
        super().__init__("choose_safe_move", blackboard)

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
        rankings = rank_actions(model, eligible, EMERGENCY_ESCAPE)

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
    
class AvoidThreat(ActionNode):
    """Choose an eligible action using the emergency evaluation profile."""
    def __init__(self, blackboard):
        super().__init__("avoid_threat", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "AvoidThreat"})
        legal = legal_candidate_actions(model)
        safety_results = [assess_immediate_safety(model, action) for action in legal]
        eligible = [result.action for result in safety_results if result.eligible]
        _, future_danger = monster_reachable_layers(model)
        rankings = rank_actions(model, eligible, EMERGENCY_ESCAPE)
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
    
class NavigateToExit(ActionNode):
    """Use A* for route context, then avoid an unsafe or risky next step."""
    def __init__(self, blackboard):
        super().__init__("navigate_to_exit", blackboard)

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
            rankings = rank_actions(model, eligible, NORMAL_NAVIGATION)
        else:
            path_score = evaluate_action(model, action, NORMAL_NAVIGATION)
            if path_score.future_monster_risk == 0 and path_score.future_trap_risk == 0:
                rankings = [path_score]
            else:
                rankings = rank_actions(model, eligible, NORMAL_NAVIGATION)
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

class FindSafeFallback(ActionNode):
    """Choose the best immediately safe action when normal navigation fails."""
    def __init__(self, blackboard):
        super().__init__("find_safe_fallback", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "FindSafeFallback"})
        legal = legal_candidate_actions(model)
        safety_results = [assess_immediate_safety(model, action) for action in legal]
        eligible = [result.action for result in safety_results if result.eligible]
        rankings = rank_actions(model, eligible, FALLBACK)
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


class RootController:
    def __init__(self, blackboard):
        self.blackboard = blackboard
        self.root = SelectorNode("root", blackboard)

        #safety branch for handling immediate and future dangers
        safety_selector = SelectorNode("safety_selector", blackboard)
        imediate_danger_branch = SequenceNode("immediate_danger", blackboard)
        imediate_danger_branch.add_child(IsImmediateDanger(blackboard))
        imediate_danger_branch.add_child(ChooseSafeMove(blackboard))

        future_danger_branch = SequenceNode("future_danger", blackboard)
        future_danger_branch.add_child(IsDangerSoon(blackboard))
        future_danger_branch.add_child(AvoidThreat(blackboard))

        safety_selector.add_child(imediate_danger_branch)
        safety_selector.add_child(future_danger_branch)

        #navigation branch
        navigation_seq = SequenceNode("navigation_seq", blackboard)
        navigation_seq.add_child(NavigateToExit(blackboard))

        #fallback branch for handling situations when no other actions are possible
        fallback_selector = SelectorNode("fallback_selector", blackboard)
        explosion_sequence = SequenceNode("explosion_sequence", blackboard)
        explosion_sequence.add_child(IsExitBlocked(blackboard))
        explosion_sequence.add_child(ExplodeWall(blackboard))
        fallback_selector.add_child(explosion_sequence)
        fallback_selector.add_child(FindSafeFallback(blackboard))
        fallback_selector.add_child(Wait(blackboard))

        self.root.add_child(safety_selector)
        self.root.add_child(navigation_seq)
        self.root.add_child(fallback_selector)

    def tick(self):
        return self.root.tick()


class BombermanAgent(CharacterEntity):
    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.blackboard = BlackBoard()
        self.root = RootController(self.blackboard)

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
