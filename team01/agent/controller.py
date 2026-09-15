import sys

sys.path.insert(0, '../../bomberman')

from entity import CharacterEntity

from agent.behavior_tree import ActionNode, ConditionNode, SelectorNode, SequenceNode, Status
from agent.black_board import BlackBoard
from agent.blackboard_keys import BBKeys
from agent.world_model import WorldModel


class IsImmediateDanger(ConditionNode):
    def __init__(self, blackboard):
        super().__init__("is_immediate_danger", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "IsImmediateDanger"})
        # TODO: Implement the logic for checking if immediate danger exists.
        return Status.FAILURE
        
class ChooseSafeMove(ActionNode):
    def __init__(self, blackboard):
        super().__init__("choose_safe_move", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "ChooseSafeMove"})
        # TODO: Implement the logic for choosing a safe move.
        return Status.FAILURE

class IsDangerSoon(ConditionNode):
    def __init__(self, blackboard):
        super().__init__("is_danger_soon", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        # TODO: Implement the logic for checking if danger is imminent.
        return Status.FAILURE
    
class AvoidThreat(ActionNode):
    def __init__(self, blackboard):
        super().__init__("avoid_threat", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "AvoidThreat"})
        # TODO: Implement the logic for avoiding potential threats.
        return Status.FAILURE
    
class NavigateToExit(ActionNode):
    def __init__(self, blackboard):
        super().__init__("navigate_to_exit", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.exit_position is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "NavigateToExit"})
        # TODO: Implement the logic for navigating to the exit.
        return Status.FAILURE

class IsExitBlocked(ConditionNode):
    def __init__(self, blackboard):
        super().__init__("is_exit_blocked", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.exit_position is None or model.self_position is None:
            return Status.FAILURE

        # TODO: Implement the logic for checking if the exit is blocked.
        return Status.FAILURE

class ExplodeWall(ActionNode):
    def __init__(self, blackboard):
        super().__init__("explode_wall", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "ExplodeWall"})
        # TODO: Implement the logic for exploding a wall.
        return Status.FAILURE

class FindSafeFallback(ActionNode):
    def __init__(self, blackboard):
        super().__init__("find_safe_fallback", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "FindSafeFallback"})
        # TODO: Implement the logic for finding a safe fallback position.
        return Status.FAILURE

class Wait(ActionNode):
    def __init__(self, blackboard):
        super().__init__("wait", blackboard)

    def tick(self):
        model = self.blackboard.get(BBKeys.WORLD_MODEL)
        if model is None or model.self_position is None:
            return Status.FAILURE

        self.blackboard.set(BBKeys.DEBUG_INFO, {"active_behavior": "Wait"})
        self.blackboard.set(BBKeys.SELECTED_ACTION, None)
        return Status.FAILURE


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

        self.root.tick()
        action = self.blackboard.get(BBKeys.SELECTED_ACTION)

        if action is None:
            self.move(0, 0)
            return

        self.move(action.dx, action.dy)
        if action.place_bomb:
            self.place_bomb()
