from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional

from agent.black_board import BlackBoard


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class BTNode(ABC):
    def __init__(self, name: str, blackboard: Optional[BlackBoard] = None):
        self.name = name
        self.status: Optional[Status] = None
        self.blackboard = blackboard

    @abstractmethod
    def tick(self) -> Status:
        """Run one behavior-tree tick and return the status."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset this node and its children."""
        self.status = None
        self._reset_local()
        self.reset_children()

    def _reset_local(self) -> None:
        """Override for local state things (time counters, etc.)"""
        pass

    def reset_children(self) -> None:
        """Override for resetting children"""
        pass


class CompositeNode(BTNode):
    def __init__(self, name: str, blackboard: Optional[BlackBoard] = None):
        super().__init__(name, blackboard)
        self.children: List[BTNode] = []

    def add_child(self, child: BTNode) -> None:
        """Add a child node."""
        if hasattr(child, "blackboard") and child.blackboard is None:
            child.blackboard = self.blackboard
        self.children.append(child)

    def reset_children(self) -> None:
        """Reset all children."""
        for child in self.children:
            child.reset()


class SequenceNode(CompositeNode):
    """Success if all children succeed, fail on the first failure."""
    def __init__(self, name: str, blackboard: Optional[BlackBoard] = None):
        super().__init__(name, blackboard)

    def tick(self) -> Status:
        for child in self.children:
            s = child.tick()
            if s == Status.SUCCESS:
                continue

            if s == Status.RUNNING:
                self.status = Status.RUNNING
                return self.status
            
            # FAILURE
            self.status = Status.FAILURE
            return self.status

        #All children succeeded
        self.status = Status.SUCCESS
        return self.status
    
    def reset_children(self) -> None:
        """Reset all children and reset current child index."""
        super().reset_children()


class SelectorNode(CompositeNode):
    """Success if any child succeeds, fail if all children fail."""
    def __init__(self, name: str, blackboard: Optional[BlackBoard] = None):
        super().__init__(name, blackboard)

    def tick(self) -> Status:
        for child in self.children:
            s = child.tick()
            if s == Status.FAILURE:
                continue

            if s == Status.RUNNING:
                self.status = Status.RUNNING
                return self.status
            
            # SUCCESS
            self.status = Status.SUCCESS
            return self.status

        #All children failed
        self.status = Status.FAILURE
        return self.status
    
    def reset_children(self) -> None:
        """Reset all children and reset current child index."""
        super().reset_children()

class ParallelNode(CompositeNode):
    """Success if at least success_threshold children succeed,
       fail if any child fails (FAIL_FAST) or all children finish but not enough succeed (WAIT_ALL)."""
    def __init__(self, name: str, success_threshold: Optional[int] = None, blackboard: Optional[BlackBoard] = None):
        super().__init__(name, blackboard)
        self.success_threshold = success_threshold

    def tick(self) -> Status:
        success = 0
        running = 0
        failure = 0

        if len(self.children) == 0:
            self.status = Status.SUCCESS
            return self.status

        for child in self.children:
            status = child.tick()
            if status == Status.SUCCESS:
                success += 1
            elif status == Status.RUNNING:
                running += 1
            else:
                failure += 1

        threshold = len(self.children) if self.success_threshold is None else self.success_threshold
        threshold = max(1, min(threshold, len(self.children)))  # clamp threshold to [1, num_children]

        if success >= threshold:
            self.status = Status.SUCCESS
            return self.status
        elif failure > len(self.children) - threshold:
            self.status = Status.FAILURE
            return self.status
        else:
            self.status = Status.RUNNING
            return self.status


class DecoratorNode(BTNode):
    """Base class for decorator nodes (inverters, repeaters, etc.)."""
    def __init__(self, name: str, child: BTNode, blackboard: Optional[BlackBoard] = None):
        super().__init__(name, blackboard)
        self.child = child
        if self.child.blackboard is None:
            self.child.blackboard = blackboard

    def reset_children(self) -> None:
        self.child.reset()

class LeafNode(BTNode):
    """Base class for leaf nodes (actions and conditions)."""
    def reset_children(self) -> None:
        pass


class ActionNode(LeafNode):
    """Success if action succeeds, fail if action fails. Always runs from the start."""
    @abstractmethod
    def tick(self) -> Status:
        raise NotImplementedError


class ConditionNode(LeafNode):
    """Success if condition is true, fail if condition is false. Always runs from the start."""
    @abstractmethod
    def tick(self) -> Status:
        raise NotImplementedError