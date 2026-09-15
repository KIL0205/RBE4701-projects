import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.behavior_tree import (
    ActionNode,
    DecoratorNode,
    ParallelNode,
    SelectorNode,
    SequenceNode,
    Status,
)
from agent.black_board import BlackBoard


class StubNode(ActionNode):
    def __init__(self, result, blackboard=None):
        super().__init__("stub", blackboard)
        self.result = result
        self.calls = 0

    def tick(self):
        self.calls += 1
        self.status = self.result
        return self.result

    def _reset_local(self):
        self.calls = 0


class StubDecorator(DecoratorNode):
    def tick(self):
        self.status = self.child.tick()
        return self.status


def test_sequence_propagates_status_and_order():
    first = StubNode(Status.SUCCESS)
    second = StubNode(Status.RUNNING)
    third = StubNode(Status.SUCCESS)
    sequence = SequenceNode("sequence")
    sequence.add_child(first)
    sequence.add_child(second)
    sequence.add_child(third)

    assert sequence.tick() == Status.RUNNING
    assert [first.calls, second.calls, third.calls] == [1, 1, 0]


def test_selector_rechecks_high_priority_child_each_tick():
    high = StubNode(Status.FAILURE)
    fallback = StubNode(Status.SUCCESS)
    selector = SelectorNode("selector")
    selector.add_child(high)
    selector.add_child(fallback)

    assert selector.tick() == Status.SUCCESS
    high.result = Status.SUCCESS
    assert selector.tick() == Status.SUCCESS
    assert high.calls == 2
    assert fallback.calls == 1


def test_reset_clears_status_and_local_state_recursively():
    child = StubNode(Status.SUCCESS)
    sequence = SequenceNode("sequence")
    sequence.add_child(child)
    assert sequence.tick() == Status.SUCCESS
    assert sequence.status == Status.SUCCESS
    assert child.calls == 1

    sequence.reset()
    assert sequence.status is None
    assert child.status is None
    assert child.calls == 0


def test_blackboard_is_shared_with_children_and_decorator():
    board = BlackBoard()
    child = StubNode(Status.SUCCESS)
    sequence = SequenceNode("sequence", board)
    sequence.add_child(child)
    assert child.blackboard is board

    decorated_child = StubNode(Status.SUCCESS)
    decorator = StubDecorator("decorator", decorated_child, board)
    assert decorator.blackboard is board
    assert decorated_child.blackboard is board


def test_parallel_threshold_and_running_status():
    first = StubNode(Status.SUCCESS)
    second = StubNode(Status.RUNNING)
    parallel = ParallelNode("parallel", success_threshold=2)
    parallel.add_child(first)
    parallel.add_child(second)

    assert parallel.tick() == Status.RUNNING
    second.result = Status.SUCCESS
    assert parallel.tick() == Status.SUCCESS
