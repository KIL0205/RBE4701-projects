import math
from typing import Dict, List
from team01.agent.actions import AgentAction
from team01.agent.black_board import BlackBoardValue

# TODO: most typing is temporary
# TODO: might make more sense to combine this and minimax into one class...

class Expectimax():
    """
    Generic expectimax implementation for decision making in the Bomberman agent
    """
    def __init__(self, graph, depth: int):
        self.graph = graph
        self.max_depth = depth
        self.curr_depth = 0
        
    def expectimax():
        # TODO: implement... when i fix minimax
        pass