import math
from typing import Dict, List
from team01.agent.actions import AgentAction
from team01.agent.black_board import BlackBoardValue

# TODO: most typing is temporary
# TODO: I FORGOT THE KRIFFING ALPHA-BETA PRUNING OSIK OSIK OSIK

class Minimax():
    """
    Generic minimax implementation for decision making in the Bomberman agent.
    """
    def __init__(self, graph, depth: int):
        self.graph = graph
        self.max_depth = depth
        self.curr_depth = 0
        
    
    ### Minimax Functions
    def minimax(self, state) -> AgentAction:
        '''executes the minimax function on the given graph and returns an action'''
        self.curr_depth = 0 # initialize current depth, alpha, and beta
        alpha = -math.inf
        beta  =  math.inf
        return self.get_action(self.min_val(self.result(state), alpha, beta)) # aka return the index of the max choice returned by the min_val() fn given the result of a state and action


    def max_val(self, state, alpha: float, beta: float): # returns a utility value (optimal cell)
        '''returns the maximum value possible to achieve from the actions available for state'''
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth: # TODO: update with whatever the actual return val is here lol
            return self.utility(state)
        
        v = - math.inf
        for a in self.actions(state): 
            v = max(v, self.min_val(self.result(state, a), a))
        return v
    

    def min_val(self, state, alpha: float, beta: float): # returns a utility value (optimal cell)
        '''returns the maximum value possible to achieve from the actions available for state'''
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth: # TODO: update with whatever the actual return val is here lol
            return self.utility(state)
        
        v = math.inf
        for a in self.actions(state): 
            v = min(v, self.max_val(self.result(state, a), a)) 
        return v
    
    
    ### Supporting Functions
    def terminal_test(self, state) -> bool:
            if self.actions(state) == []: # TODO: replace with whatever the terminal test actually is
                return True
            return False
    
    
    def actions(self, state) -> List[AgentAction]: # TODO: update with actual method for getting possible actions from a state
            return [AgentAction(0,0,False)]
        
        
    def result(self, state): # TODO: update with actual method of getting the result of taking an action a at state
        return state
    
    
    def utility(self, state) -> float:  # TODO: update with whatever the actual utility ends up being
        return 0.0
    
    
    def get_action(self, val) -> AgentAction: # TODO: update with correct method of getting index
        return AgentAction(0,0,False)
    