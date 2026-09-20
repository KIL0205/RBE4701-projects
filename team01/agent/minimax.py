import math
from typing import Dict, List
from team01.agent.actions import AgentAction
from team01.agent.black_board import BlackBoardValue

# TODO: most typing is temporary
# TODO: might want to pass around [v, state], where <state> is the closest state 
#       that leads to v, so it's easier to identify it later

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
        return self.get_action(self.min_val(state, alpha, beta)) # aka return the index of the max choice returned by the min_val() fn given the result of a state and action


    def max_val(self, state, alpha: float, beta: float): # returns a utility value (optimal cell)
        """ Returns the maximum value possible to achieve from the actions available for state """
        
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth: # TODO: update with whatever the actual return val is here lol
            return self.utility(state)
        
        v = - math.inf
        
        for a in self.actions(state): # update v to max value of possible actions, pruning if necessary
            v = max(v, self.min_val(self.result(state, a), a))
            if v >= beta:
                return v
            alpha = max(alpha, v)
            
        return v
    

    def min_val(self, state, alpha: float, beta: float): # returns a utility value (optimal cell)
        """ Returns the minimum value possible to achieve from the actions available for state """
        
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth: # TODO: update with whatever the actual return val is here lol
            return self.utility(state)
        
        v = math.inf
        
        for a in self.actions(state): 
            v = min(v, self.max_val(self.result(state, a), a)) 
            if v <= alpha:
                return v
            beta = min(beta, v)
            
        return v
    
    
    ### Supporting Functions
    def terminal_test(self, s) -> bool:
        """ Returns true if the state is a utility node, and false otherwise """
        if self.actions(s) == []: # TODO: replace with whatever the terminal test actually is
            return True
        return False
    
    
    def actions(self, s) -> List[AgentAction]: # TODO: update with actual method for getting possible actions from a state
        """ Returns a list of all possible actions for the given state """
        return [AgentAction(0,0,False)]
        
        
    def result(self, s, a: AgentAction): # TODO: update with actual method of getting the result of taking an action a at state
        """ Returns the state (or node) _s'_ that results from taking action _a_ in state (or node) _s_ """
        return s
    
    
    def utility(self, s) -> float:  # TODO: update with whatever the actual utility ends up being
        """ Returns the utility of node _s_"""
        return 0.0
    
    
    def get_action(self, val) -> AgentAction: # TODO: update with correct method of getting index. this is probably not the way to do this.
        """ Returns the action associated with the value _val_ """
        return AgentAction(0,0,False)
    