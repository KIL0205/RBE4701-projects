import math
from typing import Dict, List
from team01.agent.actions import AgentAction
from team01.agent.black_board import BlackBoardValue
from team01.agent.world_model import WorldModel

# TODO: most typing is temporary
# TODO: test & double check depth limiting
# TODO: might want to pass around [v, state], where <state> is the closest state 
#       that leads to v, so it's easier to identify it later (or [v, action])

class AdversarialSearch():
    """
    Implementation of both alpha-beta limited minimax search as well as expectimax search. Initialized
    with a graph to search, as well as a depth limit. If no depth limit is set, the search is not depth limited.
    """
    
    ### Class Functions ------------------------------------------------------------------------
    def __init__(self, world, depth: float = math.inf) -> None:
        self.graph = world
        self.max_depth = depth
        self.curr_depth = 0
        
    def set_world(self, world) -> None:
        self.graph = world
        
    def set_max_depth(self, depth: float = math.inf) -> None:
        self.max_depth = depth
        
    ### Expectimax Functions -------------------------------------------------------------------
    def expectimax(self, state) -> AgentAction:
        """ 
        Generic expectimax implementation for decision making in the Bomberman agent. Executes 
        the minimax function on the given graph and returns an action 
        """
        self.curr_depth = 0 # initialize current depth, alpha, and beta
        return self.get_action(self.exp_val(state)) # aka return the index of the max choice returned by the min_val() fn given the result of a state and action
        
    def exp_val(self, state): # returns a utility value
        """ Returns the expected value of a node/state, using expectimax search. """
        self.curr_depth += 1 # increment depth
                
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth:
            return self.utility(state) # TODO: update with whatever the actual return val is here lol
        
        v = 0
        
        for a in self.actions(state): # update v to max value of possible actions, pruning if necessary
            p = self.probability(a)
            v = v + p * self.max_val_exp(self.result(state, a))
            
        return v
    
    def max_val_exp(self, state): # returns a utility value
        """ Returns the maximum value possible to achive from the actions available for the state, using expectimax search. """
        self.curr_depth += 1 # increment depth
        
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth:
            return self.utility(state) # TODO: update with whatever the actual return val is here lol
        
        v = - math.inf
        
        for a in self.actions(state): # update v to max value of possible actions, pruning if necessary
            v = max(v, self.exp_val(self.result(state, a), a))
            
        return v
        
    
    ### Minimax Functions ----------------------------------------------------------------------
    def minimax(self, state) -> AgentAction:
        """ 
        Generic minimax implementation with alpha-beta pruning for decision making in the Bomberman agent. Executes 
        the minimax function on the given graph and returns an action 
        """
        self.curr_depth = 0 # initialize current depth, alpha, and beta
        alpha = -math.inf
        beta  =  math.inf
        return self.get_action(self.min_val(state, alpha, beta)) # aka return the index of the max choice returned by the min_val() fn given the result of a state and action


    def max_val(self, state, alpha: float, beta: float): # returns a utility value (optimal cell)
        """ Returns the maximum value possible to achieve from the actions available for state """
        
        self.curr_depth += 1 # increment depth
        
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth:
            return self.utility(state) # TODO: update with whatever the actual return val is here lol
        
        v = - math.inf
        
        for a in self.actions(state): # update v to max value of possible actions, pruning if necessary
            v = max(v, self.min_val(self.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
            
        return v
    

    def min_val(self, state, alpha: float, beta: float): # returns a utility value (optimal cell)
        """ Returns the minimum value possible to achieve from the actions available for state """
        
        self.curr_depth += 1 # increment depth
        
        if self.terminal_test(state):
            return self.utility(state)
        
        # depth restricting
        if self.curr_depth == self.max_depth: # TODO: update with whatever the actual return val is here lol
            return self.utility(state)
        
        v = math.inf
        
        for a in self.actions(state): 
            v = min(v, self.max_val(self.result(state, a), alpha, beta)) 
            if v <= alpha:
                return v
            beta = min(beta, v)
            
        return v
    
    
    ### Supporting Functions -------------------------------------------------------------------
    def probability(self, a) -> float: # a is not an agent action, but an opponent action
        return 0.0 # TODO: update placeholder value
    
    def terminal_test(self, s) -> bool:
        """ Returns true if the state is a utility node, and false otherwise """
        if self.actions(s) == []: # TODO: replace with whatever the terminal test actually is
            return True
        return False
    
    
    def actions(self, s) -> List[AgentAction]: # TODO: update with actual method for getting possible actions from a state
        """ Returns a list of all possible actions for the given state """
        return [AgentAction(0,0,False)]
        # return graph.neighbors(s) ## if directly traversing the world
        
        
    def result(self, s, a: AgentAction): # TODO: update with actual method of getting the result of taking an action a at state
        """ Returns the state (or node) _s'_ that results from taking action _a_ in state (or node) _s_ """
        return s
    
    
    def utility(self, s) -> float:  # TODO: update with whatever the actual utility ends up being
        """ Returns the utility of node _s_"""
        return 0.0
    
    
    def get_action(self, val) -> AgentAction: # TODO: update with correct method of getting index. this is probably not the way to do this.
        """ Returns the action associated with the value _val_. Equivalent to argmax for this implementation. """
        return AgentAction(0,0,False)
    