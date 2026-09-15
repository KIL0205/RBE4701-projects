# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster

# TODO This is your code!
sys.path.insert(1, '../team01')
from testcharacter import TestCharacter
from agent.controller import BombermanAgent

# Create the game
random.seed(random.randint(0, 1000))
g = Game.fromfile('map.txt')
g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            3, 9      # position
))

g.add_character(BombermanAgent("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Run!
g.go(1)
