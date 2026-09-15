# This one is to just play through CLI

import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../team01')
from interactivecharacter import InteractiveCharacter

# Create the game
random.seed(random.randint(0, 1000))
g = Game.fromfile('map.txt')
g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            3, 5,     # position
))
g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 13,        # position
                                    1             # detection range
))

# g.add_character(BombermanAgent("me", "C", 0, 0))
g.add_character(InteractiveCharacter("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Run!
g.go(1)