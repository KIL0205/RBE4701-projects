from typing import List, Tuple

from agent.world_model import WorldModel

import heapq
import math

Position = Tuple[int, int]


def find_path(model: WorldModel, start: Position, goal: Position) -> List[Position]:
    if start == goal:
        print("start same as goal")
        return []
    if model.neighbors[start] == []:
        print("Nowhere to traverse")
        return []

    came_from = set()
    cost_so_far = set()
    came_from[start] = None
    cost_so_far[start] = 0
    gx, gy = goal
    monster_worry = 1
    
    frontier = []
    heapq.heappush(frontier, (0,start))
    while frontier:
        prio, curr = heapq.heappop(frontier)

        # Path reconstruction
        if curr == goal:
            A_star_path = []
            while came_from[curr] != None:
                A_star_path.insert(0,curr)
                curr = came_from[curr]
            print("A* path generated")
            return A_star_path

        # Using neighbors to grab traversable tiles (In-bounds, Not wall, Not bomb)
        # Will need future updates to assign weight to walls instead
        neighbors = model.neighbors[curr]
        for node in neighbors:
            # Sets the travel distance to the neighbor as the cost
            cx, cy = curr
            nx, ny = node
            travel = math.sqrt(2)
            if ((abs(nx - cx) == 1) and (abs(ny - cy) == 0)) or ((abs(nx - cx) == 0) and (abs(ny - cy) == 1)):
                travel = 1
            # Updates to total costand checks if the node is worth visiting from here
            new_cost = cost_so_far[curr] + travel
            if node not in cost_so_far or new_cost < cost_so_far[node]:
                cost_so_far[node] = new_cost
                
                # Distance to each monster from node
                monster_dist = 0
                for mx, my in model.monster_positions:
                    monster_dist = 1/(math.sqrt((mx-nx)**2 + (my-ny)**2)) + monster_dist
                # Priority is for the frontier heapq; 
                # The cost to reach node + the straight-line distance to the goal from node + monster priority
                priority = new_cost + math.sqrt((gx-nx)**2 + (gy-ny)**2) + (monster_dist * monster_worry)
                heapq.heappush(frontier, (priority,node))
                # We went to node from curr (best option)
                came_from[node] = curr
    print("Exit is blocked")
    return [] # Tuple[-math.inf, math.inf]