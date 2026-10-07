from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Position = Tuple[int, int]


@dataclass
class WorldModel:
    width: int
    height: int
    bomb_time: int = -1
    explosion_duration: int = -1
    explosion_range: int = -1
    exit_position: Optional[Position] = None
    self_position: Optional[Position] = None
    walls: Set[Position] = field(default_factory=set)
    need_wall_groups: bool = True
    wallgroups: list[list[Position]] = field(default_factory=list)
    bombs: Set[Position] = field(default_factory=set)
    bomb_timers: Dict[Position, int] = field(default_factory=dict)
    explosions: Set[Position] = field(default_factory=set)
    explosion_timers: Dict[Position, int] = field(default_factory=dict)
    monsters: Set[Position] = field(default_factory=set)
    characters: Dict[str, Position] = field(default_factory=dict)

    @classmethod
    def from_sensed_world(cls, wrld) -> "WorldModel":
        """Build a current factual snapshot from the sensed framework world."""
        model = cls(
            wrld.width(),
            wrld.height(),
            bomb_time=wrld.bomb_time,
            explosion_duration=wrld.expl_duration,
            explosion_range=wrld.expl_range,
        )

        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.wall_at(x, y):
                    model.walls.add((x, y))

        if model.need_wall_groups:
            model.wallgroups = model.get_wall_groups()
            model.need_wall_groups = False

        if wrld.exitcell is not None:
            model.exit_position = wrld.exitcell

        for k, characterList in wrld.characters.items():
            for c in characterList:
                model.characters[c.name] = (c.x, c.y)
                if c.name == "me":
                    model.self_position = (c.x, c.y)

        for k, monsterList in wrld.monsters.items():
            for m in monsterList:
                model.monsters.add((m.x, m.y))

        for bomb in wrld.bombs.values():
            position = (bomb.x, bomb.y)
            model.bombs.add(position)
            model.bomb_timers[position] = bomb.timer

        for explosion in wrld.explosions.values():
            position = (explosion.x, explosion.y)
            model.explosions.add(position)
            model.explosion_timers[position] = explosion.timer

        return model

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, position: Position) -> bool:
        return position in self.walls

    def is_traversable(self, position: Position) -> bool:
        if not self.in_bounds(position):
            return False
        if self.is_wall(position):
            return False
        if position in self.bombs:
            return False
        return True

    def neighbors(self, current_position: Position) -> List[Position]:
        """
        Returns the traversable neighboring cells (including diagonals) of the given position in the world model.
        Traversable cells are those that are within bounds and not blocked by walls or bombs.
        :param current_position [(int, int)] The coordinate in the grid.
        :return        [[(int,int)]] A list of neighboring cells.
        """
        x, y = current_position
        moves = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                next_pos = (x + dx, y + dy)
                if self.is_traversable(next_pos):
                    moves.append(next_pos)
        return moves

    def wall_neighbors(self, current_position: Position) -> List[Position]:
        """
        Returns the NSEW neighboring cells occupied by walls of the given position in the world model.
        :param current_position [(int, int)] The coordinate in the grid.
        :return        [[int,int]] A list of neighboring cells.
        """
        x, y = current_position
        walls = []
        pos = [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]
        for n in pos:
            if self.is_wall(n):
                walls.append(n)
        return walls

    def get_wall_groups(self) -> list[list[Position]]:
        """Returns groups of adjacent walls, should run once at start.
        Used to log all starting wall groups for loop detection"""
        ret_wall_groups = [[]]
        walls = self.walls
        while walls:
            start_wall = walls(0)
            wall_group = []
            group_queue = []
            group_queue.insert(start_wall)
            while group_queue:
                group_queue_copy = group_queue
                for w in group_queue_copy:
                    if w not in wall_group:
                        wall_group.insert(w)
                        if w in walls:
                            walls.remove(w)
                    wall_neighbors = self.wall_neighbors(w)
                    for wn in wall_neighbors:
                        if wn not in wall_group:
                            wall_group.insert(wn)
                            group_queue.insert(wn)
                            if wn in walls:
                                walls.remove(wn)
                    group_queue.remove(w)
            ret_wall_groups.insert(wall_group)
        return ret_wall_groups

    def monster_positions(self) -> List[Position]:
        return list(self.monsters)

    def bomb_positions(self) -> List[Position]:
        return list(self.bombs)

    def explosion_positions(self) -> List[Position]:
        return list(self.explosions)
