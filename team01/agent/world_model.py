from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Position = Tuple[int, int]

@dataclass
class WorldModel:
    width: int
    height: int
    exit_position: Optional[Position] = None
    self_position: Optional[Position] = None
    walls: Set[Position] = field(default_factory=set)
    bombs: Set[Position] = field(default_factory=set)
    explosions: Set[Position] = field(default_factory=set)
    monsters: Set[Position] = field(default_factory=set)
    characters: Dict[str, Position] = field(default_factory=dict)

    @classmethod
    def from_sensed_world(cls, wrld) -> "WorldModel":
        model = cls(wrld.width(), wrld.height())

        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.wall_at(x, y):
                    model.walls.add((x, y))

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

        for k, b in wrld.bombs.items():
            model.bombs.add((b.x, b.y))

        for k, e in wrld.explosions.items():
            model.explosions.add((e.x, e.y))

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

    def monster_positions(self) -> List[Position]:
        return list(self.monsters)

    def bomb_positions(self) -> List[Position]:
        return list(self.bombs)

    def explosion_positions(self) -> List[Position]:
        return list(self.explosions)
