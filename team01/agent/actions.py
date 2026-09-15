from dataclasses import dataclass


@dataclass(frozen=True)
class AgentAction:
    dx: int
    dy: int
    place_bomb: bool = False
