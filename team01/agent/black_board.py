"""Thread-safe shared state blackboard used by BT and FSM coordination."""

from typing import Any, Dict, List
from threading import Lock


BlackBoardValue = Any


class BlackBoard:
    """Concurrent key-value store for cross-node coordination.

    The blackboard is shared between behavior tree nodes, state machine logic,
    and runtime I/O nodes. Values are intentionally heterogeneous.
    """

    def __init__(self) -> None:
        """Initialize an empty blackboard with an internal lock."""
        self._data: Dict[str, BlackBoardValue] = {}
        self._lock = Lock()

    def set(self, key: str, value: Any) -> None:
        """Set a value in the blackboard."""
        with self._lock:
            self._data[key] = value

    def get(self, key: str) -> BlackBoardValue | None:
        """Get a value from the blackboard. Returns None if the key does not exist."""
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> None:
        """Delete a value from the blackboard."""
        with self._lock:
            if key in self._data:
                del self._data[key]

    def clear(self) -> None:
        """Clear all data from the blackboard."""
        with self._lock:
            self._data.clear()

    def keys(self) -> List[str]:
        """Return a list of all keys in the blackboard."""
        with self._lock:
            return list(self._data.keys())

    def __str__(self) -> str:
        return f"BlackBoard({self._data})"