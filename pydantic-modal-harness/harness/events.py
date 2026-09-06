"""
Per-run event bus.

Events are persisted in the store (so late subscribers can replay history)
and fanned out live to any open SSE connections through asyncio queues.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from .models import RunEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent | None]]] = defaultdict(set)

    def subscribe(self, run_id: str) -> asyncio.Queue[RunEvent | None]:
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        self._subscribers[run_id].add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[RunEvent | None]) -> None:
        self._subscribers[run_id].discard(queue)
        if not self._subscribers[run_id]:
            self._subscribers.pop(run_id, None)

    def publish(self, event: RunEvent) -> None:
        for queue in list(self._subscribers.get(event.run_id, ())):
            queue.put_nowait(event)

    def close(self, run_id: str) -> None:
        """Signal end-of-stream to every subscriber of a run."""
        for queue in list(self._subscribers.get(run_id, ())):
            queue.put_nowait(None)
