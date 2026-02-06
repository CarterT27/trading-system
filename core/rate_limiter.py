from __future__ import annotations

import time
from collections import deque


class RequestRateLimiter:
    """Sliding-window API rate limiter."""

    def __init__(self, max_requests_per_minute: int = 190) -> None:
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        self.max_requests_per_minute = int(max_requests_per_minute)
        self.window_seconds = 60.0
        self.request_times: deque[float] = deque()

    def wait_for_slot(self) -> None:
        now = time.monotonic()
        self._evict_old(now)
        if len(self.request_times) >= self.max_requests_per_minute:
            earliest = self.request_times[0]
            sleep_for = max(0.0, earliest + self.window_seconds - now) + 0.02
            time.sleep(sleep_for)
            now = time.monotonic()
            self._evict_old(now)
        self.request_times.append(time.monotonic())

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
