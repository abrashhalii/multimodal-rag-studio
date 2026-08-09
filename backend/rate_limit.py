"""
Client-side rate limiting.

Free-tier Gemini gives 15 requests/minute and 500/day. Building a summary tree
means ~50 calls in a burst, which without pacing produces a wall of 429s and
then exponential backoff that takes longer than just waiting properly.

A token bucket paces outbound calls: tokens refill at a steady rate, and a
burst is allowed up to the bucket's capacity. This is the standard choice for
outbound throttling because it smooths sustained load while still permitting
short bursts - unlike a fixed window, which allows 2x the limit across a window
boundary, and unlike a leaky bucket, which forbids bursts entirely.

Thread-safe: FastAPI serves requests on a thread pool, so two chat requests can
call the LLM concurrently.
"""

import threading
import time

import config


class TokenBucket:
    """
    CAPACITY IS 1 BY DEFAULT, AND THAT MATTERS.

    The obvious reading of "15 requests per minute" is "a bucket holding 15
    tokens". That is wrong against a provider-side fixed window: a full bucket
    lets you fire all 15 instantly, which spends the entire minute's quota in
    two seconds, and request 16 is rejected before the window rolls over.

    Observed exactly that - 16 chapter summaries went out, then
    429 "limit: 15, retry in 39s".

    With capacity 1 there is no burst: requests are spaced 60/RPM seconds apart,
    which is what a hard provider cap actually requires.
    """

    def __init__(self, rate_per_minute: int, capacity: int = 1):
        self.rate = max(1, rate_per_minute) / 60.0        # tokens per second
        self.capacity = max(1, capacity)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.waits = 0
        self.total_wait = 0.0
        self.requests = 0
        self.quota_errors = 0

    def acquire(self, tokens: int = 1) -> float:
        """Block until `tokens` are available. Returns seconds spent waiting."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.requests += 1
                    return waited
                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate
            self.waits += 1
            self.total_wait += sleep_for
            waited += sleep_for
            time.sleep(min(sleep_for, 5.0))

    def stats(self) -> dict:
        return {
            "rate_per_minute": round(self.rate * 60),
            "capacity": self.capacity,
            "tokens_available": round(self._tokens, 2),
            "requests_made": self.requests,
            "times_throttled": self.waits,
            "total_wait_seconds": round(self.total_wait, 1),
            "quota_errors_recovered": self.quota_errors,
        }


_bucket = None


def get_bucket() -> TokenBucket:
    global _bucket
    if _bucket is None:
        # Run a little under the advertised ceiling. The provider counts in its
        # own window, which is not aligned with ours, so pacing exactly at the
        # limit still collides at window boundaries.
        effective = max(1, int(config.LLM_RPM * config.RATE_LIMIT_SAFETY))
        _bucket = TokenBucket(effective, capacity=1)
    return _bucket


def retry_on_quota(fn, *, attempts: int = 4, label: str = ""):
    """Run fn(), honouring the server's own retryDelay on a 429.

    The provider tells us exactly how long to wait - parsing that beats guessing
    with pure exponential backoff, and it is the difference between recovering
    in 26 seconds and hammering a closed door.
    """
    import re as _re
    last = None
    for attempt in range(attempts):
        throttle(label)
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            last = e
            get_bucket().quota_errors += 1
            m = _re.search(r"retry in ([0-9.]+)s", msg) or \
                _re.search(r"'retryDelay': '(\d+)s'", msg)
            wait = float(m.group(1)) + 1 if m else (5 * (2 ** attempt))
            wait = min(wait, 65)
            if config.DEBUG_LOG:
                print(f"[ratelimit] 429 on {label or 'request'}; "
                      f"waiting {wait:.0f}s (attempt {attempt + 1}/{attempts})")
            time.sleep(wait)
    raise last


def throttle(label: str = "") -> None:
    """Call immediately before every outbound LLM request."""
    waited = get_bucket().acquire()
    if waited > 0.1 and config.DEBUG_LOG:
        print(f"[ratelimit] waited {waited:.1f}s before {label or 'request'}")
