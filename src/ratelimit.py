import time
import random


class RateLimiter:
    """Simple token-bucket rate limiter that enforces requests-per-second."""

    def __init__(self, rps):
        self.min_interval = 1.0 / rps
        self.last_request = 0.0

    def wait(self):
        """Block until it's safe to send the next request."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


# Pre-configured limiters matching EagleView rate limits (with ~10% headroom)
auth_limiter = RateLimiter(3.5)       # limit: 4 rps
discovery_limiter = RateLimiter(4.5)  # limit: 5 rps
images_limiter = RateLimiter(4.5)     # limit: 5 rps
tiles_limiter = RateLimiter(270)      # limit: 300 rps (for future tiling)


def retry_with_backoff(fn, max_retries=5):
    """Call fn(), retrying on 429/5xx with exponential backoff + jitter."""
    for attempt in range(max_retries):
        resp = fn()
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429 or resp.status_code >= 500:
            delay = min(30, random.uniform(2 ** attempt, 2 ** (attempt + 1)))
            print(f"  [retry] {resp.status_code}, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
