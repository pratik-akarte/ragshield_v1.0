"""
rate_limit_utils.py — reusable rate limiting for any API client in the app.

Two independent, composable pieces:

1. RateLimiter          A thread-safe + asyncio-safe sliding-window limiter.
                        Paces *outgoing* calls so you stay under N calls per
                        `period` seconds — the main defense against 429s.

2. retry_with_backoff   A decorator that retries a call with exponential
                        backoff (+ jitter) specifically when the call still
                        comes back as a rate-limit / 429 error (e.g. another
                        process shares the quota, or the limiter is looser
                        than the real cap).

Both work on sync *and* async functions or methods, and are provider-agnostic
— they don't care whether the wrapped call is Gemini, OpenAI, Anthropic, or a
plain `requests`/`httpx` call. No third-party imports required.

Typical usage
-------------
    from rate_limit_utils import get_rate_limiter, rate_limited, retry_with_backoff, wrap_method

    # Define the quota once, reuse the same limiter everywhere that hits it
    gemini_limiter = get_rate_limiter("gemini-flash-lite", max_calls=12, period=60)

    # A) Decorate your own function
    @rate_limited(gemini_limiter)
    @retry_with_backoff()
    def call_gemini(prompt):
        ...

    # B) Patch methods on a third-party class you don't want to subclass
    #    (e.g. DeepEval's GeminiModel) every time you need this:
    model = GeminiModel(...)
    wrap_method(model, "generate", limiter=gemini_limiter)
    wrap_method(model, "a_generate", limiter=gemini_limiter)
"""

from __future__ import annotations

import asyncio
import functools
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional


# --------------------------------------------------------------------------
# 1. Sliding-window rate limiter
# --------------------------------------------------------------------------

class RateLimiter:
    """Blocks until fewer than `max_calls` have fired in the last `period` seconds.

    Safe to call from threads (`acquire`) and from asyncio tasks (`a_acquire`).
    Share ONE instance across every call site hitting the same quota (same
    API key / model) — that's what `get_rate_limiter` below is for.
    """

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: deque = deque()
        self._lock = threading.Lock()
        self._alock = asyncio.Lock()

    def _wait_time(self) -> float:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > self.period:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            return self.period - (now - self._calls[0])
        return 0.0

    def acquire(self) -> None:
        with self._lock:
            wait = self._wait_time()
            while wait > 0:
                time.sleep(wait)
                wait = self._wait_time()
            self._calls.append(time.monotonic())

    async def a_acquire(self) -> None:
        async with self._alock:
            wait = self._wait_time()
            while wait > 0:
                await asyncio.sleep(wait)
                wait = self._wait_time()
            self._calls.append(time.monotonic())


# Named, process-wide limiters so unrelated modules share one quota bucket
# for the same underlying API limit, instead of each creating its own and
# collectively blowing past the real per-minute cap anyway.
_limiters: Dict[str, "RateLimiter"] = {}
_limiters_lock = threading.Lock()


def get_rate_limiter(name: str, max_calls: int, period: float = 60.0) -> RateLimiter:
    """Get (or create) a shared, process-wide named RateLimiter.

    Use the same `name` everywhere that hits the same quota, e.g.
    get_rate_limiter("gemini-flash-lite", max_calls=12, period=60). The
    max_calls/period you pass only take effect the first time `name` is
    created — later calls just return the existing limiter, so ideally
    define each quota once (e.g. in a small config module) and fetch it
    by name elsewhere.
    """
    with _limiters_lock:
        if name not in _limiters:
            _limiters[name] = RateLimiter(max_calls=max_calls, period=period)
        return _limiters[name]


def rate_limited(limiter: RateLimiter) -> Callable:
    """Decorator: pace a sync or async function/method through `limiter`."""

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                await limiter.a_acquire()
                return await fn(*args, **kwargs)
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            limiter.acquire()
            return fn(*args, **kwargs)
        return sync_wrapper

    return decorator


# --------------------------------------------------------------------------
# 2. Provider-agnostic 429 detection
# --------------------------------------------------------------------------

def is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort check for whether `exc` is a rate-limit / 429 error.

    Covers common shapes across providers without hard-importing any SDK
    (so this module has zero required dependencies):
      - google.api_core.exceptions.ResourceExhausted / TooManyRequests
      - openai.RateLimitError / anthropic.RateLimitError
      - requests/httpx-style errors with a .status_code, or .response.status_code
      - anything whose message mentions 429 / rate limit / quota
    """
    if type(exc).__name__ in {"ResourceExhausted", "TooManyRequests", "RateLimitError"}:
        return True

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "quota" in message


# --------------------------------------------------------------------------
# 3. Exponential-backoff retry (for when a 429 slips through anyway)
# --------------------------------------------------------------------------

def retry_with_backoff(
    max_attempts: int = 6,
    initial_seconds: float = 10.0,
    exp_base: float = 2.0,
    cap_seconds: float = 90.0,
    jitter: float = 0.2,
    should_retry: Callable[[BaseException], bool] = is_rate_limit_error,
) -> Callable:
    """Decorator: retry a sync or async call with exponential backoff.

    Only retries exceptions where `should_retry(exc)` is True (default:
    rate-limit/429-shaped errors) — anything else raises immediately, so
    real bugs don't get silently retried for minutes.
    Delay sequence: initial_seconds, *exp_base, *exp_base, ... capped at
    cap_seconds, each ± `jitter` fraction of random jitter so concurrent
    callers don't all retry in lockstep.
    """

    def _delay(attempt: int) -> float:
        base = min(initial_seconds * (exp_base ** attempt), cap_seconds)
        return base * (1 + random.uniform(-jitter, jitter))

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        if not should_retry(exc) or attempt == max_attempts - 1:
                            raise
                        await asyncio.sleep(_delay(attempt))
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if not should_retry(exc) or attempt == max_attempts - 1:
                        raise
                    time.sleep(_delay(attempt))
        return sync_wrapper

    return decorator


# --------------------------------------------------------------------------
# 4. Patch a method on an object/class you don't own (e.g. a third-party
#    model wrapper) instead of writing a new subclass every time.
# --------------------------------------------------------------------------

def wrap_method(
    target: Any,
    method_name: str,
    limiter: Optional[RateLimiter] = None,
    retry: bool = True,
    **retry_kwargs: Any,
) -> None:
    """Monkey-patch `target.method_name` in place with rate limiting + retry.

    `target` can be an instance or a class. Call this once per object —
    calling it twice on the same method re-wraps the already-wrapped
    version, which still works but adds redundant layers.
    """
    original = getattr(target, method_name)

    wrapped = original
    if retry:
        wrapped = retry_with_backoff(**retry_kwargs)(wrapped)
    if limiter is not None:
        wrapped = rate_limited(limiter)(wrapped)

    setattr(target, method_name, wrapped)