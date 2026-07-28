"""Retry policy for streaming LLM calls.

Design rationale
----------------
Retrying a *stream* is a different problem from retrying a plain request.
Once the first token has reached the user, silently restarting the call
would either duplicate text already rendered or interleave two partial
answers -- both worse than a clear error. The policy here therefore only
retries the connection / first-chunk phase, where nothing has been shown to
anyone yet. Any failure that happens *after* the first chunk is reported,
never retried.

Backoff is exponential with jitter rather than a fixed delay: if a provider
is rate-limiting several concurrent users at once, jittered retries spread
out instead of re-synchronising on the same schedule and re-tripping the
limit together.
"""

from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator, Callable

from src.utils.exceptions import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def stream_with_retry(
    open_stream: Callable[[], AsyncIterator[str]],
    *,
    max_retries: int,
    translate_error: Callable[[Exception], LLMError],
    base_delay_seconds: float = 0.5,
) -> AsyncIterator[str]:
    """Yield text chunks from a provider stream, retrying only before the first chunk.

    Args:
        open_stream: Zero-argument factory that opens a *fresh* provider
            stream on each call. Must be a factory rather than an
            already-open iterator, since a retry needs to start a genuinely
            new attempt rather than resume a failed one.
        max_retries: Number of additional attempts after the first.
        translate_error: Maps a raw provider-SDK exception to one of this
            application's :class:`LLMError` subclasses, so callers only ever
            see the application's own exception hierarchy.
        base_delay_seconds: Backoff base. Attempt *n* waits approximately
            ``base_delay_seconds * 2**n`` seconds plus jitter.

    Yields:
        Text fragments, in order, each exactly once.

    Raises:
        LLMError: If every attempt fails before a first chunk is produced,
            or if the stream fails after at least one chunk was already
            yielded to the caller.

    Examples:
        >>> async def flaky():
        ...     raise TimeoutError("boom")
        ...     yield "unreachable"  # pragma: no cover
        >>> async def consume():
        ...     async for _ in stream_with_retry(
        ...         flaky, max_retries=0, translate_error=lambda e: LLMError(str(e))
        ...     ):
        ...         pass
        >>> import asyncio; asyncio.run(consume())  # doctest: +SKIP
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        agen = open_stream()
        try:
            first_chunk = await agen.__anext__()
        except StopAsyncIteration:
            return  # Provider legitimately returned an empty stream.
        except Exception as exc:  # noqa: BLE001 -- translated immediately below
            last_error = exc
            translated = translate_error(exc)
            if attempt < max_retries:
                delay = base_delay_seconds * (2**attempt) + random.uniform(0, 0.25)
                logger.warning(
                    "LLM stream failed before first chunk, retrying",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "delay_seconds": round(delay, 2),
                        **translated.to_log_fields(),
                    },
                )
                await asyncio.sleep(delay)
                continue
            logger.error(
                "LLM stream failed before first chunk; retries exhausted",
                extra=translated.to_log_fields(),
            )
            raise translated from exc

        # The first chunk succeeded. Yield it, then hand off the remainder
        # of this same generator -- no further retries from here.
        yield first_chunk
        try:
            async for chunk in agen:
                yield chunk
        except Exception as exc:  # noqa: BLE001
            translated = translate_error(exc)
            logger.error(
                "LLM stream failed mid-response; not retrying to avoid "
                "duplicated or interleaved output",
                extra=translated.to_log_fields(),
            )
            raise translated from exc
        return

    if last_error is not None:  # pragma: no cover -- defensive; loop always returns/raises
        raise translate_error(last_error) from last_error
