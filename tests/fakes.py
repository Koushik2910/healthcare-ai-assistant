"""Test doubles for the LLM layer.

None of these touch a network. ``FakeLLMProvider`` implements the same
``LLMProvider`` contract as a real backend, so the chat service (Phase 4)
and the retry tests below can exercise realistic streaming and failure
behaviour without an API key, matching the "45 tests, zero API calls"
property already established in Phase 1.
"""

from __future__ import annotations

from typing import AsyncIterator

from src.llm.base import LLMProvider
from src.models.chat import Message
from src.models.llm import ProviderName


class FakeLLMProvider(LLMProvider):
    """A provider that replays a scripted sequence of chunks or raises.

    Args:
        chunks: Text fragments yielded in order on a successful call.
        fail_before_first_chunk: When set, raised instead of yielding
            anything -- exercises the "retry the connection" path.
        fail_after_chunks: When set, this many chunks are yielded and then
            the exception is raised -- exercises the "do not retry
            mid-stream" path.
        name: Overrides the reported provider identity; defaults to GEMINI
            since most tests don't care which one is faked.
    """

    def __init__(
        self,
        chunks: list[str] | None = None,
        *,
        fail_before_first_chunk: Exception | None = None,
        fail_after_chunks: Exception | None = None,
        name: ProviderName = ProviderName.GEMINI,
        model: str = "fake-model",
    ) -> None:
        super().__init__(
            model=model, temperature=0.3, max_output_tokens=1024, timeout_seconds=30.0
        )
        self.chunks = chunks or []
        self.fail_before_first_chunk = fail_before_first_chunk
        self.fail_after_chunks = fail_after_chunks
        self.name = name
        self.call_count = 0

    async def stream(
        self, messages: list[Message], *, system_prompt: str
    ) -> AsyncIterator[str]:
        self.call_count += 1
        if self.fail_before_first_chunk is not None:
            raise self.fail_before_first_chunk
        for chunk in self.chunks:
            yield chunk
        if self.fail_after_chunks is not None:
            raise self.fail_after_chunks


class FakeAsyncIterator:
    """Wraps a plain list as an async iterator, optionally failing partway.

    Used to emulate the async generator both the Gemini SDK and the OpenAI
    SDK return from their respective streaming calls, without depending on
    the real SDK's internal event types.
    """

    def __init__(
        self,
        items: list,
        *,
        fail_after: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self._items = list(items)
        self._index = 0
        self._fail_after = fail_after
        self._error = error

    def __aiter__(self) -> "FakeAsyncIterator":
        return self

    async def __anext__(self):
        if self._fail_after is not None and self._index == self._fail_after:
            assert self._error is not None
            raise self._error
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class _FakeGeminiEvent:
    """Mimics the ``.text`` attribute the Gemini SDK exposes per chunk."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGeminiModels:
    def __init__(
        self,
        events: list[_FakeGeminiEvent],
        *,
        fail_after: int | None,
        mid_stream_error: Exception | None,
        fail_first_n_opens: int,
        open_error: Exception | None,
    ) -> None:
        self._events = events
        self._fail_after = fail_after
        self._mid_stream_error = mid_stream_error
        self._fail_first_n_opens = fail_first_n_opens
        self._open_error = open_error
        self._open_attempts = 0

    async def generate_content_stream(self, *, model, contents, config):
        self._open_attempts += 1
        if self._open_attempts <= self._fail_first_n_opens:
            assert self._open_error is not None
            raise self._open_error
        return FakeAsyncIterator(
            self._events, fail_after=self._fail_after, error=self._mid_stream_error
        )


class FakeGeminiClient:
    """Fakes the ``client.aio.models.generate_content_stream`` surface.

    Args:
        texts: Text fragments the fake stream yields once a call succeeds.
        fail_first_n_opens: The first this-many calls raise ``open_error``
            instead of opening a stream at all -- tests the "retry before
            first chunk, then recover" path. ``0`` (default) means every
            call opens successfully.
        open_error: Exception raised during the first ``fail_first_n_opens``
            calls.
        fail_after: Index at which a successfully opened stream raises
            ``mid_stream_error`` instead of yielding its next fragment --
            tests the "do not retry mid-stream" path.
        mid_stream_error: Exception raised at ``fail_after``.
    """

    def __init__(
        self,
        texts: list[str],
        *,
        fail_first_n_opens: int = 0,
        open_error: Exception | None = None,
        fail_after: int | None = None,
        mid_stream_error: Exception | None = None,
    ) -> None:
        events = [_FakeGeminiEvent(text) for text in texts]

        class _Aio:
            def __init__(self) -> None:
                self.models = _FakeGeminiModels(
                    events,
                    fail_after=fail_after,
                    mid_stream_error=mid_stream_error,
                    fail_first_n_opens=fail_first_n_opens,
                    open_error=open_error,
                )

        self.aio = _Aio()


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _FakeDelta(content)


class _FakeChatCompletionChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)] if content is not None else []


class FakeOpenAICompletions:
    def __init__(
        self,
        texts: list[str],
        *,
        fail_after: int | None,
        mid_stream_error: Exception | None,
        fail_first_n_opens: int,
        open_error: Exception | None,
    ) -> None:
        self._chunks = [_FakeChatCompletionChunk(text) for text in texts]
        self._fail_after = fail_after
        self._mid_stream_error = mid_stream_error
        self._fail_first_n_opens = fail_first_n_opens
        self._open_error = open_error
        self._open_attempts = 0

    async def create(self, **kwargs):
        self._open_attempts += 1
        if self._open_attempts <= self._fail_first_n_opens:
            assert self._open_error is not None
            raise self._open_error
        return FakeAsyncIterator(
            self._chunks, fail_after=self._fail_after, error=self._mid_stream_error
        )


class FakeOpenAIClient:
    """Fakes the ``client.chat.completions.create(..., stream=True)`` surface.

    Shared by the Groq and OpenRouter tests since both subclass
    :class:`~src.llm.openai_compatible.OpenAICompatibleProvider`. See
    :class:`FakeGeminiClient` for the meaning of each parameter -- the two
    fakes are kept symmetric on purpose.
    """

    def __init__(
        self,
        texts: list[str],
        *,
        fail_first_n_opens: int = 0,
        open_error: Exception | None = None,
        fail_after: int | None = None,
        mid_stream_error: Exception | None = None,
    ) -> None:
        class _Chat:
            def __init__(self) -> None:
                self.completions = FakeOpenAICompletions(
                    texts,
                    fail_after=fail_after,
                    mid_stream_error=mid_stream_error,
                    fail_first_n_opens=fail_first_n_opens,
                    open_error=open_error,
                )

        self.chat = _Chat()
