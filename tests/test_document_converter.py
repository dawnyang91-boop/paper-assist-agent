import pytest

from DocumentConverter import _embed_with_retry


class FakeEmbeddingResponse:
    data = [type("EmbeddingData", (), {"embedding": [0.1, 0.2, 0.3]})()]


class FlakyEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary embedding failure")
        return FakeEmbeddingResponse()


class AlwaysFailEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise RuntimeError("still failing")


class FakeClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def test_embed_with_retry_returns_after_transient_failures():
    embeddings = FlakyEmbeddings()
    client = FakeClient(embeddings)

    response = _embed_with_retry(
        client=client,
        embed_kwargs={"input": "hello", "model": "fake"},
        max_try=3,
        wait_seconds=0,
        backoff=1,
    )

    assert response.data[0].embedding == [0.1, 0.2, 0.3]
    assert embeddings.calls == 3


def test_embed_with_retry_raises_after_max_try():
    embeddings = AlwaysFailEmbeddings()
    client = FakeClient(embeddings)

    with pytest.raises(RuntimeError, match="still failing"):
        _embed_with_retry(
            client=client,
            embed_kwargs={"input": "hello", "model": "fake"},
            max_try=2,
            wait_seconds=0,
            backoff=1,
        )

    assert embeddings.calls == 2
