"""Offline tests for the OpenAlex client's error messages.

No network: `requests.get` is monkeypatched with tiny fake responses. What is
being checked is which *advice* a given HTTP status produces — a corporate
proxy answering 403 must not be reported as a mistake in the query.
"""

from __future__ import annotations

import pytest

from cda.diagnose import openalex


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, body: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = body

    def json(self):
        if self._payload is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


def _serve(monkeypatch, response, calls: list | None = None):
    """Answer every request with ``response`` and never really sleep."""

    def fake_get(url, params=None, timeout=None):
        if calls is not None:
            calls.append(url)
        return response

    monkeypatch.setattr(openalex.requests, "get", fake_get)
    monkeypatch.setattr(openalex.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("OPENALEX_MAILTO", "someone@example.org")


def test_ok_response_returns_the_payload(monkeypatch):
    _serve(monkeypatch, _FakeResponse(200, {"results": [{"id": "W1"}]}))
    assert openalex.get_json("works") == {"results": [{"id": "W1"}]}


@pytest.mark.parametrize("status", [401, 403, 407])
def test_auth_statuses_blame_the_network_not_the_query(monkeypatch, status):
    """OpenAlex needs no credentials, so 401/403/407 is a proxy, VPN, or
    firewall answering in its place — not a malformed query."""
    _serve(monkeypatch, _FakeResponse(status, body="<html>Blocked</html>"))
    with pytest.raises(openalex.OpenAlexError) as caught:
        openalex.get_json("works")
    message = str(caught.value)
    assert "proxy" in message
    assert "api.openalex.org" in message
    assert "query parameters" not in message


@pytest.mark.parametrize("status", [400, 404])
def test_client_error_statuses_keep_the_query_advice(monkeypatch, status):
    """A genuine bad request must still point at the query, not at the
    network — the fix must not swallow real bugs."""
    _serve(monkeypatch, _FakeResponse(status))
    with pytest.raises(openalex.OpenAlexError) as caught:
        openalex.get_json("works")
    message = str(caught.value)
    assert "query parameters" in message
    assert "docs.openalex.org" in message


def test_non_json_200_names_the_captive_portal(monkeypatch):
    _serve(monkeypatch, _FakeResponse(200, payload=None, body="<html>Sign in</html>"))
    with pytest.raises(openalex.OpenAlexError) as caught:
        openalex.get_json("works")
    assert "proxy" in str(caught.value)


def test_repeated_5xx_retries_then_names_the_network(monkeypatch):
    """A gateway (or an outage) answering 5xx exhausts the retries; the final
    message must name the possibilities and a way to tell them apart."""
    calls: list[str] = []
    _serve(monkeypatch, _FakeResponse(503), calls)
    with pytest.raises(openalex.OpenAlexError) as caught:
        openalex.get_json("works", max_retries=3)
    assert len(calls) == 3
    message = str(caught.value)
    assert "proxy" in message or "firewall" in message
    assert "https://api.openalex.org" in message


# --- DOI normalization on the diagnose path -----------------------------------


def test_works_by_dois_normalizes_before_batching_and_lookup(monkeypatch):
    """A "doi:"-prefixed or dx.doi.org-wrapped field is a fine DOI; sending it
    raw made OpenAlex miss it and the report call it 'unknown to OpenAlex'."""
    filters: list[str] = []

    def fake_cached(path, params, cache):
        filters.append(params["filter"])
        return {"results": [{"doi": "https://doi.org/10.1000/ABC", "id": "W1"}]}

    monkeypatch.setattr(openalex, "cached_get_json", fake_cached)
    resolved = openalex.works_by_dois(
        ["doi: 10.1000/ABC", "https://dx.doi.org/10.1000/def", ""],
        progress=lambda _m: None,
    )
    assert filters == ["doi:10.1000/abc|10.1000/def"]
    # OpenAlex's own resolver-URL spelling normalizes to the same key.
    assert "10.1000/abc" in resolved


def test_malformed_dois_are_excluded_from_batches_and_announced(monkeypatch):
    """'|' and ',' are OpenAlex filter syntax: one malformed DOI field used to
    corrupt or crash the whole 40-DOI batch it rode in. It must be left out,
    named, and surface as unresolvable."""
    filters: list[str] = []

    def fake_cached(path, params, cache):
        filters.append(params["filter"])
        return {"results": []}

    monkeypatch.setattr(openalex, "cached_get_json", fake_cached)
    messages: list[str] = []
    resolved = openalex.works_by_dois(
        ["10.1000/good", "10.1000/bad|pipe", "10.1000/bad,comma"],
        progress=messages.append,
    )
    assert filters == ["doi:10.1000/good"]  # the good DOI still went out alone
    assert resolved == {}
    assert len(messages) == 2
    assert any("10.1000/bad|pipe" in m for m in messages)
    assert all("unresolvable" in m for m in messages)


def test_corpus_works_normalizes_doi_fields():
    """The corpus side of the same contract: the membership check against
    works_by_dois' result must compare normalized with normalized."""
    from cda.diagnose.report import corpus_works

    class _FakeCollection:
        def __init__(self, metadatas):
            self._metadatas = metadatas

        def get(self, include=None, limit=None, offset=0):
            return {"metadatas": self._metadatas[offset : offset + (limit or 1000)]}

    works = corpus_works(
        _FakeCollection(
            [
                {"pdf_filename": "a.pdf", "pdf_relpath": "a.pdf", "doi": "doi: 10.1000/ABC"},
                {
                    "pdf_filename": "b.pdf",
                    "pdf_relpath": "b.pdf",
                    "doi": "https://dx.doi.org/10.1000/def",
                },
            ]
        )
    )
    assert [work["doi"] for work in works] == ["10.1000/abc", "10.1000/def"]


# --- the sqlite response cache under concurrency ------------------------------


def test_cache_connects_with_a_busy_timeout(tmp_path, monkeypatch):
    """GUI and CLI diagnose runs share one cache file; without a busy timeout
    the second writer fails immediately instead of waiting its turn."""
    captured: dict = {}
    real_connect = openalex.sqlite3.connect

    def spy(path, timeout=0.0, **kwargs):
        captured["timeout"] = timeout
        return real_connect(path, timeout=timeout, **kwargs)

    monkeypatch.setattr(openalex.sqlite3, "connect", spy)
    cache = openalex.OpenAlexCache(tmp_path / "cache.sqlite")
    cache.put("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    cache.close()
    assert captured["timeout"] >= 30


def test_a_locked_cache_becomes_an_openalex_error_with_advice(tmp_path):
    """A lock that outlasts the busy timeout must say which run to let finish
    — cli catches OpenAlexError, so this also keeps the traceback away."""
    import sqlite3

    cache = openalex.OpenAlexCache(tmp_path / "cache.sqlite")

    class _Locked:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        def commit(self):  # pragma: no cover - execute raises first
            pass

    cache._conn = _Locked()
    with pytest.raises(openalex.OpenAlexError) as caught:
        cache.get("k")
    message = str(caught.value)
    assert "database is locked" in message
    assert "diagnose" in message
    with pytest.raises(openalex.OpenAlexError):
        cache.put("k", {"a": 1})
