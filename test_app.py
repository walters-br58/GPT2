"""
EECE 4520 - Milestone 5
Unit and API tests for app.py using pytest.

Run with:
    pytest test_app.py -v
"""

import json
import pytest

# ── Patch model/tokenizer so tests run without the actual .pt file ─────────

import types, sys

# Minimal fake tokenizer
class FakeEncoding:
    ids = [1, 2, 3]

class FakeTokenizer:
    def encode(self, text):           return FakeEncoding()
    def decode(self, ids):            return "generated text continuation"
    def token_to_id(self, tok):       return 0

# Minimal fake model
import torch
import torch.nn as nn

class FakeModel(nn.Module):
    class cfg:
        block_size = 128
    def forward(self, idx, targets=None):
        B, T = idx.shape
        vocab_size = 8000
        logits = torch.zeros(B, T, vocab_size)
        return logits, None
    def eval(self): return self
    def to(self, device): return self
    def load_state_dict(self, state): pass

# Patch at import time before app is loaded
import unittest.mock as mock

@pytest.fixture(autouse=True, scope="session")
def patch_model_and_tokenizer():
    """Replace model and tokenizer globals with fakes for all tests."""
    with mock.patch("model_factory.ModelFactory.create_model", return_value=FakeModel()), \
         mock.patch("tokenizer_singleton.TokenizerSingleton.get_instance", return_value=FakeTokenizer()), \
         mock.patch("torch.load", return_value={"model_state": {}}):
        import app
        app.model     = FakeModel()
        app.tokenizer = FakeTokenizer()
        yield app


@pytest.fixture()
def client(patch_model_and_tokenizer):
    """Flask test client."""
    patch_model_and_tokenizer.app.config["TESTING"] = True
    with patch_model_and_tokenizer.app.test_client() as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").get_json()
        assert "status"    in data
        assert "model"     in data
        assert "tokenizer" in data
        assert "device"    in data

    def test_health_status_ok(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# /generate — valid requests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateValid:
    def _post(self, client, payload):
        return client.post(
            "/generate",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_basic_prompt_returns_200(self, client):
        res = self._post(client, {"prompt": "The history of AI"})
        assert res.status_code == 200

    def test_response_has_required_fields(self, client):
        data = self._post(client, {"prompt": "Hello world"}).get_json()
        assert "prompt"     in data
        assert "response"   in data
        assert "strategy"   in data
        assert "latency_ms" in data

    def test_prompt_echoed_in_response(self, client):
        prompt = "Once upon a time"
        data   = self._post(client, {"prompt": prompt}).get_json()
        assert data["prompt"] == prompt

    def test_greedy_strategy(self, client):
        data = self._post(client, {"prompt": "Test", "strategy": "greedy"}).get_json()
        assert data["strategy"] == "greedy"

    def test_nucleus_strategy(self, client):
        data = self._post(client, {"prompt": "Test", "strategy": "nucleus"}).get_json()
        assert data["strategy"] == "nucleus"

    def test_top_k_strategy(self, client):
        data = self._post(client, {"prompt": "Test", "strategy": "top_k"}).get_json()
        assert data["strategy"] == "top_k"

    def test_custom_temperature(self, client):
        res = self._post(client, {"prompt": "Test", "temperature": 1.2})
        assert res.status_code == 200

    def test_custom_max_tokens(self, client):
        res = self._post(client, {"prompt": "Test", "max_new_tokens": 50})
        assert res.status_code == 200

    def test_latency_is_positive_number(self, client):
        data = self._post(client, {"prompt": "Test"}).get_json()
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# /generate — invalid requests (error handling)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateInvalid:
    def _post(self, client, payload):
        return client.post(
            "/generate",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_missing_prompt_returns_400(self, client):
        res = self._post(client, {})
        assert res.status_code == 400

    def test_empty_prompt_returns_400(self, client):
        res = self._post(client, {"prompt": "   "})
        assert res.status_code == 400

    def test_prompt_too_long_returns_400(self, client):
        res = self._post(client, {"prompt": "a" * 501})
        assert res.status_code == 400

    def test_invalid_strategy_returns_400(self, client):
        res = self._post(client, {"prompt": "Test", "strategy": "magic"})
        assert res.status_code == 400

    def test_zero_temperature_returns_400(self, client):
        res = self._post(client, {"prompt": "Test", "temperature": 0})
        assert res.status_code == 400

    def test_negative_temperature_returns_400(self, client):
        res = self._post(client, {"prompt": "Test", "temperature": -0.5})
        assert res.status_code == 400

    def test_max_tokens_out_of_range_returns_400(self, client):
        res = self._post(client, {"prompt": "Test", "max_new_tokens": 9999})
        assert res.status_code == 400

    def test_non_json_body_returns_400(self, client):
        res = client.post("/generate", data="not json", content_type="text/plain")
        assert res.status_code == 400

    def test_error_response_has_error_field(self, client):
        data = self._post(client, {"prompt": ""}).get_json()
        assert "error" in data


# ─────────────────────────────────────────────────────────────────────────────
# Static / index
# ─────────────────────────────────────────────────────────────────────────────

class TestIndex:
    def test_index_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_index_is_html(self, client):
        res = client.get("/")
        assert b"<!DOCTYPE html>" in res.data or b"<html" in res.data
