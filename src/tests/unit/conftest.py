"""Shared test fixtures for unit tests."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MockStreamWriter:
    """In-memory StreamWriter that captures written data."""

    def __init__(self, target_reader: asyncio.StreamReader | None = None):
        self._target_reader = target_reader
        self._buffer = bytearray()
        self._closed = False

    def write(self, data: bytes):
        self._buffer.extend(data)
        if self._target_reader:
            self._target_reader.feed_data(data)

    async def drain(self):
        pass

    def close(self):
        self._closed = True

    async def wait_closed(self):
        pass

    def get_extra_info(self, name: str):
        if name == "peername":
            return ("127.0.0.1", 12345)
        return None

    def get_written_data(self) -> bytes:
        return bytes(self._buffer)


@pytest.fixture
async def mock_stream_pair():
    """Create a connected (reader, writer) pair for testing send/receive."""
    reader = asyncio.StreamReader()
    writer = MockStreamWriter(target_reader=reader)
    return reader, writer


@pytest.fixture
def mock_layer_passthrough():
    """A mock transformer layer that passes through input hidden states."""

    def forward_fn(hidden_states, **kwargs):
        return (hidden_states,)

    layer = MagicMock()
    layer.side_effect = forward_fn
    return layer


@pytest.fixture
def mock_model():
    """A mock Model with all methods stubbed."""
    model = MagicMock()
    model.eos_token_id = 2
    model.tokenize.return_value = [1, 2, 3]
    model.detokenize.return_value = "generated text"
    model.embed.return_value = torch.randn(1, 3, 576)
    model.get_position_embeddings.return_value = (
        torch.randn(1, 3, 576),
        torch.randn(1, 3, 576),
    )
    model.apply_lm_head.return_value = torch.randn(1, 3, 50000)
    model.sample.return_value = 42
    model.reset_position = MagicMock()
    return model
