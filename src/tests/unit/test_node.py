"""Unit tests for Node class."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from network.tensor_wire import serialize_tensors
from node.node import Node


def test_node_init():
    """Node constructor sets all expected attributes."""
    node = Node(
        shards_dir="/path/to/shards",
        layer_start=5,
        layer_end=10,
        host="127.0.0.1",
        port=8080,
    )

    assert node.shards_dir == "/path/to/shards"
    assert node.layer_start == 5
    assert node.layer_end == 10
    assert node.host == "127.0.0.1"
    assert node.port == 8080
    assert node.layers == []
    assert node.kv_caches == {}
    assert node.log is not None


def test_forward_with_no_layers():
    """forward() with empty layers returns input unchanged."""
    node = Node("/shards", 0, 0, "localhost", 8000)
    node.layers = []
    node.kv_caches[b"\x01\x02\x03\x04"] = MagicMock()

    hidden = torch.randn(1, 3, 576)
    pos_emb = (torch.randn(1, 3, 576), torch.randn(1, 3, 576))

    result = node.forward(b"\x01\x02\x03\x04", hidden, pos_emb)

    assert torch.equal(result, hidden)


def test_forward_passes_through_layers(mock_layer_passthrough):
    """forward() runs hidden states through all layers."""
    node = Node("/shards", 0, 2, "localhost", 8000)
    node.layers = [mock_layer_passthrough, mock_layer_passthrough]
    session_id = b"\xaa\xbb\xcc\xdd"
    node.kv_caches[session_id] = MagicMock()

    hidden = torch.randn(1, 3, 576)
    pos_emb = (torch.randn(1, 3, 576), torch.randn(1, 3, 576))

    result = node.forward(session_id, hidden, pos_emb)

    assert mock_layer_passthrough.call_count == 2
    assert result.shape == hidden.shape


@pytest.mark.asyncio
async def test_on_connect_creates_session(mock_stream_pair, mock_layer_passthrough):
    """_on_connect creates KV cache for new session."""
    reader, writer = mock_stream_pair
    node = Node("/shards", 0, 1, "localhost", 8000)
    node.layers = [mock_layer_passthrough]

    session_id = b"\x11\x22\x33\x44"
    hidden = torch.randn(1, 3, 576)
    pos_emb = (torch.randn(1, 3, 576), torch.randn(1, 3, 576))
    request_data = serialize_tensors(hidden, pos_emb)

    # Send session ID (4 bytes) + length (4 bytes) + data
    import struct

    header = session_id + struct.pack(">I", len(request_data))
    reader.feed_data(header + request_data)
    reader.feed_eof()

    await node._on_connect(reader, writer)

    assert session_id in node.kv_caches
    assert writer._closed is True


@patch("node.node.AutoConfig.from_pretrained")
@patch("node.node.torch.load")
def test_load_layers_creates_correct_number(mock_torch_load, mock_config):
    """load_layers creates one layer per index in range."""
    mock_config.return_value = MagicMock(
        hidden_size=576,
        intermediate_size=1536,
        num_attention_heads=9,
        num_key_value_heads=3,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        _attn_implementation="sdpa",
    )
    mock_torch_load.return_value = {}

    node = Node("/shards", 0, 3, "localhost", 8000)

    with patch("node.node.LlamaDecoderLayer") as mock_layer_class:
        mock_layer_instance = MagicMock()
        mock_layer_class.return_value = mock_layer_instance

        node.load_layers()

        assert mock_layer_class.call_count == 3
        assert len(node.layers) == 3
