"""Unit tests for Pipeline class."""

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from network.tensor_wire import serialize_tensors
from pipeline.pipeline import Pipeline


def test_pipeline_init(mock_model):
    """Pipeline constructor sets model and nodes."""
    nodes = [("127.0.0.1", 8000), ("127.0.0.1", 8001)]
    pipeline = Pipeline(model=mock_model, nodes_addresses=nodes)

    assert pipeline.model is mock_model
    assert pipeline.nodes_addresses == nodes
    assert pipeline._connections == []


@pytest.mark.asyncio
async def test_connect_to_nodes(mock_model):
    """_connect_to_nodes opens TCP connection to each node."""
    nodes = [("127.0.0.1", 8000), ("127.0.0.1", 8001)]
    pipeline = Pipeline(mock_model, nodes)

    with patch("asyncio.open_connection", new_callable=AsyncMock) as mock_open:
        mock_open.return_value = (MagicMock(), MagicMock())
        await pipeline._connect_to_nodes()

    assert len(pipeline._connections) == 2
    assert mock_open.call_count == 2


@pytest.mark.asyncio
async def test_close_connections(mock_model):
    """_close_connections closes all writers."""
    pipeline = Pipeline(mock_model, [])

    for _ in range(2):
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        pipeline._connections.append((MagicMock(), writer))

    await pipeline._close_connections()

    assert pipeline._connections == []


@pytest.mark.asyncio
async def test_forward_through_nodes_with_session_id(mock_model):
    """_forward_through_nodes sends session ID with data."""
    pipeline = Pipeline(mock_model, [])
    session_id = b"\xaa\xbb\xcc\xdd"

    reader = asyncio.StreamReader()
    response_hidden = torch.randn(1, 3, 576)
    pos_emb = (torch.randn(1, 3, 576), torch.randn(1, 3, 576))
    response_data = serialize_tensors(response_hidden, pos_emb)

    # Feed session ID + length + data
    header = session_id + struct.pack(">I", len(response_data))
    reader.feed_data(header + response_data)

    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()

    pipeline._connections = [(reader, writer)]

    hidden = torch.randn(1, 3, 576)
    result = await pipeline._forward_through_nodes(session_id, hidden, pos_emb)

    assert result.shape == response_hidden.shape
    assert writer.write.call_count == 1


@pytest.mark.asyncio
async def test_generate_stops_on_eos(mock_model):
    """generate() stops when EOS token is generated."""
    pipeline = Pipeline(mock_model, [])

    pipeline._connect_to_nodes = AsyncMock()
    pipeline._close_connections = AsyncMock()

    call_count = 0

    async def mock_generate_token(_session_id, _tokens):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            return mock_model.eos_token_id
        return 100 + call_count

    pipeline._generate_next_token = mock_generate_token

    await pipeline.generate("test", max_new_tokens=10)

    assert call_count == 3


@pytest.mark.asyncio
async def test_generate_returns_detokenized_text(mock_model):
    """generate() returns detokenized generated tokens."""
    pipeline = Pipeline(mock_model, [])

    pipeline._connect_to_nodes = AsyncMock()
    pipeline._close_connections = AsyncMock()
    pipeline._generate_next_token = AsyncMock(side_effect=[10, 20, mock_model.eos_token_id])

    result = await pipeline.generate("test")

    mock_model.detokenize.assert_called_once_with([10, 20])
    assert result == "generated text"
