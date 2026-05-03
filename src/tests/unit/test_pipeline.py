"""Unit tests for Pipeline class."""

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from network.tensor_wire import serialize_tensors
from node.coordinator import PipelineCoordinator
from node.node_group import NodeInfo, SelectionStrategy
from pipeline.pipeline import Pipeline


def test_pipeline_init(mock_model):
    """Pipeline constructor sets model and coordinator."""
    coordinator = PipelineCoordinator()
    coordinator.register_node(NodeInfo("node1", "127.0.0.1", 8000, group_id=0))
    coordinator.register_node(NodeInfo("node2", "127.0.0.1", 8001, group_id=1))

    pipeline = Pipeline(model=mock_model, coordinator=coordinator)

    assert pipeline.model is mock_model
    assert pipeline.coordinator is coordinator
    assert pipeline.selection_strategy == SelectionStrategy.ROUND_ROBIN


@pytest.mark.asyncio
async def test_connect_to_nodes(mock_model):
    """_connect_to_nodes opens TCP connection to each node."""
    coordinator = PipelineCoordinator()
    coordinator.register_node(NodeInfo("node1", "127.0.0.1", 8000, group_id=0))
    coordinator.register_node(NodeInfo("node2", "127.0.0.1", 8001, group_id=1))

    pipeline = Pipeline(mock_model, coordinator)

    with patch("asyncio.open_connection", new_callable=AsyncMock) as mock_open:
        mock_open.return_value = (MagicMock(), MagicMock())
        await pipeline._connect_to_nodes()

    # Check that connections were stored in NodeInfo objects
    all_groups = coordinator.get_all_groups()
    for group in all_groups:
        for node in group.nodes:
            assert node.reader is not None
            assert node.writer is not None

    assert mock_open.call_count == 2


@pytest.mark.asyncio
async def test_close_connections(mock_model):
    """_close_connections closes all writers."""
    coordinator = PipelineCoordinator()
    node1 = NodeInfo("node1", "127.0.0.1", 8000, group_id=0)
    node2 = NodeInfo("node2", "127.0.0.1", 8001, group_id=1)

    # Attach mock connections
    for node in [node1, node2]:
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        node.reader = MagicMock()
        node.writer = writer

    coordinator.register_node(node1)
    coordinator.register_node(node2)

    pipeline = Pipeline(mock_model, coordinator)
    await pipeline._close_connections()

    # Verify all connections were closed
    for group in coordinator.get_all_groups():
        for node in group.nodes:
            assert node.reader is None
            assert node.writer is None


@pytest.mark.asyncio
async def test_forward_through_nodes_with_session_id(mock_model):
    """_forward_through_nodes sends session ID with data."""
    coordinator = PipelineCoordinator()
    node1 = NodeInfo("node1", "127.0.0.1", 8000, group_id=0)

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

    node1.reader = reader
    node1.writer = writer

    coordinator.register_node(node1)
    pipeline = Pipeline(mock_model, coordinator)

    hidden = torch.randn(1, 3, 576)
    result = await pipeline._forward_through_nodes(session_id, hidden, pos_emb)

    assert result.shape == response_hidden.shape
    assert writer.write.call_count == 1


@pytest.mark.asyncio
async def test_generate_stops_on_eos(mock_model):
    """generate() stops when EOS token is generated."""
    coordinator = PipelineCoordinator()
    pipeline = Pipeline(mock_model, coordinator)

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
    coordinator = PipelineCoordinator()
    pipeline = Pipeline(mock_model, coordinator)

    pipeline._connect_to_nodes = AsyncMock()
    pipeline._close_connections = AsyncMock()
    pipeline._generate_next_token = AsyncMock(side_effect=[10, 20, mock_model.eos_token_id])

    result = await pipeline.generate("test")

    mock_model.detokenize.assert_called_once_with([10, 20])
    assert result == "generated text"
