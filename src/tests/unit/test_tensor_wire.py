"""Unit tests for tensor_wire serialization and message protocol."""

import pytest
import torch

from network.tensor_wire import (
    deserialize_tensors,
    receive_message,
    send_message,
    serialize_tensors,
)


def test_serialize_deserialize_roundtrip():
    """Serialize and deserialize tensors preserves data."""
    hidden = torch.randn(1, 3, 576)
    pos_emb = (torch.randn(1, 3, 576), torch.randn(1, 3, 576))

    data = serialize_tensors(hidden, pos_emb)
    h_out, p_out = deserialize_tensors(data)

    assert torch.allclose(hidden, h_out)
    assert torch.allclose(pos_emb[0], p_out[0])
    assert torch.allclose(pos_emb[1], p_out[1])


@pytest.mark.asyncio
async def test_send_receive_with_session_id(mock_stream_pair):
    """Send and receive message with session ID."""
    reader, writer = mock_stream_pair
    session_id = b"\x01\x02\x03\x04"
    payload = b"test data"

    await send_message(writer, session_id, payload)
    recv_session_id, recv_payload = await receive_message(reader)

    assert recv_session_id == session_id
    assert recv_payload == payload


@pytest.mark.asyncio
async def test_send_receive_tensor_roundtrip(mock_stream_pair):
    """Full roundtrip: serialize → send → receive → deserialize."""
    reader, writer = mock_stream_pair
    session_id = b"\xaa\xbb\xcc\xdd"

    hidden = torch.randn(1, 3, 576)
    pos_emb = (torch.randn(1, 3, 576), torch.randn(1, 3, 576))
    data = serialize_tensors(hidden, pos_emb)

    await send_message(writer, session_id, data)
    recv_session_id, recv_data = await receive_message(reader)
    h_out, p_out = deserialize_tensors(recv_data)

    assert recv_session_id == session_id
    assert torch.allclose(hidden, h_out)
    assert torch.allclose(pos_emb[0], p_out[0])
