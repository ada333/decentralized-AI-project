"""Tensor serialization and length-prefixed network protocol.

Shared by both Pipeline and Node for sending/receiving tensors over TCP.
"""

import asyncio
import io
import struct

import structlog
import torch

log = structlog.get_logger()


def serialize_tensors(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> bytes:
    """Pack hidden states and position embeddings into bytes."""
    buffer = io.BytesIO()
    torch.save((hidden_states, position_embeddings), buffer)
    return buffer.getvalue()


def deserialize_tensors(
    data: bytes,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Unpack hidden states and position embeddings from bytes."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=True)


async def send_message(writer: asyncio.StreamWriter, session_id: bytes, data: bytes) -> None:
    """Send session-tagged, length-prefixed data over a stream.

    Args:
        writer: The stream to write to.
        session_id: 4-byte session identifier.
        data: Payload bytes to send.
    """
    header = session_id + struct.pack(">I", len(data))
    writer.write(header + data)
    await writer.drain()
    log.debug("message_sent", session_id=session_id.hex(), bytes=len(data))


async def receive_message(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    """Receive session-tagged, length-prefixed data from a stream.

    Returns:
        Tuple of (session_id, data) where session_id is 4 bytes.
    """
    session_id = await reader.readexactly(4)
    length_header = await reader.readexactly(4)
    length = struct.unpack(">I", length_header)[0]
    data = await reader.readexactly(length)
    log.debug("message_received", session_id=session_id.hex(), bytes=length)
    return session_id, data
