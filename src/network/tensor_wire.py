"""Tensor serialization and length-prefixed network protocol.

Shared by both Pipeline and Node for sending/receiving tensors over TCP.
"""

import asyncio
import io
import struct
import torch
import structlog

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


async def send_message(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Send length-prefixed data over a stream."""
    header = struct.pack(">I", len(data))
    writer.write(header + data)
    await writer.drain()
    log.debug("message_sent", bytes=len(data))


async def receive_message(reader: asyncio.StreamReader) -> bytes:
    """Receive length-prefixed data from a stream."""
    header = await reader.readexactly(4)
    length = struct.unpack(">I", header)[0]
    data = await reader.readexactly(length)
    log.debug("message_received", bytes=length)
    return data
