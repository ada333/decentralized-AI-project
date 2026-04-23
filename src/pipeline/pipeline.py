import asyncio
import io
import struct
import torch

from model.model import Model


class Pipeline:
    """Orchestrates distributed inference by routing tensors through remote nodes."""

    def __init__(self, model: Model, nodes_addresses: list[tuple[str, int]]):
        self.model = model
        self.nodes_addresses = nodes_addresses
        self._connections: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    async def _connect_to_nodes(self):
        """Open a TCP connection to each node."""
        for host, port in self.nodes_addresses:
            reader, writer = await asyncio.open_connection(host, port)
            self._connections.append((reader, writer))

    async def _close_connections(self):
        """Close all TCP connections."""
        for _, writer in self._connections:
            writer.close()
            await writer.wait_closed()
        self._connections = []

    async def _send(self, node_index: int, data: bytes):
        """Send length-prefixed data to a node."""
        _, writer = self._connections[node_index]
        header = struct.pack(">I", len(data))
        writer.write(header + data)
        await writer.drain()

    async def _receive(self, node_index: int) -> bytes:
        """Receive length-prefixed data from a node."""
        reader, _ = self._connections[node_index]
        header = await reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        return await reader.readexactly(length)

    def _serialize_tensors(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> bytes:
        """Pack hidden states and position embeddings into bytes."""
        buffer = io.BytesIO()
        torch.save((hidden_states, position_embeddings), buffer)
        return buffer.getvalue()

    def _deserialize_tensors(
        self, data: bytes
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Unpack hidden states and position embeddings from bytes."""
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)

    async def _forward_through_nodes(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Send hidden states through all nodes and return the final output."""
        data = self._serialize_tensors(hidden_states, position_embeddings)

        for i in range(len(self.nodes_addresses)):
            await self._send(i, data)
            data = await self._receive(i)

        hidden_states, _ = self._deserialize_tensors(data)
        return hidden_states

    async def _generate_next_token(self, token_ids: list[int]) -> int:
        """Embed tokens, forward through nodes, and sample next token."""
        hidden_states = self.model.embed(token_ids)
        position_embeddings = self.model.get_position_embeddings(len(token_ids))
        hidden_states = await self._forward_through_nodes(
            hidden_states, position_embeddings
        )
        logits = self.model.apply_lm_head(hidden_states)
        return self.model.sample(logits)

    @torch.no_grad()
    async def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text for a single prompt by distributing inference across nodes."""
        self.model.reset_position()
        await self._connect_to_nodes()

        try:
            token_ids = self.model.tokenize(prompt)
            next_token = await self._generate_next_token(token_ids)

            if next_token == self.model.eos_token_id:
                return ""

            generated = [next_token]

            for _ in range(max_new_tokens - 1):
                next_token = await self._generate_next_token([next_token])

                if next_token == self.model.eos_token_id:
                    break

                generated.append(next_token)

            return self.model.detokenize(generated)

        finally:
            await self._close_connections()
