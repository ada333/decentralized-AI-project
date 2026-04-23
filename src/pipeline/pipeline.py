import asyncio
import io
import struct
import torch
import structlog

from model.model import Model

log = structlog.get_logger()


class Pipeline:
    """Coordinates distributed text generation by routing hidden states through remote nodes.
    
    Owns the tokenizer and model head components. Drives the generation loop: tokenize prompt,
    embed, forward through nodes, apply LM head, sample, repeat.
    """
    def __init__(self, model: Model, nodes_addresses: list[tuple[str, int]]):
        self.model = model
        self.nodes_addresses = nodes_addresses
        self._connections: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    async def _connect_to_nodes(self):
        """Open a TCP connection to each node."""
        log.info("connecting_to_nodes", count=len(self.nodes_addresses))
        for host, port in self.nodes_addresses:
            try:
                reader, writer = await asyncio.open_connection(host, port)
            except OSError as e:
                log.error("node_connection_failed", host=host, port=port, error=str(e))
                raise
            self._connections.append((reader, writer))
            log.debug("node_connected", host=host, port=port)
        log.info("all_nodes_connected")

    async def _close_connections(self):
        """Close all TCP connections."""
        log.info("closing_connections", count=len(self._connections))
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
        log.debug("sent_to_node", node_index=node_index, bytes=len(data))

    async def _receive(self, node_index: int) -> bytes:
        """Receive length-prefixed data from a node."""
        reader, _ = self._connections[node_index]
        header = await reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        data = await reader.readexactly(length)
        log.debug("received_from_node", node_index=node_index, bytes=length)
        return data

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
            try:
                await self._send(i, data)
                data = await self._receive(i)
            except (OSError, asyncio.IncompleteReadError) as e:
                log.error("node_communication_failed", node_index=i, error=str(e))
                raise

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
        log.info("generation_start", prompt_length=len(prompt), max_new_tokens=max_new_tokens)
        self.model.reset_position()
        await self._connect_to_nodes()

        try:
            token_ids = self.model.tokenize(prompt)
            log.debug("prompt_tokenized", num_tokens=len(token_ids))
            next_token = await self._generate_next_token(token_ids)

            if next_token == self.model.eos_token_id:
                log.info("generation_complete", tokens_generated=0, reason="eos_immediate")
                return ""

            generated = [next_token]

            for _ in range(max_new_tokens - 1):
                next_token = await self._generate_next_token([next_token])

                if next_token == self.model.eos_token_id:
                    break

                generated.append(next_token)

            log.info("generation_complete", tokens_generated=len(generated))
            return self.model.detokenize(generated)

        except Exception as e:
            log.error("generation_failed", error=str(e), exc_info=True)
            raise
        finally:
            await self._close_connections()
