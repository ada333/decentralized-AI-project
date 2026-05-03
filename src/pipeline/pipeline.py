import asyncio
import os

import structlog
import torch

from model.model import Model
from network.tensor_wire import (
    deserialize_tensors,
    receive_message,
    send_message,
    serialize_tensors,
)
from node.coordinator import PipelineCoordinator
from node.node_group import SelectionStrategy

log = structlog.get_logger()


class Pipeline:
    """Coordinates distributed text generation by routing hidden states through remote nodes.

    Owns the tokenizer and model head components. Drives the generation loop: tokenize prompt,
    embed, forward through nodes, apply LM head, sample, repeat.

    Uses a PipelineCoordinator to manage node groups and enable load balancing / fault tolerance.

    Args:
        model: Model instance containing tokenizer, embedding layer, and LM head.
        coordinator: PipelineCoordinator instance managing node groups.
        selection_strategy: Node selection strategy (see SelectionStrategy enum).
    """

    def __init__(
        self,
        model: Model,
        coordinator: PipelineCoordinator,
        selection_strategy: SelectionStrategy = SelectionStrategy.ROUND_ROBIN,
    ):
        self.model = model
        self.coordinator = coordinator
        self.selection_strategy = selection_strategy

    async def _connect_to_nodes(self):
        """Open a TCP connection to each node in all groups."""
        all_groups = self.coordinator.get_all_groups()
        total_nodes = sum(len(group.nodes) for group in all_groups)
        log.info("connecting_to_nodes", num_groups=len(all_groups), total_nodes=total_nodes)

        for group in all_groups:
            for node in group.nodes:
                try:
                    reader, writer = await asyncio.open_connection(node.host, node.port)
                    node.reader = reader
                    node.writer = writer
                    log.debug(
                        "node_connected",
                        node_id=node.node_id,
                        host=node.host,
                        port=node.port,
                        group_id=node.group_id,
                    )
                except OSError as e:
                    log.error(
                        "node_connection_failed",
                        node_id=node.node_id,
                        host=node.host,
                        port=node.port,
                        error=str(e),
                    )
                    raise
        log.info("all_nodes_connected")

    async def _close_connections(self):
        """Close all TCP connections."""
        all_groups = self.coordinator.get_all_groups()
        total_nodes = sum(len(group.nodes) for group in all_groups)
        log.info("closing_connections", total_nodes=total_nodes)

        for group in all_groups:
            for node in group.nodes:
                if node.writer:
                    node.writer.close()
                    await node.writer.wait_closed()
                    node.reader = None
                    node.writer = None

    async def _forward_through_nodes(
        self,
        session_id: bytes,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Send hidden states through all node groups and return the final output.

        For each group in the pipeline, selects an available node using the configured
        strategy (least_loaded, round_robin, etc.) and routes the request through it.

        Args:
            session_id: 4-byte session identifier for KV cache lookup.
            hidden_states: Tensor of shape [batch, seq_len, hidden_dim].
            position_embeddings: Tuple of (cos, sin) rotary embeddings.

        Returns:
            Hidden states after passing through all groups' layers.
        """
        data = serialize_tensors(hidden_states, position_embeddings)
        current_group = self.coordinator.get_first_group()

        while current_group:
            # Select a node from this group
            node = current_group.get_available_node(strategy=self.selection_strategy)
            node.active_sessions += 1

            try:
                await send_message(node.writer, session_id, data)
                _, data = await receive_message(node.reader)
                log.debug(
                    "group_forward_complete",
                    node_id=node.node_id,
                    group_id=node.group_id,
                    session_id=session_id.hex(),
                )
            except (OSError, asyncio.IncompleteReadError) as e:
                log.error(
                    "node_communication_failed",
                    node_id=node.node_id,
                    group_id=node.group_id,
                    error=str(e),
                )
                raise
            finally:
                node.active_sessions -= 1

            # Move to next group
            current_group = current_group.next_group

        hidden_states, _ = deserialize_tensors(data)
        return hidden_states

    async def _generate_next_token(self, session_id: bytes, token_ids: list[int]) -> int:
        """Embed tokens, forward through nodes, and sample next token.

        Args:
            session_id: 4-byte session identifier for KV cache lookup.
            token_ids: List of token IDs to process. For the initial prompt this is
                all tokens; for subsequent steps it's just the previously generated token.

        Returns:
            The sampled next token ID.
        """
        hidden_states = self.model.embed(token_ids)
        position_embeddings = self.model.get_position_embeddings(len(token_ids))
        hidden_states = await self._forward_through_nodes(
            session_id, hidden_states, position_embeddings
        )
        logits = self.model.apply_lm_head(hidden_states)
        return self.model.sample(logits)

    @torch.no_grad()
    async def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text for a single prompt by distributing inference across nodes.

        Args:
            prompt: Input text to continue from.
            max_new_tokens: Maximum number of tokens to generate. Generation stops
                early if the model produces an EOS token.

        Returns:
            The generated text (excluding the original prompt).
        """
        session_id = os.urandom(4)
        log.info(
            "generation_start",
            session_id=session_id.hex(),
            prompt_length=len(prompt),
            max_new_tokens=max_new_tokens,
        )
        self.model.reset_position()
        await self._connect_to_nodes()

        try:
            token_ids = self.model.tokenize(prompt)
            log.debug("prompt_tokenized", num_tokens=len(token_ids))
            next_token = await self._generate_next_token(session_id, token_ids)

            if next_token == self.model.eos_token_id:
                log.info("generation_complete", tokens_generated=0, reason="eos_immediate")
                return ""

            generated = [next_token]

            for _ in range(max_new_tokens - 1):
                next_token = await self._generate_next_token(session_id, [next_token])

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
