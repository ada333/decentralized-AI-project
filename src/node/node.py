import asyncio
import io
import os
import struct
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.cache_utils import DynamicCache
import structlog


class Node:
    layer_start: int
    layer_end: int
    layers: list[nn.Module]
    kv_cache: DynamicCache
    host: str
    port: int

    def __init__(self, shards_dir: str, layer_start: int, layer_end: int, host: str, port: int):
        self.shards_dir = shards_dir
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.layers: list[nn.Module] = []
        self.kv_cache = DynamicCache()
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self.log = structlog.get_logger().bind(
            node_id=f"{host}:{port}",
            layers=f"{layer_start}-{layer_end}",
        )

    def load_layers(self):
        """Load this node's layer range from individual shard files.

        Each layer_X.pt contains a state_dict (weights only), so we create
        an empty LlamaDecoderLayer, load the weights, and set it to eval mode.
        """
        self.log.info("loading_layers", shards_dir=self.shards_dir)
        try:
            config = AutoConfig.from_pretrained(os.path.join(self.shards_dir, "tokenizer"))
        except Exception as e:
            self.log.error("config_load_failed", shards_dir=self.shards_dir, error=str(e))
            raise
        config._attn_implementation = "sdpa"  # Match the model's attention implementation

        for i in range(self.layer_start, self.layer_end):
            local_idx = i - self.layer_start  # Use local index for KV cache
            layer = LlamaDecoderLayer(config, layer_idx=local_idx)
            layer_path = os.path.join(self.shards_dir, f"layer_{i}.pt")
            try:
                state_dict = torch.load(layer_path, weights_only=True)
            except FileNotFoundError:
                self.log.error("layer_file_not_found", layer_idx=i, path=layer_path)
                raise
            except Exception as e:
                self.log.error("layer_load_failed", layer_idx=i, path=layer_path, error=str(e))
                raise
            layer.load_state_dict(state_dict)
            layer.eval()
            self.layers.append(layer)
            self.log.debug("layer_loaded", layer_idx=i)
        self.log.info("layers_loaded", count=len(self.layers))


    async def start(self):
        """Start TCP server and listen for a connection from the Pipeline."""
        self.log.info("server_starting")
        try:
            server = await asyncio.start_server(
                self._on_connect, self.host, self.port,
            )
        except OSError as e:
            self.log.error("server_bind_failed", error=str(e))
            raise
        self.log.info("server_listening")
        async with server:
            await server.serve_forever()


    async def _on_connect(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Called when the Pipeline connects. Runs a request loop."""
        peer = writer.get_extra_info("peername")
        self.log.info("client_connected", peer=peer)
        self._reader = reader
        self._writer = writer
        self.kv_cache = DynamicCache()  # Reset cache for new session
        
        try:
            while True:
                data = await self.receive()
                hidden_states, position_embeddings = self._deserialize_tensors(data)
                self.log.debug("forward_start", input_shape=list(hidden_states.shape))
                hidden_states = self.forward(hidden_states, position_embeddings)
                self.log.debug("forward_done", output_shape=list(hidden_states.shape))
                await self.send(self._serialize_tensors(hidden_states, position_embeddings))
        except asyncio.IncompleteReadError:
            self.log.info("client_disconnected", peer=peer)
        except Exception as e:
            self.log.error("request_handling_failed", peer=peer, error=str(e), exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()


    async def send(self, data: bytes):
        """Send length-prefixed data back to the Pipeline."""
        # header is 4 bytes long and contains the length of the data
        # >I means big endian unsigned int
        header = struct.pack(">I", len(data))
        self._writer.write(header + data)
        await self._writer.drain()
        self.log.debug("sent", bytes=len(data))

    async def receive(self) -> bytes:
        """Receive length-prefixed data from the Pipeline."""
        # read the header first which we presume to be 4 bytes long 
        # and contains the length of the data
        header = await self._reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        data = await self._reader.readexactly(length)
        self.log.debug("received", bytes=length)
        return data


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.layers:
            self.load_layers()

        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                past_key_values=self.kv_cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            # HF layer return type can vary by version:
            # - torch.Tensor
            # - tuple where index 0 is hidden_states
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return hidden_states

    def _deserialize_tensors(self, data: bytes) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Unpack hidden states and position embeddings from bytes."""
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)

    def _serialize_tensors(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> bytes:
        """Pack hidden states and position embeddings into bytes."""
        buffer = io.BytesIO()
        torch.save((hidden_states, position_embeddings), buffer)
        return buffer.getvalue()