import asyncio
import os
import struct
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.cache_utils import DynamicCache


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

    def load_layers(self):
        """Load this node's layer range from individual shard files.

        Each layer_X.pt contains a state_dict (weights only), so we create
        an empty LlamaDecoderLayer, load the weights, and set it to eval mode.
        """
        config = AutoConfig.from_pretrained(os.path.join(self.shards_dir, "tokenizer"))

        for i in range(self.layer_start, self.layer_end + 1):
            layer = LlamaDecoderLayer(config, layer_idx=i)
            state_dict = torch.load(
                os.path.join(self.shards_dir, f"layer_{i}.pt"),
                weights_only=True,
            )
            layer.load_state_dict(state_dict)
            layer.eval()
            self.layers.append(layer)


    async def start(self):
        """Start TCP server and listen for a connection from the Pipeline."""
        server = await asyncio.start_server(
            self._on_connect, self.host, self.port,
        )
        async with server:
            await server.serve_forever()


    async def _on_connect(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Called when the Pipeline connects. Stores the reader/writer pair."""
        self._reader = reader
        self._writer = writer


    async def send(self, data: bytes):
        """Send length-prefixed data back to the Pipeline."""
        # header is 4 bytes long and contains the length of the data
        # >I means big endian unsigned int
        header = struct.pack(">I", len(data))
        self._writer.write(header + data)
        await self._writer.drain()


    async def receive(self) -> bytes:
        """Receive length-prefixed data from the Pipeline."""
        # read the header first which we presume to be 4 bytes long 
        # and contains the length of the data
        header = await self._reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        return await self._reader.readexactly(length)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.layers:
            self.load_layers()

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                past_key_values=self.kv_cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
        return hidden_states