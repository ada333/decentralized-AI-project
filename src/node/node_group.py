"""Node grouping logic for redundant pipeline architecture.

A NodeGroup represents nodes handling the same layer range, providing redundancy
and load balancing for that pipeline stage.
"""

import asyncio
import random
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from node.layer_groups import get_layer_group

log = structlog.get_logger()


class SelectionStrategy(StrEnum):
    """Strategy for selecting a node from a group during routing."""

    LEAST_LOADED = "least_loaded"
    ROUND_ROBIN = "round_robin"
    FIRST = "first"


@dataclass
class NodeInfo:
    """Information about a single node in the pipeline.

    Attributes:
        node_id: Unique identifier for this node.
        host: Hostname or IP address.
        port: TCP port the node is listening on.
        group_id: ID of the layer group this node handles (see layer_groups.LAYER_GROUPS).
        active_sessions: Number of currently active inference sessions on this node.
            Used for load balancing.
        reader: Optional StreamReader for TCP connection (managed by Pipeline).
        writer: Optional StreamWriter for TCP connection (managed by Pipeline).
    """

    node_id: str
    host: str
    port: int
    group_id: int
    active_sessions: int = 0
    reader: asyncio.StreamReader | None = field(default=None, repr=False)
    writer: asyncio.StreamWriter | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate that group_id exists in LAYER_GROUPS.

        Raises:
            ValueError: If group_id is invalid, with node context.
        """
        try:
            get_layer_group(self.group_id)
        except ValueError as e:
            raise ValueError(
                f"Node '{self.node_id}' at {self.host}:{self.port} has invalid group_id: {e}"
            ) from e


class NodeGroup:
    """Manages all nodes that handle a specific layer group.

    A NodeGroup represents one "stage" in the pipeline. Multiple nodes can belong
    to the same group, providing redundancy and load balancing.

    Attributes:
        group_id: The layer group ID this NodeGroup manages.
        nodes: List of all nodes registered in this group.
        next_group: Reference to the next group in the pipeline (or None if this is the last).
    """

    def __init__(self, group_id: int):
        """Initialize a NodeGroup for a specific layer group.

        Args:
            group_id: ID of the layer group (validates against LAYER_GROUPS).

        Raises:
            ValueError: If group_id is not in LAYER_GROUPS, with context.
        """
        try:
            self.config = get_layer_group(group_id)
        except ValueError as e:
            raise ValueError(f"Cannot create NodeGroup: {e}") from e
        self.group_id = group_id
        self.nodes: list[NodeInfo] = []
        self.next_group: NodeGroup | None = None

    @property
    def layer_range(self) -> tuple[int, int]:
        """Return (start_layer, end_layer) for compatibility.

        This property provides backward compatibility with code expecting tuples.
        """
        return (self.config.start_layer, self.config.end_layer)

    def add_node(self, node: NodeInfo):
        """Register a node in this group.

        Args:
            node: NodeInfo to add.

        Raises:
            ValueError: If the node's group_id doesn't match this group's group_id.
        """
        if node.group_id != self.group_id:
            raise ValueError(f"Node group_id {node.group_id} doesn't match group {self.group_id}")
        self.nodes.append(node)
        log.debug(
            "node_added_to_group",
            node_id=node.node_id,
            group_id=self.group_id,
            layer_range=self.layer_range,
            group_size=len(self.nodes),
        )

    def remove_node(self, node_id: str):
        """Unregister a node from this group (e.g., on disconnect or failure).

        Args:
            node_id: ID of the node to remove.
        """
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
        log.info(
            "node_removed_from_group",
            node_id=node_id,
            group_id=self.group_id,
            layer_range=self.layer_range,
            remaining_nodes=len(self.nodes),
        )

    def get_available_node(
        self, strategy: SelectionStrategy = SelectionStrategy.LEAST_LOADED
    ) -> NodeInfo:
        """Select a node from this group for routing.

        Args:
            strategy: Selection strategy (see SelectionStrategy enum).

        Returns:
            NodeInfo instance to route the request to.

        Raises:
            RuntimeError: If no nodes are available in this group.
        """
        if not self.nodes:
            raise RuntimeError(
                f"No nodes available in group {self.group_id} (layers {self.layer_range})"
            )

        match strategy:
            case SelectionStrategy.LEAST_LOADED:
                return min(self.nodes, key=lambda n: n.active_sessions)
            case SelectionStrategy.ROUND_ROBIN:
                return random.choice(self.nodes)
            case SelectionStrategy.FIRST:
                return self.nodes[0]

    def has_nodes(self) -> bool:
        """Check if this group has any registered nodes."""
        return len(self.nodes) > 0
