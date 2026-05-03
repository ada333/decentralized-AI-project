"""Layer group definitions for the SmolLM-135M model pipeline.

Layer groups define contiguous ranges of transformer layers that can be assigned
to nodes. All nodes in the network must agree on these predefined groups to ensure
consistent pipeline topology.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerGroupConfig:
    """Defines a contiguous range of transformer layers.

    Attributes:
        group_id: Unique identifier for this group (used by nodes to declare capabilities).
        start_layer: First layer in this range (inclusive).
        end_layer: Last layer in this range (exclusive).
    """

    group_id: int
    start_layer: int
    end_layer: int


# Predefined layer groups for SmolLM-135M (30 layers total)
# Each group handles 10 layers for balanced compute distribution
LAYER_GROUPS: dict[int, LayerGroupConfig] = {
    0: LayerGroupConfig(group_id=0, start_layer=0, end_layer=10),
    1: LayerGroupConfig(group_id=1, start_layer=10, end_layer=20),
    2: LayerGroupConfig(group_id=2, start_layer=20, end_layer=30),
}


def get_layer_group(group_id: int) -> LayerGroupConfig:
    """Lookup a layer group by ID.

    Args:
        group_id: The group ID to lookup.

    Returns:
        LayerGroupConfig for the requested group.

    Raises:
        ValueError: If group_id is not in LAYER_GROUPS.
    """
    if group_id not in LAYER_GROUPS:
        valid_ids = sorted(LAYER_GROUPS.keys())
        raise ValueError(f"Invalid group_id {group_id}. Valid groups: {valid_ids}")
    return LAYER_GROUPS[group_id]
