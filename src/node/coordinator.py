"""Pipeline coordinator for managing node groups and routing.

The PipelineCoordinator is responsible for:
- Registering nodes into appropriate groups based on their group_id
- Linking groups in layer order (group 0 → group 1 → ...)
- Providing access to groups for pipeline traversal
"""

import structlog

from node.layer_groups import LAYER_GROUPS
from node.node_group import NodeGroup, NodeInfo

log = structlog.get_logger()


class PipelineCoordinator:
    """Manages node groups and pipeline topology.

    Responsible for:
    - Registering nodes into appropriate groups based on their group_id
    - Linking groups in sequential order (group[0] → group[1] → ...)
    - Providing access to the first group (where requests start)

    Attributes:
        groups: Dictionary mapping group_id to NodeGroup instances.
    """

    def __init__(self):
        """Initialize an empty coordinator."""
        self.groups: dict[int, NodeGroup] = {}

    def register_node(self, node: NodeInfo):
        """Register a node and add it to the appropriate group.

        Creates a new NodeGroup if one doesn't exist for the node's group_id.
        After adding the node, re-links all groups in sequential order.

        Args:
            node: NodeInfo to register.
        """
        group_id = node.group_id

        if group_id not in self.groups:
            self.groups[group_id] = NodeGroup(group_id)
            log.info(
                "group_created",
                group_id=group_id,
                layer_range=self.groups[group_id].layer_range,
            )

        self.groups[group_id].add_node(node)
        self._link_groups()

    def unregister_node(self, node_id: str):
        """Unregister a node from all groups.

        Args:
            node_id: ID of the node to unregister.
        """
        for group in self.groups.values():
            group.remove_node(node_id)

        self.groups = {k: v for k, v in self.groups.items() if v.has_nodes()}
        self._link_groups()

    def _link_groups(self):
        """Connect groups in sequential order based on group_id.

        Builds a linked list: group[0].next_group = group[1], etc.
        This allows iterating through the pipeline by following next_group pointers.
        """
        if not self.groups:
            return

        sorted_groups = sorted(self.groups.items(), key=lambda x: x[0])

        for i in range(len(sorted_groups) - 1):
            current_group = sorted_groups[i][1]
            next_group = sorted_groups[i + 1][1]
            current_group.next_group = next_group

        sorted_groups[-1][1].next_group = None

        log.debug(
            "groups_linked",
            pipeline=[
                f"group_{g.group_id}({g.layer_range[0]}-{g.layer_range[1]})"
                for _, g in sorted_groups
            ],
        )

    def get_first_group(self) -> NodeGroup:
        """Get the group with the lowest group_id (where requests start).

        Returns:
            The NodeGroup handling the earliest layers.

        Raises:
            RuntimeError: If no groups are registered.
        """
        if not self.groups:
            raise RuntimeError("No groups registered in pipeline")
        min_group_id = min(self.groups.keys())
        return self.groups[min_group_id]

    def get_all_groups(self) -> list[NodeGroup]:
        """Get all groups in pipeline order (earliest layers first).

        Returns:
            List of NodeGroup instances, sorted by group_id.
        """
        return [self.groups[gid] for gid in sorted(self.groups.keys())]

    def validate_pipeline(self) -> bool:
        """Check if all required layer groups are present and connected.

        A valid pipeline must have:
        1. At least one node in each group from 0 to max(group_id)
        2. Groups form a contiguous sequence (no gaps)

        Returns:
            True if pipeline is valid, False otherwise.
        """
        if not self.groups:
            log.warning("pipeline_validation_failed", reason="no_groups")
            return False

        group_ids = sorted(self.groups.keys())
        expected_ids = list(range(len(LAYER_GROUPS)))

        # Check for complete coverage (all groups 0..N represented)
        if group_ids != expected_ids:
            missing = set(expected_ids) - set(group_ids)
            log.warning(
                "pipeline_validation_failed",
                reason="missing_groups",
                missing_group_ids=sorted(missing),
            )
            return False

        # Check each group has at least one node
        for gid, group in self.groups.items():
            if not group.has_nodes():
                log.warning(
                    "pipeline_validation_failed",
                    reason="empty_group",
                    group_id=gid,
                )
                return False

        log.info("pipeline_validated", num_groups=len(self.groups))
        return True
