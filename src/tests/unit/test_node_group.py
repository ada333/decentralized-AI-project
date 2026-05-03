"""Unit tests for NodeGroup and PipelineCoordinator."""

import pytest

from node.coordinator import PipelineCoordinator
from node.node_group import NodeGroup, NodeInfo, SelectionStrategy


def test_node_group_add_node():
    """NodeGroup.add_node() registers a node."""
    group = NodeGroup(group_id=0)  # Group 0 is layers 0-10
    node = NodeInfo("node1", "127.0.0.1", 8000, group_id=0)

    group.add_node(node)

    assert len(group.nodes) == 1
    assert group.nodes[0] is node


def test_node_group_rejects_wrong_group_id():
    """NodeGroup.add_node() rejects nodes with wrong group_id."""
    group = NodeGroup(group_id=0)
    node = NodeInfo("node1", "127.0.0.1", 8000, group_id=1)

    with pytest.raises(ValueError, match="group_id"):
        group.add_node(node)


def test_node_group_remove_node():
    """NodeGroup.remove_node() unregisters a node by ID."""
    group = NodeGroup(group_id=0)
    node1 = NodeInfo("node1", "127.0.0.1", 8000, group_id=0)
    node2 = NodeInfo("node2", "127.0.0.1", 8001, group_id=0)

    group.add_node(node1)
    group.add_node(node2)
    group.remove_node("node1")

    assert len(group.nodes) == 1
    assert group.nodes[0].node_id == "node2"


def test_node_group_get_available_node_least_loaded():
    """get_available_node() with least_loaded strategy picks node with fewest sessions."""
    group = NodeGroup(group_id=0)
    node1 = NodeInfo("node1", "127.0.0.1", 8000, group_id=0, active_sessions=5)
    node2 = NodeInfo("node2", "127.0.0.1", 8001, group_id=0, active_sessions=2)
    node3 = NodeInfo("node3", "127.0.0.1", 8002, group_id=0, active_sessions=10)

    group.add_node(node1)
    group.add_node(node2)
    group.add_node(node3)

    selected = group.get_available_node(strategy=SelectionStrategy.LEAST_LOADED)

    assert selected.node_id == "node2"


def test_node_group_get_available_node_first():
    """get_available_node() with first strategy always picks first node."""
    group = NodeGroup(group_id=0)
    node1 = NodeInfo("node1", "127.0.0.1", 8000, group_id=0, active_sessions=10)
    node2 = NodeInfo("node2", "127.0.0.1", 8001, group_id=0, active_sessions=0)

    group.add_node(node1)
    group.add_node(node2)

    selected = group.get_available_node(strategy=SelectionStrategy.FIRST)

    assert selected.node_id == "node1"


def test_node_group_get_available_node_raises_when_empty():
    """get_available_node() raises RuntimeError when group is empty."""
    group = NodeGroup(group_id=0)

    with pytest.raises(RuntimeError, match="No nodes available"):
        group.get_available_node()


def test_pipeline_coordinator_register_node():
    """PipelineCoordinator.register_node() creates groups and links them."""
    coordinator = PipelineCoordinator()

    node1 = NodeInfo("node1", "127.0.0.1", 8000, group_id=0)
    node2 = NodeInfo("node2", "127.0.0.1", 8001, group_id=1)
    node3 = NodeInfo("node3", "127.0.0.1", 8002, group_id=0)  # Same group as node1

    coordinator.register_node(node1)
    coordinator.register_node(node2)
    coordinator.register_node(node3)

    # Two groups created: group 0 and group 1
    assert len(coordinator.groups) == 2
    assert 0 in coordinator.groups
    assert 1 in coordinator.groups

    # Group 0 has two nodes
    assert len(coordinator.groups[0].nodes) == 2

    # Group 1 has one node
    assert len(coordinator.groups[1].nodes) == 1


def test_pipeline_coordinator_links_groups():
    """PipelineCoordinator._link_groups() connects groups in sequential order."""
    coordinator = PipelineCoordinator()

    coordinator.register_node(NodeInfo("node1", "127.0.0.1", 8000, group_id=2))
    coordinator.register_node(NodeInfo("node2", "127.0.0.1", 8001, group_id=0))
    coordinator.register_node(NodeInfo("node3", "127.0.0.1", 8002, group_id=1))

    # Groups should be linked 0 -> 1 -> 2 -> None
    group_0 = coordinator.groups[0]
    group_1 = coordinator.groups[1]
    group_2 = coordinator.groups[2]

    assert group_0.next_group is group_1
    assert group_1.next_group is group_2
    assert group_2.next_group is None


def test_pipeline_coordinator_get_first_group():
    """get_first_group() returns the group with the lowest group_id."""
    coordinator = PipelineCoordinator()

    coordinator.register_node(NodeInfo("node1", "127.0.0.1", 8000, group_id=2))
    coordinator.register_node(NodeInfo("node2", "127.0.0.1", 8001, group_id=0))
    coordinator.register_node(NodeInfo("node3", "127.0.0.1", 8002, group_id=1))

    first_group = coordinator.get_first_group()

    assert first_group.group_id == 0
    assert first_group.layer_range == (0, 10)  # Group 0 is layers 0-10


def test_pipeline_coordinator_get_first_group_raises_when_empty():
    """get_first_group() raises RuntimeError when no groups are registered."""
    coordinator = PipelineCoordinator()

    with pytest.raises(RuntimeError, match="No groups registered"):
        coordinator.get_first_group()


def test_pipeline_coordinator_unregister_node():
    """unregister_node() removes node and cleans up empty groups."""
    coordinator = PipelineCoordinator()

    node1 = NodeInfo("node1", "127.0.0.1", 8000, group_id=0)
    node2 = NodeInfo("node2", "127.0.0.1", 8001, group_id=1)

    coordinator.register_node(node1)
    coordinator.register_node(node2)

    # Unregister node1 (the only node in group 0)
    coordinator.unregister_node("node1")

    # Group 0 should be removed
    assert 0 not in coordinator.groups
    assert 1 in coordinator.groups


def test_pipeline_coordinator_get_all_groups():
    """get_all_groups() returns all groups sorted by group_id."""
    coordinator = PipelineCoordinator()

    coordinator.register_node(NodeInfo("node1", "127.0.0.1", 8000, group_id=2))
    coordinator.register_node(NodeInfo("node2", "127.0.0.1", 8001, group_id=0))
    coordinator.register_node(NodeInfo("node3", "127.0.0.1", 8002, group_id=1))

    groups = coordinator.get_all_groups()

    assert len(groups) == 3
    assert groups[0].group_id == 0
    assert groups[0].layer_range == (0, 10)
    assert groups[1].group_id == 1
    assert groups[1].layer_range == (10, 20)
    assert groups[2].group_id == 2
    assert groups[2].layer_range == (20, 30)


def test_node_info_invalid_group_id_shows_context():
    """NodeInfo with invalid group_id provides helpful error message with node context."""
    with pytest.raises(
        ValueError, match=r"Node 'bad_node' at 192\.168\.1\.10:8000.*Invalid group_id 999"
    ):
        NodeInfo("bad_node", "192.168.1.10", 8000, group_id=999)


def test_node_group_invalid_group_id_shows_context():
    """NodeGroup with invalid group_id provides helpful error message."""
    with pytest.raises(ValueError, match=r"Cannot create NodeGroup.*Invalid group_id 999"):
        NodeGroup(group_id=999)
