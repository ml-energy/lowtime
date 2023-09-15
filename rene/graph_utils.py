from __future__ import annotations

import logging
from itertools import count
from typing import Any, Literal, TypeVar, Generator, TYPE_CHECKING

import networkx as nx
import matplotlib.pyplot as plt

from rene.operation import DummyOperation
from rene.exceptions import ReneGraphError

if TYPE_CHECKING:
    from rene.operation import Operation

logger = logging.getLogger(__name__)


NodeT = TypeVar("NodeT")


def add_source_node(graph: nx.DiGraph, source_node: Any) -> None:
    """Add a source node to the given graph.

    The graph may have multiple source nodes, and this function adds a new source
    node that is connected to all existing source nodes.
    """
    graph.add_node(source_node, op=DummyOperation())
    for node_id, in_degree in graph.in_degree():
        if node_id != source_node and in_degree == 0:
            graph.add_edge(source_node, node_id)


def add_sink_node(graph: nx.DiGraph, sink_node: Any) -> None:
    """Add a sink node to the given graph.

    The graph may have multiple sink nodes, and this function adds a new sink
    node that is connected to all existing sink nodes.
    """
    graph.add_node(sink_node, op=DummyOperation())
    for node_id, out_degree in graph.out_degree():
        if node_id != sink_node and out_degree == 0:
            graph.add_edge(node_id, sink_node)


def bfs_nodes(graph: nx.Graph, source_node: NodeT) -> Generator[NodeT, None, None]:
    """Yield nodes in the order they are visited by BFS.

    Reference:
    Example in
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_edges.html
    """
    yield source_node
    for _, v in nx.bfs_edges(graph, source_node):
        yield v


def relabel_nodes_to_int(graph: nx.DiGraph) -> tuple[nx.DiGraph, dict[Any, int]]:
    """Relabel the node IDs of a graph so that they are consecutive integers.

    Keeps "source_node" and "sink_node" graph attributes consistent.

    Returns:
        The relabeled graph (copied) and the mapping from the original node IDs
        to new ones.
    """
    node_id_counter = count()
    mapping = {node_id: next(node_id_counter) for node_id in graph.nodes}
    relabeled_graph = nx.relabel_nodes(graph, mapping, copy=True)
    relabeled_graph.graph["source_node"] = mapping[graph.graph["source_node"]]
    relabeled_graph.graph["sink_node"] = mapping[graph.graph["sink_node"]]
    return relabeled_graph, mapping


def aon_dag_to_aoa_dag(
    aon: nx.DiGraph,
    attr_name: str = "op",
) -> nx.DiGraph:
    """Convert an activity-on-node DAG to an activity-on-arc DAG.

    Node attributes keyed by `attr_name` are moved to edge attributes.

    Assumptions:
        - The given directed graph is a DAG.
        - The source node is annotated as "source_node" on the graph.
        - The sink node is annoated as "sink_node" on the graph.

    Returns:
        The activity-on-arc DAG with "source_node" and "sink_node" annotated as
        graph attributes.
    """
    if not nx.is_directed_acyclic_graph(aon):
        raise ValueError("The given graph is not a DAG.")

    if "source_node" not in aon.graph or "sink_node" not in aon.graph:
        raise ValueError(
            "`source_node` and `sink_node` must be annotated on the graph."
        )

    # Relabel the original AON so that node IDs are all integers.
    # This allows us to start creating integer node IDs that are guaranteed to
    # not overlap with any existing nodes.
    aon, mapping = relabel_nodes_to_int(aon)
    source_node = aon.graph["source_node"]
    sink_node = aon.graph["sink_node"]

    # Walk the AON DAG in BFS order and convert each node by splitting it into
    # two nodes (left and right) and connecting the left one with original
    # predecessors and the right one with original successors.
    aoa = nx.DiGraph(aon)
    new_source_node, new_sink_node = None, None
    node_id_counter = count(start=max(mapping.values()) + 1)
    for cur_aon_id in bfs_nodes(aon, source_node):
        # Attribute on the current node to move to the edge.
        op = aon.nodes[cur_aon_id][attr_name]
        attr = {attr_name: op}

        # Needed to re-connect split nodes to the rest of the graph.
        pred_aon_ids = list(aoa.predecessors(cur_aon_id))
        succ_aon_ids = list(aoa.successors(cur_aon_id))

        # Remove the current AON node from the AOA DAG.
        aoa.remove_node(cur_aon_id)

        # Split the node into two, left and right, and connect them.
        left_id, right_id = next(node_id_counter), next(node_id_counter)
        # TODO(JW): I don't think we need to add attributes to the left and right
        # nodes. First keep it and later check whether removing it is fine.
        aoa.add_node(left_id, **attr)
        aoa.add_node(right_id, **attr)
        # NOTE(JW): The old implementation added weight=duration, but I'm pretty
        # sure that's not used anywhere.
        aoa.add_edge(left_id, right_id, **attr)
        logger.debug("Connecting %d -> %d (%s)", left_id, right_id, str(op))

        # Track the new source and sink nodes.
        if cur_aon_id == source_node:
            new_source_node = left_id
        if cur_aon_id == sink_node:
            new_sink_node = right_id

        # Connect the left node to the predecessors of the current node.
        for pred_aon_id in pred_aon_ids:
            dummy_attr = {attr_name: DummyOperation()}
            aoa.add_edge(pred_aon_id, left_id, **dummy_attr)
            logger.debug("Connecting %d -> %d (DummyOperation())", pred_aon_id, left_id)

        # Connect the successors of the current node to the right node.
        for succ_aon_id in succ_aon_ids:
            dummy_attr = {attr_name: DummyOperation()}
            aoa.add_edge(right_id, succ_aon_id, **dummy_attr)
            logger.debug(
                "Connecting %d -> %d (DummyOperation())", right_id, succ_aon_id
            )

    aoa.graph["source_node"] = new_source_node
    aoa.graph["sink_node"] = new_sink_node

    # Do a final relabel so that all node IDs are consecutive integers from zero.
    aoa, _ = relabel_nodes_to_int(aoa)

    logger.info("Converted AON to AOA.")
    logger.debug("%d nodes and %d edges.", aoa.number_of_nodes(), aoa.number_of_edges())
    logger.debug("Source node ID: %s, sink node ID: %s", new_source_node, new_sink_node)

    # Sanity checks.
    if new_source_node is None or new_sink_node is None:
        raise ReneGraphError(
            "New source and sink nodes could not be determined. "
            "Check whether the source and sink nodes were in the original graph."
        )

    # Check source/sink node correctness and membership.
    if aoa.in_degree(aoa.graph["source_node"]) != 0:
        raise ReneGraphError("The new source node has incoming edges.")
    if aoa.out_degree(aoa.graph["sink_node"]) != 0:
        raise ReneGraphError("The new sink node has outgoing edges.")

    # The graph should be one piece.
    if not nx.is_weakly_connected(aoa):
        raise ReneGraphError("The converted graph is not connected.")

    # It should still be a DAG.
    if not nx.is_directed_acyclic_graph(aoa):
        raise ReneGraphError("The converted graph is not a DAG.")

    # All nodes are split into two nodes.
    # All original edges are intact and each original node contributes one edge.
    aon_nnodes = aon.number_of_nodes()
    aon_nedges = aon.number_of_edges()
    aoa_nnodes = aoa.number_of_nodes()
    aoa_nedges = aoa.number_of_edges()
    if aon_nnodes * 2 != aoa_nnodes:
        raise ReneGraphError(f"Expected {aon_nnodes * 2} nodes, got {aoa_nnodes}.")
    if aon_nedges + aon_nnodes != aoa_nedges:
        raise ReneGraphError(
            f"Expected {aon_nedges + aon_nnodes} edges, got {aoa_nedges}."
        )
    
    logger.info("All sanity checks passed.")

    return aoa


def get_total_cost(graph: nx.DiGraph, mode: Literal["edge", "node"]) -> float:
    """Return the total cost of the given graph.
    
    Assumptions:
        - The graph has a "op" attribute on each edge that holds `Operation`.
    """
    if mode == "edge":
        cost = 0.0
        for _, _, edge_attr in graph.edges(data=True):
            op: Operation = edge_attr["op"]
            if op.is_dummy:
                continue
            cost += op.get_cost()
        return cost
    elif mode == "node":
        cost = 0.0
        for _, node_attr in graph.nodes(data=True):
            op: Operation = node_attr["op"]
            if op.is_dummy:
                continue
            cost += op.get_cost()
        return cost


def get_total_time(critical_dag: nx.DiGraph) -> int:
    """Find the total execution time of the given critical DAG."""
    source_node = critical_dag.graph["source_node"]
    sink_node = critical_dag.graph["sink_node"]

    # Any path from the source to sink node is a critical path, so we can just
    # traverse in a DFS order and when we reach the sink node, we're done.
    total_time = 0
    for cur_node, next_node in nx.dfs_edges(critical_dag, source_node):
        op: Operation = critical_dag[cur_node][next_node]["op"]
        if not op.is_dummy:
            total_time += op.duration
        if next_node == sink_node:
            break

    return total_time