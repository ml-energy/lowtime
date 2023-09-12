from __future__ import annotations

from itertools import count
from typing import Any, TypeVar, Generator

import networkx as nx

from rene.operation import DummyOperation


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
    # Fetch source and sink nodes.
    if not nx.is_directed_acyclic_graph(aon):
        raise ValueError("The given graph is not a DAG.")
    source_node = aon.graph["source_node"]
    sink_node = aon.graph["sink_node"]

    aoa = nx.DiGraph()
    node_id = count()
    new_source_node = None
    new_sink_node = None
    for cur_id in bfs_nodes(aon, source_node):
        # Attribute on the current node to move to the edge.
        op = aon.nodes[cur_id][attr_name]
        attr = {attr_name: op}

        # Needed to re-connect split nodes to the rest of the graph.
        pred_ids = aon.predecessors(cur_id)
        succ_ids = aon.successors(cur_id)

        # Split the node into two, left and right, and connect them.
        left_id, right_id = next(node_id), next(node_id)
        # TODO(JW): I don't think we need to add attributes to the left and right
        # nodes. First keep it and later check whether removing it is fine.
        aoa.add_node(left_id, **attr)
        aoa.add_node(right_id, **attr)
        # NOTE(JW): The old implementation added weight=duration, but I'm pretty
        # sure that's not used anywhere.
        aoa.add_edge(left_id, right_id, **attr)

        # Track the new source and sink nodes.
        if cur_id == source_node:
            new_source_node = left_id
        if cur_id == sink_node:
            new_sink_node = right_id

        # Connect the left node to the predecessors of the current node.
        for pred_id in pred_ids:
            aoa.add_edge(pred_id, left_id, **attr)

        # Connect the successors of the current node to the right node.
        for succ_id in succ_ids:
            aoa.add_edge(right_id, succ_id, **attr)

    if new_source_node is None or new_sink_node is None:
        raise ValueError(
            "New source and sink nodes could not be determined. "
            "Check whether the source and sink nodes were in the original graph."
        )

    aoa.graph["source_node"] = new_source_node
    aoa.graph["sink_node"] = new_sink_node

    return aoa