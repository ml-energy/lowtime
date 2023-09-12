from __future__ import annotations

from itertools import count
from typing import TypeVar, Generator

import networkx as nx


NodeT = TypeVar("NodeT")


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

    Assumptions:
        - The given directed graph is a strongly connected DAG.
        - The source node is annotated as "source_node" on the graph.
        - The sink node is annoated as "sink_node" on the graph.
        - Nodes have the specified attribute name, which will be moved to edges.

    Returns:
        The activity-on-arc DAG with "source_node" and "sink_node" annotated as
        graph attributes.
    """
    # Fetch source and sink nodes.
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