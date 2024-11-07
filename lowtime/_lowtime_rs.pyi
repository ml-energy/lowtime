from __future__ import annotations

import networkx as nx

class PhillipsDessouky:
    node_ids: list[int]

    def __init__(
        self,
        node_ids: list[int] | nx.classes.reportviews.NodeView,
        source_node_id: int,
        sink_node_id: int,
        edges_raw: list[tuple[tuple[int, int], float]],
    ) -> None: ...
    def max_flow(self) -> list[tuple[tuple[int, int], float]]: ...
