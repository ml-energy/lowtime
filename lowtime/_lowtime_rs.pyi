from __future__ import annotations

import networkx as nx

class PhillipsDessouky:
    def __init__(
        self,
        node_ids: list[int] | nx.classes.reportviews.NodeView,
        source_node_id: int,
        sink_node_id: int,
        edges_raw: list[tuple[tuple[int, int], float]],
    ) -> None: ...
    def max_flow(self) -> list[
        tuple[
            tuple[int, int],
            tuple[float, float, float, float],
            tuple[bool, int, int, int, int, int, int, int] | None,
        ]
    ]: ...

# TODO(ohjun): add CostModel interface
