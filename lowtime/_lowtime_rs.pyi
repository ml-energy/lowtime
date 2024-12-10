from __future__ import annotations

import networkx as nx

class PhillipsDessouky:
    def __init__(
        self,
        fp_error: float,
        node_ids: list[int] | nx.classes.reportviews.NodeView,
        source_node_id: int,
        sink_node_id: int,
        edges_raw: list[
            tuple[
                tuple[int, int],
                tuple[float, float, float, float],
            ]
        ],
    ) -> None: ...
    def find_min_cut(self) -> tuple[set[int], set[int]]: ...
