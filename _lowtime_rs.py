from __future__ import annotations

class PhillipsDessouky:
    node_ids: list[int]
    
    def __init__(
        self,
        node_ids: list[int],
        source_node_id: int,
        sink_node_id: int,
        edges_raw: list[tuple[tuple[int, int], float]]
    ) -> None: ...

    def max_flow(self) -> list[tuple[tuple[int, int], float]]: ...
