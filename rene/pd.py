"""A linear time-cost trade-off solver using PD-algorithm"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from queue import SimpleQueue

from rene.instruction import Instruction, InstructionType, _Dummy

class PD_Solver:
    """The PD solver for linear time-cost trade-off given an InstructionDAG
    """
    def __init__(
        self,
        entry_node: Instruction
        ) -> None:
        """Create critical path graph, annotate it with lower-bound and upper-bound capacity

        Arguments:
            entry_node: Start node of the InstructionDAG
        """
        self.node_id: int = 0
        self.inst_id_map: dict[str, int] = dict()
        self.critical_graph_aon: nx.DiGraph = nx.DiGraph()

        # Start to construct critical path graph, in AON format
        # This is different than get_critical_path() in InstructionDAG as it allows multiple critcal paths
        q: SimpleQueue[Instruction] = SimpleQueue()
        q.put(entry_node)

        while not q.empty():
            node = q.get()
            print("current ", node.__repr__())
            # Create a new critical node for current instruction
            parent_is_critical: bool = False
            if abs(node.latest_finish - node.earliest_start - node.duration) < 1e-10:
                parent_is_critical = True
                if not isinstance(node, _Dummy) and node.__repr__() not in self.inst_id_map:
                    self.critical_graph_aon.add_node(self.node_id)
                    self.inst_id_map[node.__repr__()] = self.node_id
                    self.node_id += 1
            for child in node.children:
                q.put(child)
                if abs(child.latest_finish - child.earliest_start - child.duration) < 1e-10:
                    # Create a new node for child instruction
                    if not isinstance(child, _Dummy) and child.__repr__() not in self.inst_id_map:
                        self.critical_graph_aon.add_node(self.node_id)
                        self.inst_id_map[child.__repr__()] = self.node_id
                        self.node_id += 1  
                    # Add a critical edge in the graph
                    if not isinstance(node, _Dummy) and not isinstance(child, _Dummy) and parent_is_critical:
                        self.critical_graph_aon.add_edge(self.inst_id_map[node.__repr__()], self.inst_id_map[child.__repr__()]) 

    def draw_aon_graph(self):
        nx.draw(self.critical_graph_aon, with_labels=True, font_weight='bold')
        plt.savefig("aon_graph.png", format="PNG")