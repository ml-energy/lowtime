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
        # TODO: change node access to instruction instead of ids
        self.node_id: int = 0
        self.inst_map: dict[str, Instruction] = dict()
        self.inst_id_map: dict[str, int] = dict()
        self.entry_node = entry_node
        self.critical_graph_aon: nx.DiGraph = self.generate_aon_graph()
        self.draw_aon_graph()
        # print("aon", list(self.critical_graph_aon.edges()))
        self.critical_graph_aoa: nx.DiGraph = self.aon_to_aoa()
        self.draw_aoa_graph()
        # print("aoa", list(self.critical_graph_aoa.edges()))

    def generate_aon_graph(self) -> nx.DiGraph:
        dag: nx.DiGraph = nx.DiGraph()
        # Start to construct critical path graph, in AON format
        # This is different than get_critical_path() in InstructionDAG as it allows multiple critcal paths
        q: SimpleQueue[Instruction] = SimpleQueue()
        q.put(self.entry_node)

        while not q.empty():
            node = q.get()
            # print("current ", node.__repr__())
            parent_is_critical: bool = False
            if abs(node.latest_finish - node.earliest_start - node.duration) < 1e-10:
                parent_is_critical = True
                # Create a new critical node for current instruction
                if not isinstance(node, _Dummy) and node.__repr__() not in self.inst_id_map:
                    dag.add_node(self.node_id, repr=node.__repr__())
                    self.inst_map[node.__repr__()] = node
                    self.inst_id_map[node.__repr__()] = self.node_id
                    self.node_id += 1
            for child in node.children:
                q.put(child)
                if abs(child.latest_finish - child.earliest_start - child.duration) < 1e-10:
                    # Create a new critical node for child instruction
                    if not isinstance(child, _Dummy) and child.__repr__() not in self.inst_id_map:
                        dag.add_node(self.node_id, repr=child.__repr__())
                        self.inst_map[child.__repr__()] = child
                        self.inst_id_map[child.__repr__()] = self.node_id
                        self.node_id += 1  
                    # Add a critical edge in the graph if both parent and child are critical
                    if not isinstance(node, _Dummy) and not isinstance(child, _Dummy) and parent_is_critical:
                        dag.add_edge(self.inst_id_map[node.__repr__()], self.inst_id_map[child.__repr__()], weight=0.0) 
        return dag
    
    def aon_to_aoa(self) -> nx.DiGraph:
        # TODO: crash dummy nodes for optimization
        # do a BFS to split all nodes and reconnect
        dag: nx.DiGraph = nx.DiGraph(self.critical_graph_aon)
        q: SimpleQueue[int] = SimpleQueue()
        q.put(0)
        # Sanity check
        if len(list(dag.predecessors(0))) != 0:
            raise ValueError(
                        f"The first operation in the AON graph has {len(dag.predecessors(0))} predecessors!"
                    )
        targets = list(dag.nodes)
        while not q.empty():
            cur_id: int = q.get()
            # Skip processed nodes in AON and new nodes in AOA
            if cur_id not in targets:
                continue
            cur_inst: Instruction = self.inst_map[dag.nodes[cur_id]["repr"]]
            # Store current node's predecessors and successors
            pred_ids: list[int] = list(dag.predecessors(cur_id))
            succ_ids: list[int] = list(dag.successors(cur_id))
            # Remove current node
            dag.remove_node(cur_id)
            # Split node
            left_id = self.node_id
            right_id = left_id + 1
            dag.add_node(left_id, repr=cur_inst.__repr__())
            dag.add_node(right_id, repr=cur_inst.__repr__())
            # Create activity-on-edge
            dag.add_edge(left_id, right_id, weight=cur_inst.duration, repr=cur_inst.__repr__())
            self.node_id += 2
            # Reconnect with predecessors and successors
            for pred_id in pred_ids:
                dag.add_edge(pred_id, left_id, weight=0.0, repr="Dummy")
            for succ_id in succ_ids:
                dag.add_edge(right_id, succ_id, weight=0.0, repr="Dummy")
                q.put(succ_id)
            targets.remove(cur_id)

        return dag

        
    def draw_aon_graph(self) -> None:
        pos = nx.spring_layout(self.critical_graph_aon)
        nx.draw(self.critical_graph_aon, pos, with_labels=True, font_weight='bold')
        labels = nx.get_node_attributes(self.critical_graph_aon, "repr")
        nx.draw_networkx_labels(self.critical_graph_aon, pos, labels=labels)
        plt.tight_layout()
        plt.savefig("aon_graph.png", format="PNG")
        plt.clf()

    def draw_aoa_graph(self) -> None:
        pos = nx.spring_layout(self.critical_graph_aoa)
        nx.draw(self.critical_graph_aoa, pos, with_labels=True, font_weight='bold')
        node_labels = nx.get_node_attributes(self.critical_graph_aoa, "repr")
        nx.draw_networkx_labels(self.critical_graph_aoa, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.critical_graph_aoa, "weight")
        nx.draw_networkx_edge_labels(self.critical_graph_aoa, pos, edge_labels=edge_labels)
        plt.tight_layout()
        plt.savefig("aoa_graph.png", format="PNG")
        plt.clf()

def aon_to_aoa_pure() -> nx.DiGraph:
    # TODO: crash dummy nodes for optimization
    # do a BFS to split all nodes and reconnect
    weight_dict = {1: 20, 2: 30, 3: 10, 4: 25}
    dag: nx.DiGraph = nx.DiGraph()
    dag.add_nodes_from([1,2,3,4])
    dag.add_edges_from([(1,2), (1,3), (2,4), (3,4)])
    node_id = 5
    pos = nx.spring_layout(dag)
    nx.draw(dag, pos, with_labels=True, font_weight='bold')
    plt.tight_layout()
    plt.savefig("aon_proof_of_concept.png", format="PNG")
    plt.clf()
    q: SimpleQueue[int] = SimpleQueue()
    q.put(1)
    targets = [1, 2, 3, 4]
    while not q.empty():
        cur_id: int = q.get()
        if cur_id not in targets:
            continue
        print("current: ", cur_id)
        # Store current node's predecessors and successors
        pred_ids: list[int] = list(dag.predecessors(cur_id))
        succ_ids: list[int] = list(dag.successors(cur_id))
        # Remove current node
        dag.remove_node(cur_id)
        # Split node
        left_id = node_id
        right_id = left_id + 1
        dag.add_node(left_id)
        dag.add_node(right_id)
        # Create activity-on-edge
        dag.add_edge(left_id, right_id, weight=weight_dict[cur_id])
        node_id += 2
        # Reconnect with predecessors and successors
        for pred_id in pred_ids:
            dag.add_edge(pred_id, left_id, weight=0.0)
        for succ_id in succ_ids:
            dag.add_edge(right_id, succ_id, weight=0.0)
            q.put(succ_id)
        targets.remove(cur_id)
    pos = nx.spring_layout(dag)
    nx.draw(dag, pos, with_labels=True, font_weight='bold')
    edge_labels = nx.get_edge_attributes(dag, "weight")
    nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels)
    plt.tight_layout()
    plt.savefig("aoa_proof_of_concept.png", format="PNG")
    plt.clf()

if __name__ == "__main__":
    aon_to_aoa_pure()