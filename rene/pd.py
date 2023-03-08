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
        entry_node: Instruction,
        exit_node: Instruction
        ) -> None:
        """Create critical path graph, annotate it with lower-bound and upper-bound capacity

        Arguments:
            entry_node: Start node of the InstructionDAG
            end_node: End node of the InstructionDAG
        """
        # TODO: change node access to instruction instead of ids
        self.node_id: int = 0
        self.inst_map: dict[str, Instruction] = dict()
        self.inst_id_map: dict[str, int] = dict()
        self.iteration: int = 0
        self.entry_node = entry_node
        self.exit_node = exit_node
        self.run_pd_algorithm()

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
                if node.__repr__() not in self.inst_id_map:
                    dag.add_node(self.node_id, repr=node.__repr__())
                    self.inst_map[node.__repr__()] = node
                    self.inst_id_map[node.__repr__()] = self.node_id
                    self.node_id += 1
            for child in node.children:
                q.put(child)
                if abs(child.latest_finish - child.earliest_start - child.duration) < 1e-10:
                    # Create a new critical node for child instruction
                    if child.__repr__() not in self.inst_id_map:
                        dag.add_node(self.node_id, repr=child.__repr__())
                        self.inst_map[child.__repr__()] = child
                        self.inst_id_map[child.__repr__()] = self.node_id
                        self.node_id += 1  
                    # Add a critical edge in the graph if both parent and child are critical
                    if parent_is_critical:
                        dag.add_edge(self.inst_id_map[node.__repr__()], self.inst_id_map[child.__repr__()], weight=0.0, inst=_Dummy(-1, -1, duration=0, min_duration=0, max_duration=float('inf'))) 
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
            for succ_id in succ_ids:
                q.put(succ_id)
            if cur_inst is self.entry_node or cur_inst is self.exit_node:
                continue
            # Remove current node
            dag.remove_node(cur_id)
            # Split node
            left_id = self.node_id
            right_id = left_id + 1
            dag.add_node(left_id, inst=cur_inst)
            dag.add_node(right_id, inst=cur_inst)
            # Create activity-on-edge
            dag.add_edge(left_id, right_id, weight=cur_inst.duration, inst=cur_inst)
            # print(f"add edge {left_id}, {right_id}, weight {cur_inst.duration}")
            self.node_id += 2
            # Reconnect with predecessors and successors
            for pred_id in pred_ids:
                dag.add_edge(pred_id, left_id, weight=0.0, inst=_Dummy(-1, -1, duration=0, min_duration=0, max_duration=float('inf')))
            for succ_id in succ_ids:
                dag.add_edge(right_id, succ_id, weight=0.0, inst=_Dummy(-1, -1, duration=0, min_duration=0, max_duration=float('inf')))
            targets.remove(cur_id)

        return dag

    def generate_capacity_graph(self) -> nx.DiGraph:
        # Require self.critical_graph_aoa to be present
        cap_graph: nx.DiGraph = nx.DiGraph(self.critical_graph_aoa)
        # Relabel all nodes
        node_ids: list[int] = list(cap_graph.nodes)
        mapping: dict = dict()
        index = 0
        for node_id in node_ids:
            mapping[node_id] = index
            index += 1
        cap_graph = nx.relabel_nodes(cap_graph, mapping)
        # print(list(cap_graph.nodes))
        self.inst_id_map["Entry"] = mapping[self.inst_id_map["Entry"]]
        self.inst_id_map["Exit"] = mapping[self.inst_id_map["Exit"]]

        q: SimpleQueue[int] = SimpleQueue()
        q.put(0)
        while not q.empty():
            cur_id: int = q.get()
            for succ_id in list(cap_graph.successors(cur_id)):
                q.put(succ_id)
                if isinstance(cap_graph[cur_id][succ_id]["inst"], _Dummy) or cap_graph[cur_id][succ_id]["inst"].max_duration == cap_graph[cur_id][succ_id]["inst"].duration and cap_graph[cur_id][succ_id]["inst"].min_duration == cap_graph[cur_id][succ_id]["inst"].duration:
                    cap_graph[cur_id][succ_id]["lb"]: float = 0.0
                    cap_graph[cur_id][succ_id]["ub"]: float = float('inf')
                elif cap_graph[cur_id][succ_id]["inst"].duration == cap_graph[cur_id][succ_id]["inst"].min_duration:
                    cap_graph[cur_id][succ_id]["lb"]: float = cap_graph[cur_id][succ_id]['inst'].unit_cost
                    cap_graph[cur_id][succ_id]["ub"]: float = float('inf')
                elif cap_graph[cur_id][succ_id]["inst"].duration == cap_graph[cur_id][succ_id]["inst"].max_duration:
                    cap_graph[cur_id][succ_id]["lb"]: float = 0.0
                    cap_graph[cur_id][succ_id]["ub"]: float = cap_graph[cur_id][succ_id]['inst'].unit_cost
                else:
                    cap_graph[cur_id][succ_id]["lb"]: float = cap_graph[cur_id][succ_id]['inst'].unit_cost        
                    cap_graph[cur_id][succ_id]["ub"]: float = cap_graph[cur_id][succ_id]['inst'].unit_cost             

        # Change weight to max capacity
        for u, v in cap_graph.edges:
            cap_graph[u][v]["weight"] = cap_graph[u][v]["ub"]
        
        return cap_graph


    def search_path_bfs(self, graph: nx.DiGraph, s: int, t: int) -> tuple(bool, list[int]):
        parents: list[int] = [-1] * graph.number_of_nodes()
        visited: list[bool] = [False] * graph.number_of_nodes()
        # print(list(graph.nodes))
        q: SimpleQueue[int] = SimpleQueue()
        q.put(s)
        while not q.empty():
            cur_id = q.get()
            visited[cur_id] = True
            if cur_id == t:
                break
            for child_id in list(graph.successors(cur_id)):
                if visited[child_id] == False and graph[cur_id][child_id]["weight"] > 0:
                    parents[child_id] = cur_id
                    q.put(child_id)

        return (visited, parents)

    def find_min_cut(self) -> tuple(set[int], set[int]):
        residual_graph: nx.DiGraph = nx.DiGraph(self.capacity_graph)
        entry_id = self.inst_id_map["Entry"]
        exit_id = self.inst_id_map["Exit"]
        while True:
            # Step 1: get a path from entry to exit
            visited, parents = self.search_path_bfs(residual_graph, entry_id, exit_id)
            if visited[exit_id] == False:
                break
            # Step 2: find min capacity along the path
            right_ptr = exit_id
            left_ptr = parents[exit_id]
            path = [right_ptr, left_ptr]
            min_capacity = residual_graph[left_ptr][right_ptr]["weight"]
            while left_ptr != entry_id:
                right_ptr = left_ptr
                left_ptr = parents[left_ptr]
                path.append(left_ptr)
                min_capacity = min(residual_graph[left_ptr][right_ptr]["weight"], min_capacity)
            print(f"path  {path}")
            
            # Step 3: update residual graph
            right_ptr = exit_id
            left_ptr = parents[exit_id]
            while True:
                
                residual_graph[left_ptr][right_ptr]["weight"] -= min_capacity
                # Create residual edge if needed
                if residual_graph.has_edge(right_ptr, left_ptr) == False:
                    residual_graph.add_edge(right_ptr, left_ptr, weight=min_capacity, inst=residual_graph[left_ptr][right_ptr]["inst"])
                else:
                    residual_graph[right_ptr][left_ptr]["weight"] += min_capacity
                if left_ptr == entry_id:
                    break
                right_ptr = left_ptr
                left_ptr = parents[left_ptr]
            
        # Step 4: find the cut:
        visited, _ = self.search_path_bfs(residual_graph, entry_id, exit_id)
        s_set: set[int] = set()
        t_set: set[int] = set()
        for i in range(len(visited)):
            if visited[i] == True:
                s_set.add(i)
            else:
                t_set.add(i)
        print(f"Iteration {self.iteration}: s_set: ",s_set)
        print(f"Iteration {self.iteration}: t_set ", t_set)
        return (s_set, t_set)
    
    def run_pd_algorithm(self) -> None:
        # TODO: need to start the following iterations using the assigned flows
        while True:
            self.clear_annotations()
            self.inst_id_map.clear()
            self.inst_map.clear()
            self.node_id = 0
            self.annotate_nodes()
            self.critical_graph_aon: nx.DiGraph = self.generate_aon_graph()
            self.draw_aon_graph()
            # print("aon", list(self.critical_graph_aon.edges()))
            self.critical_graph_aoa: nx.DiGraph = self.aon_to_aoa()
            self.draw_aoa_graph()
            self.capacity_graph: nx.DiGraph = self.generate_capacity_graph()
            self.draw_capacity_graph()
            s_set, t_set = self.find_min_cut()
            total_cost = self.reduce_duration(s_set, t_set)
            print(f"Iteration {self.iteration}: total cost {total_cost}")
            if total_cost == float('inf'):
                break
            self.iteration += 1

    def annotate_nodes(self) -> None:
        """Annotate earliest/latest start/finish/slack times in nodes.
        """
        # Forward computation: Assign earliest start and finish times
        self.entry_node.earliest_start = 0.0
        self.entry_node.earliest_finish = self.entry_node.earliest_start + self.entry_node.duration
        q: SimpleQueue[Instruction] = SimpleQueue()
        q.put(self.entry_node)

        while not q.empty():
            node = q.get()
            for child in node.children:
                child.earliest_start = max(child.earliest_start, node.earliest_finish)
                child.earliest_finish = child.earliest_start + child.duration
                q.put(child)

        # Backward computation: Assign latest start and finish times
        # Exit node has duration 0, so latest finish and latest start should be the same.
        self.exit_node.latest_finish = (
            self.exit_node.latest_start
        ) = self.exit_node.earliest_start
        q.put(self.exit_node)

        while not q.empty():
            node = q.get()
            for child in node.parents:
                child.latest_start = min(
                    child.latest_start, node.latest_start - child.duration
                )
                child.latest_finish = child.latest_start + child.duration
                child.slack = child.latest_finish - child.earliest_start - child.duration
                q.put(child)
    
    def clear_annotations(self) -> None:
        q: SimpleQueue[Instruction] = SimpleQueue()
        q.put(self.entry_node)

        while not q.empty():
            cur_node = q.get()
            cur_node.earliest_start = 0.0
            cur_node.latest_start = float("inf")
            cur_node.earliest_finish = 0.0
            cur_node.latest_finish = float("inf")
            cur_node.slack = 0.0
            for child in cur_node.children:
                q.put(child)
    
    def reduce_duration(self, s: set[int], t: set[int]) -> float:
        reduce_edges: list[tuple(int, int)] = list()
        increase_edges: list[tuple(int, int)] = list()
        for node_id in s:
            for child_id in list(self.capacity_graph.successors(node_id)):
                if child_id in t:
                    reduce_edges.append((node_id, child_id))
        
        for node_id in t:
            for child_id in list(self.capacity_graph.successors(node_id)):
                if child_id in s:
                    increase_edges.append((node_id, child_id))
        
        print(f"Iteration {self.iteration}: reduce edges {reduce_edges}")
        print(f"Iteration {self.iteration}: increase edges {increase_edges}")

        total_cost = 0

        for u, v in reduce_edges:
            inst: Instruction = self.capacity_graph[u][v]["inst"]
            if inst.min_duration == inst.duration:
                return float('inf')
            else:
                inst.duration -= 1
                total_cost += inst.unit_cost


        for u, v in increase_edges:
            inst: Instruction = self.capacity_graph[u][v]["inst"]
            if inst.duration < inst.max_duration:
                inst.duration += 1
                total_cost += inst.unit_cost
        
        return total_cost

    def draw_aon_graph(self) -> None:
        pos = nx.spring_layout(self.critical_graph_aon)
        nx.draw(self.critical_graph_aon, pos, with_labels=True, font_weight='bold')
        labels = nx.get_node_attributes(self.critical_graph_aon, "repr")
        nx.draw_networkx_labels(self.critical_graph_aon, pos, labels=labels)
        plt.tight_layout()
        plt.savefig(f"aon_graph_{self.iteration}.png", format="PNG")
        plt.clf()

    def draw_aoa_graph(self) -> None:
        pos = nx.spring_layout(self.critical_graph_aoa)
        nx.draw(self.critical_graph_aoa, pos, with_labels=True, font_weight='bold')
        node_labels = nx.get_node_attributes(self.critical_graph_aoa, "repr")
        nx.draw_networkx_labels(self.critical_graph_aoa, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.critical_graph_aoa, "weight")
        nx.draw_networkx_edge_labels(self.critical_graph_aoa, pos, edge_labels=edge_labels)
        plt.tight_layout()
        plt.savefig(f"aoa_graph_{self.iteration}.png", format="PNG")
        plt.clf()

    def draw_capacity_graph(self) -> None:
        pos = nx.circular_layout(self.capacity_graph)
        nx.draw(self.capacity_graph, pos, with_labels=True, font_weight='bold')
        # node_labels = nx.get_node_attributes(self.critical_graph_aoa, "repr")
        # nx.draw_networkx_labels(self.critical_graph_aoa, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.capacity_graph, "weight")
        nx.draw_networkx_edge_labels(self.capacity_graph, pos, edge_labels=edge_labels)
        plt.tight_layout()
        plt.savefig(f"capacity_graph_{self.iteration}.png", format="PNG")
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