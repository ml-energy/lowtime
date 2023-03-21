"""A linear time-cost trade-off solver using PD-algorithm"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure # type: ignore
from matplotlib.ticker import FormatStrFormatter  # type: ignore
from queue import SimpleQueue
from collections import deque
from typing import Generator

from rene.dag import ReneDAG, CriticalDAG
from rene.instruction import Instruction, InstructionType, _Dummy, Forward, Backward

UNIT_SCALE = 0.01

DEFAULT_RECTANGLE_ARGS = {
    Forward: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    Backward: dict(facecolor="#9fc887", edgecolor="#000000", linewidth=1.0),
}

DEFAULT_ANNOTATION_ARGS = {
    Forward: dict(color="#ffffff", fontsize=20.0, ha="center", va="center"),
    Backward: dict(color="#000000", fontsize=20.0, ha="center", va="center"),
}

DEFAULT_LINE_ARGS = dict(color="#ff9900", linewidth=4.0)


class PD_Solver:
    """The PD solver for linear time-cost trade-off given an ReneDAG
    """
    def __init__(
        self,
        critical_dag: CriticalDAG,
        output_dir: str,
        ) -> None:
        """Given a critical graph, iteratively run the PD algorithm and update the instruction durations

        Arguments:
            critical_dag: a critical DAG in the form of ReneDAG
            output_dir: output directory for figures
        """
        # TODO: change node access to instruction instead of ids
        self.iteration: int = 0
        self.output_dir: str = output_dir
        self.critical_dag_aon: CriticalDAG = critical_dag
        self.node_id: int = self.critical_dag_aon.node_id
        self.annotation_args = DEFAULT_ANNOTATION_ARGS
        self.rectangle_args = DEFAULT_RECTANGLE_ARGS
        self.line_args = DEFAULT_LINE_ARGS
        self.entry_id: int = 0
        self.exit_id: int = 1
        # self.run_pd_algorithm()

        # print("aoa", list(self.critical_dag_aoa.edges()))

    def generate_aon_graph(self) -> nx.DiGraph:
        dag: nx.DiGraph = nx.DiGraph()
        # Start to construct critical path graph, in AON format
        # This is different than get_critical_path() in ReneDAG as it allows multiple critcal paths
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
        dag: nx.DiGraph = nx.DiGraph(self.critical_dag_aon.dag)
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
            cur_inst: Instruction = dag.nodes[cur_id]["inst"]
            # Store current node's predecessors and successors
            pred_ids: list[int] = list(dag.predecessors(cur_id))
            succ_ids: list[int] = list(dag.successors(cur_id))
            for succ_id in succ_ids:
                q.put(succ_id)
            if cur_inst is self.critical_dag_aon.entry_node or cur_inst is self.critical_dag_aon.exit_node:
                continue
            # Remove current node
            dag.remove_node(cur_id)
            # Split node
            left_id = self.node_id
            right_id = left_id + 1
            dag.add_node(left_id, inst=cur_inst, repr=cur_inst.__repr__())
            dag.add_node(right_id, inst=cur_inst, repr=cur_inst.__repr__())
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
        # Require self.critical_dag_aoa to be present
        cap_graph: nx.DiGraph = nx.DiGraph(self.critical_dag_aoa)
        # Relabel all nodes
        node_ids: list[int] = list(cap_graph.nodes)
        mapping: dict = dict()
        index = 0
        for node_id in node_ids:
            mapping[node_id] = index
            index += 1
        cap_graph = nx.relabel_nodes(cap_graph, mapping)
        # print(list(cap_graph.nodes))
        self.entry_id = mapping[self.entry_id]
        self.exit_id = mapping[self.exit_id]

        q: SimpleQueue[int] = SimpleQueue()
        q.put(0)
        visited: list[int] = []
        while not q.empty():
            cur_id: int = q.get()
            if cur_id in visited:
                continue
            visited.append(cur_id)
            for succ_id in list(cap_graph.successors(cur_id)):
                q.put(succ_id)
                if isinstance(cap_graph[cur_id][succ_id]["inst"], _Dummy) or abs(cap_graph[cur_id][succ_id]["inst"].max_duration - cap_graph[cur_id][succ_id]["inst"].duration) < 1e-5 and abs(cap_graph[cur_id][succ_id]["inst"].min_duration - cap_graph[cur_id][succ_id]["inst"].duration) < 1e-5:
                    cap_graph[cur_id][succ_id]["lb"]: float = 0.0
                    cap_graph[cur_id][succ_id]["ub"]: float = float('inf')
                elif cap_graph[cur_id][succ_id]["inst"].duration - UNIT_SCALE < cap_graph[cur_id][succ_id]["inst"].min_duration:
                    cap_graph[cur_id][succ_id]["lb"]: float = cap_graph[cur_id][succ_id]['inst'].unit_cost
                    cap_graph[cur_id][succ_id]["ub"]: float = float('inf')
                elif cap_graph[cur_id][succ_id]["inst"].duration + UNIT_SCALE > cap_graph[cur_id][succ_id]["inst"].max_duration:
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

        while True:
            # Step 1: get a path from entry to exit
            visited, parents = self.search_path_bfs(residual_graph, self.entry_id, self.exit_id)
            if visited[self.exit_id] == False:
                break
            # Step 2: find min capacity along the path
            right_ptr = self.exit_id
            left_ptr = parents[self.exit_id]
            path = [right_ptr, left_ptr]
            min_capacity = residual_graph[left_ptr][right_ptr]["weight"]
            while left_ptr != self.entry_id:
                right_ptr = left_ptr
                left_ptr = parents[left_ptr]
                path.append(left_ptr)
                min_capacity = min(residual_graph[left_ptr][right_ptr]["weight"], min_capacity)
            print(f"path  {path}")
            
            # Step 3: update residual graph
            right_ptr = self.exit_id
            left_ptr = parents[self.exit_id]
            while True:
                
                residual_graph[left_ptr][right_ptr]["weight"] -= min_capacity
                # Create residual edge if needed
                if residual_graph.has_edge(right_ptr, left_ptr) == False:
                    residual_graph.add_edge(right_ptr, left_ptr, weight=min_capacity, inst=residual_graph[left_ptr][right_ptr]["inst"])
                else:
                    residual_graph[right_ptr][left_ptr]["weight"] += min_capacity
                if left_ptr == self.entry_id:
                    break
                right_ptr = left_ptr
                left_ptr = parents[left_ptr]
            
        # Step 4: find the cut:
        visited, _ = self.search_path_bfs(residual_graph, self.entry_id, self.exit_id)
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
            self.critical_dag_aon.clear_annotations()
            self.node_id = self.critical_dag_aon.node_id
            self.critical_dag_aon.annotate_nodes()
            # Do eager assignment and draw the current pipeline
            for inst in self.critical_dag_aon.insts:
                inst.actual_start = inst.earliest_start
                inst.actual_finish = inst.earliest_finish
            self.draw_pipeline_graph(os.path.join(self.output_dir, f"pipeline_{self.iteration}.png"), draw_time_axis=True)
            # self.critical_dag_aon.draw_aon_graph(os.path.join(self.output_dir, f"aon_graph_{self.iteration}.png"))
            # print("aon", list(self.critical_graph_aon.edges()))
            self.critical_dag_aoa: nx.DiGraph = self.aon_to_aoa()
            # self.draw_aoa_graph(os.path.join(self.output_dir, f"aoa_graph_{self.iteration}.png"))
            self.capacity_graph: nx.DiGraph = self.generate_capacity_graph()
            # self.draw_capacity_graph(os.path.join(self.output_dir, f"capacity_graph_{self.iteration}.png"))
            s_set, t_set = self.find_min_cut()
            cost_change = self.reduce_duration(s_set, t_set)
            total_cost = self.calculate_total_cost()
            print(f"Iteration {self.iteration}: cost change {cost_change}")
            print(f"Iteration {self.iteration}: total cost {total_cost}")
            if cost_change == float('inf'):
                break
            self.iteration += 1
        # need to output final frequency assignment
        self.assign_frequency()
    
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

        cost_change = 0

        for u, v in reduce_edges:
            inst: Instruction = self.capacity_graph[u][v]["inst"]
            if inst.duration - UNIT_SCALE < inst.min_duration:
                return float('inf')
            else:
                inst.duration -= UNIT_SCALE
                cost_change += inst.unit_cost * UNIT_SCALE


        for u, v in increase_edges:
            inst: Instruction = self.capacity_graph[u][v]["inst"]
            if inst.duration < inst.max_duration:
                inst.duration += UNIT_SCALE
                cost_change += inst.unit_cost * UNIT_SCALE
        
        return cost_change

    def calculate_total_cost(self) -> float:
        q: SimpleQueue[int] = SimpleQueue()
        q.put(0)
        visited: list[str] = list()
        total_cost: float = 0.0

        while not q.empty():
            cur_id: int = q.get()
            cur_node: Instruction = self.critical_dag_aon.complete_dag.nodes[cur_id]["inst"]
            if not isinstance(cur_node, _Dummy) and cur_node.__repr__() not in visited:
                total_cost += cur_node.duration * cur_node.k + cur_node.b
                visited.append(cur_node.__repr__())
            for child_id in self.critical_dag_aon.complete_dag.successors(cur_id):
                q.put(child_id)

        return total_cost
    
    def assign_frequency(self):
        # do binary search on inst.time_costs, list of (duration, cost, frequency) tuples, sorted by reverse duration
        q: SimpleQueue[int] = SimpleQueue()
        q.put(0)
        # stage_id -> list of Instructions with that stage_id
        stage_view: dict[int, list[Instruction]] = dict()
        visited: list[str] = list()
        while not q.empty():
            cur_id: int = q.get()
            cur_node: Instruction = self.critical_dag_aon.complete_dag.nodes[cur_id]["inst"]
            if cur_node.__repr__() in visited:
                continue
            visited.append(cur_node.__repr__())
            for child_id in self.critical_dag_aon.complete_dag.successors(cur_id):
                q.put(child_id)
            if isinstance(cur_node, _Dummy):
                continue
            # max/min duration should be common case
            if abs(cur_node.time_costs[0][0] - cur_node.duration) < 1e-5:
                cur_node.frequency = cur_node.time_costs[0][2]
            elif abs(cur_node.time_costs[-1][0] - cur_node.duration) < 1e-5:
                cur_node.frequency = cur_node.time_costs[-1][2]
            else:
                left = 0
                right = len(cur_node.time_costs) - 1
                while left < right:
                    mid = (left + right) // 2
                    if abs(cur_node.time_costs[mid][0] - cur_node.duration) < 1e-5 or mid == 0 or mid == len(cur_node.time_costs) - 1:
                        cur_node.frequency = cur_node.time_costs[mid][2]
                        break
                    elif cur_node.time_costs[mid][0] < cur_node.duration:
                        if cur_node.time_costs[mid-1][0] > cur_node.duration:
                            mid_duration = (cur_node.time_costs[mid][0] + cur_node.time_costs[mid-1][0]) / 2
                            if mid_duration < cur_node.duration:
                                cur_node.frequency = cur_node.time_costs[mid-1][2]
                            else:
                                cur_node.frequency = cur_node.time_costs[mid][2]
                            break
                        right = mid
                    elif cur_node.time_costs[mid][0] > cur_node.duration:
                        if cur_node.time_costs[mid+1][0] < cur_node.duration:
                            mid_duration = (cur_node.time_costs[mid][0] + cur_node.time_costs[mid+1][0]) / 2
                            if mid_duration < cur_node.duration:
                                cur_node.frequency = cur_node.time_costs[mid][2]
                            else:
                                cur_node.frequency = cur_node.time_costs[mid+1][2]
                            break
                        left = mid + 1
                                   
            if cur_node.stage_id not in stage_view:
                stage_view[cur_node.stage_id] = [cur_node]
            else:
                stage_view[cur_node.stage_id].append(cur_node)

        for stage_id, insts in stage_view.items():
            print(f"Stage {stage_id} frequency assignment ")
            freqs = []
            reprs = []
            for inst in insts:
                assert(inst.frequency != -1)
                freqs.append(inst.frequency)
                reprs.append(inst.__repr__())
            print(f"Freqs: {freqs}")
            print(f"Reprs: {reprs}")

    def draw_aoa_graph(self, path: str) -> None:
        pos = nx.spring_layout(self.critical_dag_aoa)
        nx.draw(self.critical_dag_aoa, pos, with_labels=True, font_weight='bold')
        node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
        nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.critical_dag_aoa, "weight")
        nx.draw_networkx_edge_labels(self.critical_dag_aoa, pos, edge_labels=edge_labels)
        plt.tight_layout()
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_capacity_graph(self, path: str) -> None:
        pos = nx.circular_layout(self.capacity_graph)
        nx.draw(self.capacity_graph, pos, with_labels=True, font_weight='bold')
        # node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
        # nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.capacity_graph, "weight")
        nx.draw_networkx_edge_labels(self.capacity_graph, pos, edge_labels=edge_labels)
        plt.tight_layout()
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_pipeline_graph(self, path: str, draw_time_axis: bool = False) -> None:
        """Draw the pipeline on the given Axes object."""
        fig, ax = plt.subplots(figsize=(12, 4), tight_layout=True)
        ax.set_xlim(0, 23)
        for inst in self.critical_dag_aon.insts:
            # Draw rectangle for Instructions
            inst.draw(ax, self.rectangle_args, self.annotation_args)

        if draw_time_axis:
            ax.yaxis.set_visible(False)
            ax.grid(visible=False)

            total_time = self.critical_dag_aon.exit_node.earliest_finish
            ax.set_xlabel("time")
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_xticks(
                [float(t * 5) for t in range(int(total_time) // 5)] + [total_time]
            )

            for side in ["top", "left", "right"]:
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_bounds(0.0, total_time)
        else:
            ax.set_axis_off()

        ax.autoscale()
        ax.invert_yaxis()
        self.draw_critical_path(ax)
        fig.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_critical_path(self, ax: Axes) -> None:
        """Draw the critical path of the DAG on the given Axes object."""
        critical_path = self.get_critical_path()
        for inst1, inst2 in zip(critical_path, critical_path[1:]):
            ax.plot(
                [
                    (inst1.actual_start + inst1.actual_finish) / 2,
                    (inst2.actual_start + inst2.actual_finish) / 2,
                ],
                [inst1.stage_id + 0.75, inst2.stage_id + 0.75],
                **self.line_args,
            )


    def get_critical_path(self) -> list[Instruction]:
        """Return the critical path of the DAG.

        When there are multiple possible critical paths, choose the smoothest,
        i.e. one with minimum number of `stage_id` changes along the path.
        """
        # Length is the amount of total `stage_id` changes along the critical path.
        smallest_length, critical_path = float("inf"), []
        stack: deque[tuple[float, list[Instruction]]] = deque()
        stack.append((0.0, [self.critical_dag_aon.entry_node]))

        while stack:
            length, path = stack.pop()
            node = path[-1]
            if node is self.critical_dag_aon.exit_node and length < smallest_length:
                smallest_length, critical_path = length, path
            for child in node.children:
                # Only go through nodes on the critical path.
                # Cannot use the `==` operator due to floating point errors.
                if abs(child.earliest_start - child.latest_start) < 1e-10:
                    if isinstance(node, _Dummy) or isinstance(child, _Dummy):
                        stage_diff = 0.0
                    else:
                        stage_diff = abs(node.stage_id - child.stage_id)
                    stack.append((length + stage_diff, path + [child]))

        # Slice out entry and exit nodes
        return list(filter(lambda node: not isinstance(node, _Dummy), critical_path))

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