"""A linear time-cost trade-off solver using PD-algorithm"""

from __future__ import annotations

import logging
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



DEFAULT_RECTANGLE_ARGS = {
    Forward: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    Backward: dict(facecolor="#9fc887", edgecolor="#000000", linewidth=1.0),
}

DEFAULT_ANNOTATION_ARGS = {
    Forward: dict(color="#2e00ff", fontsize=20.0, ha="center", va="center"),
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
        interval: int = 100,
        unit_scale: float = 0.01,
        ) -> None:
        """Given a critical graph, iteratively run the PD algorithm and update the instruction durations

        Arguments:
            critical_dag: a critical DAG in the form of ReneDAG
            output_dir: output directory for figures
            interval: number of iterations to output pipeline graph and frequency assignment
            unit_scale: the single unit of duration deduction per PD iteration
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
        self.interval = interval
        self.unit_scale = unit_scale


        # for Pareto frontier
        self.costs: list[float] = []
        self.times: list[float] = []
        # self.run_pd_algorithm()

        # logging.info("aoa", list(self.critical_dag_aoa.edges()))

    def generate_aon_graph(self) -> nx.DiGraph:
        dag: nx.DiGraph = nx.DiGraph()
        # Start to construct critical path graph, in AON format
        # This is different than get_critical_path() in ReneDAG as it allows multiple critcal paths
        q: SimpleQueue[Instruction] = SimpleQueue()
        q.put(self.entry_node)

        while not q.empty():
            node = q.get()
            # logging.info("current ", node.__repr__())
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
            # logging.info(f"add edge {left_id}, {right_id}, weight {cur_inst.duration}")
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
        # logging.info(list(cap_graph.nodes))
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
                cur_inst: Instruction = cap_graph[cur_id][succ_id]["inst"]
                if isinstance(cur_inst, _Dummy):
                    cap_graph[cur_id][succ_id]["lb"]: float = 0.0
                    cap_graph[cur_id][succ_id]["ub"]: float = 10000.0
                elif abs(cur_inst.max_duration - cur_inst.duration) < 1e-5 and abs(cur_inst.min_duration - cur_inst.duration) < 1e-5:
                    cap_graph[cur_id][succ_id]["lb"]: float = 0.0
                    cap_graph[cur_id][succ_id]["ub"]: float = 10000.0
                elif cur_inst.duration - self.unit_scale < cur_inst.min_duration:
                    cap_graph[cur_id][succ_id]["lb"]: float = cur_inst.get_derivative(cur_inst.duration, cur_inst.duration + self.unit_scale)
                    cap_graph[cur_id][succ_id]["ub"]: float = 10000.0
                elif cur_inst.duration + self.unit_scale > cur_inst.max_duration:
                    cap_graph[cur_id][succ_id]["lb"]: float = 0.0
                    cap_graph[cur_id][succ_id]["ub"]: float = cur_inst.get_derivative(cur_inst.duration, cur_inst.duration - self.unit_scale)
                else:
                    cap_graph[cur_id][succ_id]["lb"]: float = cur_inst.get_derivative(cur_inst.duration, cur_inst.duration + self.unit_scale)        
                    cap_graph[cur_id][succ_id]["ub"]: float = cur_inst.get_derivative(cur_inst.duration, cur_inst.duration - self.unit_scale)  
                    # if cur_id == 20 and succ_id == 21:
                    #     print(cap_graph[cur_id][succ_id]["lb"], cap_graph[cur_id][succ_id]["ub"])

        # Change weight to max capacity
        # for u, v in cap_graph.edges:
        #     cap_graph[u][v]["weight"] = cap_graph[u][v]["ub"]
        
        return cap_graph


    def search_path_bfs(self, graph: nx.DiGraph, s: int, t: int) -> tuple[dict[int, bool], dict[int, int]]:
        """Search path from s to t using BFS search, return a tuple of visited and parents"""
        parents: dict[int, int] = dict()
        visited: dict[int, bool] = dict()
        for node_id in graph.nodes:
            parents[node_id] = -1
            visited[node_id] = False
        # logging.info(list(graph.nodes))
        q: SimpleQueue[int] = SimpleQueue()
        q.put(s)
        while not q.empty():
            cur_id = q.get()
            visited[cur_id] = True
            if cur_id == t:
                break
            for child_id in list(graph.successors(cur_id)):
                if visited[child_id] == False and abs(graph[cur_id][child_id]["weight"] - 0) > 1e-5:
                    parents[child_id] = cur_id
                    q.put(child_id)

        return (visited, parents)
    
    def search_path_dfs(self, graph: nx.DiGraph, s: int, t: int) -> tuple[dict[int, bool], dict[int, int]]:
        """Search path from s to t using DFS search, return a tuple of visited and parents"""
        parents: dict[int, int] = dict()
        visited: dict[int, bool] = dict()
        for node_id in graph.nodes:
            parents[node_id] = -1
            visited[node_id] = False
        # logging.info(list(graph.nodes))
        q: deque[int] = deque()
        q.append(s)
        while len(q) > 0:
            cur_id = q.pop()
            visited[cur_id] = True
            if cur_id == t:
                break
            for child_id in list(graph.successors(cur_id)):
                if visited[child_id] == False and abs(graph[cur_id][child_id]["weight"] - 0) > 1e-5:
                    parents[child_id] = cur_id
                    q.append(child_id)

        return (visited, parents)

    def find_max_flow(self, graph: nx.DiGraph, source_id: int, sink_id: int) -> nx.DiGraph:
        """Find max flow using BFS search, each edge in the graph has a capacity [0, weight]
        
        Arguments:
            graph {nx.DiGraph} -- a DAG with annoated capacity on edges
            source_id {int} -- start node id
            sink_id {int} -- sink node id

        Returns:
            residual_graph {nx.DiGraph} -- a residual graph annotated with flow on edges
        """
        residual_graph: nx.DiGraph = nx.DiGraph(graph)

        while True:
            # Step 1: get a path from entry to exit
            visited, parents = self.search_path_dfs(residual_graph, source_id, sink_id)
            if visited[sink_id] == False:
                break
            # Step 2: find min capacity along the path
            right_ptr = sink_id
            left_ptr = parents[sink_id]
            path = [right_ptr, left_ptr]
            min_capacity = residual_graph[left_ptr][right_ptr]["weight"]
            while left_ptr != source_id:
                right_ptr = left_ptr
                left_ptr = parents[left_ptr]
                path.append(left_ptr)
                min_capacity = min(residual_graph[left_ptr][right_ptr]["weight"], min_capacity)
            # logging.info(f"path  {path}")
            
            # Step 3: update residual graph
            right_ptr = sink_id
            left_ptr = parents[sink_id]
            while True:
                
                residual_graph[left_ptr][right_ptr]["weight"] -= min_capacity
                # Create residual edge if needed
                if residual_graph.has_edge(right_ptr, left_ptr) == False:
                    residual_graph.add_edge(right_ptr, left_ptr, weight=min_capacity, inst=residual_graph[left_ptr][right_ptr]["inst"])
                else:
                    residual_graph[right_ptr][left_ptr]["weight"] += min_capacity
                if left_ptr == source_id:
                    break
                right_ptr = left_ptr
                left_ptr = parents[left_ptr]
    

        # logging.info(f"Iteration {self.iteration}: s_set: ",s_set)
        # logging.info(f"Iteration {self.iteration}: t_set ", t_set)
        return residual_graph
    
    def find_min_cut(self, residual_graph: nx.DiGraph, source_id: int, sink_id: int) -> tuple(set[int], set[int]):
        """Find min cut given a residual graph
        
        Arguments:
            residual_graph {nx.DiGraph} -- a residual graph annotated with flow on edges
            source_id {int} -- start node id
            sink_id {int} -- sink node id

        Returns:
            s_set, t_set {tuple(set[int], set[int])} -- two sets of nodes in the min cut
        """
        # Find the cut:
        visited, _ = self.search_path_dfs(residual_graph, source_id, sink_id)
        s_set: set[int] = set()
        t_set: set[int] = set()
        for i in range(len(visited)):
            if visited[i] == True:
                s_set.add(i)
            else:
                t_set.add(i)

        return (s_set, t_set)
    
    def find_max_flow_bounded(self, graph: nx.DiGraph, source_id: int, sink_id: int) -> nx.DiGraph:
        unbound_graph: nx.DiGraph = nx.DiGraph(graph)
        s_prime = self.node_id
        self.node_id += 1
        unbound_graph.add_node(s_prime)
        t_prime = self.node_id
        self.node_id += 1
        unbound_graph.add_node(t_prime)
        # Step 1: For every edge in the original graph, modify the weight to be upper bound - lower bound
        for u, v in unbound_graph.edges:
            unbound_graph[u][v]["weight"] = unbound_graph[u][v]["ub"] - unbound_graph[u][v]["lb"]

        # Step 2: Add edges from s_prime to all nodes in original graph
        # weight = sum of all lower bounds of parents of current node
        for node_id in unbound_graph.nodes:
            weight = 0
            for parent_id in unbound_graph.predecessors(node_id):
                weight += unbound_graph[parent_id][node_id]["lb"]
            unbound_graph.add_edge(s_prime, node_id, weight=weight, inst=None)
        
        # Step 3: Add edges from all nodes in original graph to t_prime
        # weight = sum of all lower bounds of children of current node
        for node_id in unbound_graph.nodes:
            if node_id == s_prime:
                continue
            weight = 0
            for child_id in unbound_graph.successors(node_id):
                weight += unbound_graph[node_id][child_id]["lb"]
            unbound_graph.add_edge(node_id, t_prime, weight=weight, inst=None)

        # Step 4: Add an edge from t to s with infinite weight
        unbound_graph.add_edge(self.exit_id, self.entry_id, weight=float("inf"), inst=None)

        # plt.figure(figsize=(30, 30))
        # pos = nx.circular_layout(unbound_graph)
        # nx.draw(unbound_graph, pos, with_labels=True, font_weight='bold')
        # # node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
        # # nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)


        # # set the attribute of edge as a combination of lb and ub, and round lb and ub to 2 decimal places
        # for edge in unbound_graph.edges:
        #     unbound_graph.edges[edge]["label"] = f"{round(unbound_graph.edges[edge]['weight'], 2)}"

        # edge_labels = nx.get_edge_attributes(unbound_graph, "label")
        # nx.draw_networkx_edge_labels(unbound_graph, pos, edge_labels=edge_labels)
        # # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, f"unbounded_{self.iteration}.png"), format="PNG")
        # plt.clf()
        # plt.close()

        # Step 5: Find min cut on the unbound graph
        # TODO: change weight to capacity, it is confusing right now
        # extend_residual_graph: nx.DiGraph = self.find_max_flow(unbound_graph, s_prime, t_prime)
        flow_value, flow_dict = nx.maximum_flow(unbound_graph, s_prime, t_prime, capacity="weight")
        # source_capacity: int = 0
        # sink_capacity: int = 0
        # for node_id in unbound_graph.successors(s_prime):
        #     source_capacity += unbound_graph[s_prime][node_id]["weight"]
        # for node_id in unbound_graph.predecessors(t_prime):
        #     sink_capacity += unbound_graph[node_id][t_prime]["weight"]

        # Step 6: Check if residual graph is saturated
        for node_id in unbound_graph.successors(s_prime):
            if abs(flow_dict[s_prime][node_id] - unbound_graph[s_prime][node_id]["weight"]) > 1e-5:
                raise Exception(f"Residual graph is not saturated at the new source (edge {s_prime}->{node_id} has weight {flow_dict[s_prime][node_id]}), no flow in the original DAG!")
        for node_id in unbound_graph.predecessors(t_prime):
            if abs(flow_dict[node_id][t_prime] - unbound_graph[node_id][t_prime]["weight"]) > 1e-5:
                raise Exception(f"Residual graph is not saturated at the new sink (edge {node_id}->{t_prime} has weight {flow_dict[node_id][t_prime]}), no flow in the original DAG!")
            
        # plt.figure(figsize=(30, 30))
        # pos = nx.circular_layout(extend_residual_graph)
        # nx.draw(extend_residual_graph, pos, with_labels=True, font_weight='bold')
        # # node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
        # # nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)


        # # set the attribute of edge as a combination of lb and ub, and round lb and ub to 2 decimal places
        # for edge in extend_residual_graph.edges:
        #     extend_residual_graph.edges[edge]["label"] = f"{round(extend_residual_graph.edges[edge]['weight'], 2)}"

        # edge_labels = nx.get_edge_attributes(extend_residual_graph, "label")
        # nx.draw_networkx_edge_labels(extend_residual_graph, pos, edge_labels=edge_labels)
        # # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, f"extended_residual_{self.iteration}.png"), format="PNG")
        # plt.clf()
        # plt.close()
        
        # Step 7: Retreive the flow for the original graph
        for u, v in graph.edges:
            # f'(u,v) = f(u,v) - lb(u,v)
            # if u in extend_residual_graph.successors(v):
            #     graph[u][v]["weight"] = extend_residual_graph[v][u]["weight"] + graph[u][v]["lb"]
            # else:
            #     graph[u][v]["weight"] = graph[u][v]["lb"]
            graph[u][v]["weight"] = flow_dict[u][v] + graph[u][v]["lb"]

        # Step 8: Modify capacity of the residual graph
        residual_graph: nx.DiGraph = nx.DiGraph(graph)
        edge_pending: list[tuple] = []
        for u, v in residual_graph.edges:
            residual_graph[u][v]["capacity"] = residual_graph[u][v]["ub"] - residual_graph[u][v]["weight"]
            if u in residual_graph.successors(v):
                residual_graph[v][u]["capacity"] = residual_graph[u][v]["weight"] - residual_graph[u][v]["lb"]
            else:
                # residual_graph.add_edge(v, u, weight=residual_graph[u][v]["weight"] - residual_graph[u][v]["lb"], inst=residual_graph[u][v]["inst"])
                edge_pending.append((v, u, residual_graph[u][v]["weight"] - residual_graph[u][v]["lb"], residual_graph[u][v]["inst"]))
                
        for u, v, capacity, inst in edge_pending:
            residual_graph.add_edge(u, v, capacity=capacity, inst=inst)

        # Step 9: Find min cut on the original graph
        # residual_graph = self.find_max_flow(residual_graph, self.entry_id, self.exit_id)
        try:
            # cut_value, partition = nx.minimum_cut(residual_graph, self.entry_id, self.exit_id, capacity="capacity")
            flow_value, flow_dict = nx.maximum_flow(residual_graph, self.entry_id, self.exit_id, capacity="capacity")
            # Add additional flow we get to the original graph
            for u, v in graph.edges:
                graph[u][v]["weight"] += flow_dict[u][v]
                graph[u][v]["weight"] -= flow_dict[v][u]
            
            # Do a minimum cut on the residual graph of the original graph
            new_residual = nx.DiGraph(graph)
            add_candidates = []
            for u, v in new_residual.edges:
                new_residual[u][v]["weight"] = graph[u][v]["ub"] - graph[u][v]["weight"]
                add_candidates.append((v, u, graph[u][v]["weight"] - graph[u][v]["lb"]))
            for u, v, weight in add_candidates:
                new_residual.add_edge(u, v, weight=weight)

            # if self.iteration >= 45:
            #     plt.figure(figsize=(30, 30))
            #     pos = nx.circular_layout(graph)
            #     nx.draw(graph, pos, with_labels=True, font_weight='bold')
            #     # node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
            #     # nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)


            #     # set the attribute of edge as a combination of lb and ub, and round lb and ub to 2 decimal places
            #     for edge in graph.edges:
            #         graph.edges[edge]["label"] = f"lb:{round(graph.edges[edge]['lb'], 2)} ub:{round(graph.edges[edge]['ub'], 2)} {round(graph.edges[edge]['weight'], 2)}"

            #     edge_labels = nx.get_edge_attributes(graph, "label")
            #     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
            #     # plt.tight_layout()
            #     plt.savefig(os.path.join(self.output_dir, f"capacity_final_flow_{self.iteration}.png"), format="PNG")
            #     plt.clf()
            #     plt.close()

        except nx.NetworkXUnbounded:
            return None
        # print(cut_value, partition)
        # plt.figure(figsize=(30, 30))
        # pos = nx.circular_layout(residual_graph)
        # nx.draw(residual_graph, pos, with_labels=True, font_weight='bold')
        # # node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
        # # nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)


        # # set the attribute of edge as a combination of lb and ub, and round lb and ub to 2 decimal places
        # for edge in residual_graph.edges:
        #     residual_graph.edges[edge]["label"] = f"{round(residual_graph.edges[edge]['weight'], 2)}"

        # edge_labels = nx.get_edge_attributes(residual_graph, "label")
        # nx.draw_networkx_edge_labels(residual_graph, pos, edge_labels=edge_labels)
        # # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, f"residual_{self.iteration}.png"), format="PNG")
        # plt.clf()
        # plt.close()
        
        # Step 10: Annotate capacity graph again with new max flow
        # for u, v in graph.edges:
        #     graph[u][v]["weight"] = flow_dict[u][v]


        return new_residual
        
    
    def run_pd_algorithm(self) -> None:
        # TODO: need to start the following iterations using the assigned flows
        while True:
            # Do eager assignment and draw the current pipeline
            for inst in self.critical_dag_aon.insts:
                inst.actual_start = inst.earliest_start
                inst.actual_finish = inst.earliest_finish

            # self.critical_dag_aon.draw_aon_graph(os.path.join(self.output_dir, f"aon_graph_{self.iteration}.png"))
            # logging.info("aon", list(self.critical_graph_aon.edges()))
            self.critical_dag_aoa: nx.DiGraph = self.aon_to_aoa()
            
            self.capacity_graph: nx.DiGraph = self.generate_capacity_graph()

            # if self.iteration >= 135:
            # # if self.iteration % self.interval == 0:
            #     self.draw_aoa_graph(os.path.join(self.output_dir, f"aoa_graph_{self.iteration}.png"))
            #     self.draw_capacity_graph(os.path.join(self.output_dir, f"capacity_graph_{self.iteration}_before.png"))

            # run double side bounded max flow algorithm on capacity graph, after this the capacity graph will be annoated with the flow
            residual_graph = self.find_max_flow_bounded(self.capacity_graph, self.entry_id, self.exit_id)
            if residual_graph is None:
                break
            # s_set, t_set = self.find_min_cut(self.capacity_graph, self.entry_id, self.exit_id, flow_dict)
            s_set, t_set = self.find_min_cut(residual_graph, self.entry_id, self.exit_id)
            cost_change = self.reduce_duration(s_set, t_set)
            if cost_change == float('inf'):
                break

            # if self.iteration >= 135:
            if self.iteration % self.interval == 0:
                # self.draw_aoa_graph(os.path.join(self.output_dir, f"aoa_graph_{self.iteration}.png"))
                self.draw_capacity_graph(os.path.join(self.output_dir, f"capacity_graph_{self.iteration}_after.png"))
                self.draw_pipeline_graph(os.path.join(self.output_dir, f"pipeline_{self.iteration}.png"), draw_time_axis=True)
            self.assign_frequency()

            total_cost = self.calculate_total_cost()
            total_time = self.calculate_total_time()
        
            logging.info(f"Iteration {self.iteration}: cost change {cost_change}")
            logging.info(f"Iteration {self.iteration}: total cost {total_cost}")
            logging.info(f"Iteration {self.iteration}: total time {total_time - self.unit_scale}")

            self.critical_dag_aon.clear_annotations()
            self.node_id = self.critical_dag_aon.node_id
            self.critical_dag_aon.update_critical_dag()

            self.costs.append(total_cost)
            self.times.append(total_time - self.unit_scale)
            self.iteration += 1
        # need to output final frequency assignment
        self.assign_frequency()
        self.draw_pipeline_graph(os.path.join(self.output_dir, f"pipeline_final.png"), draw_time_axis=True)
        self.draw_pareto_frontier(os.path.join(self.output_dir, f"Pareto_frontier.png"))
    
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
        
        logging.info(f"Iteration {self.iteration}: reduce edges {reduce_edges}")
        logging.info(f"Iteration {self.iteration}: increase edges {increase_edges}")
        # if len(reduce_edges) > 1 or len(increase_edges) > 0:
        #     raise ValueError(f"reduce edges {reduce_edges} and increase edges {increase_edges}")
        cost_change = 0

        for u, v in reduce_edges:
            inst: Instruction = self.capacity_graph[u][v]["inst"]
            if inst.duration - self.unit_scale < inst.min_duration or type(inst) == _Dummy:
                return float('inf')
            else:
                cost_change += inst.get_derivative(inst.duration, inst.duration - self.unit_scale) * self.unit_scale
                inst.duration -= self.unit_scale


        for u, v in increase_edges:
            inst: Instruction = self.capacity_graph[u][v]["inst"]
            # Notice: dummy edge is always valid for increasing duration
            if type(inst) == _Dummy:
                inst.duration += self.unit_scale
            elif inst.duration + self.unit_scale > inst.max_duration:
                return float('inf')
            else:
                cost_change -= inst.get_derivative(inst.duration, inst.duration + self.unit_scale) * self.unit_scale
                inst.duration += self.unit_scale

        
        return cost_change

    def calculate_total_cost(self) -> float:
        q: SimpleQueue[int] = SimpleQueue()
        q.put(0)
        visited: list[str] = list()
        total_cost: float = 0.0

        while not q.empty():
            cur_id: int = q.get()
            if cur_id in visited:
                continue
            visited.append(cur_id)
            cur_node: Instruction = self.critical_dag_aon.complete_dag.nodes[cur_id]["inst"]
            if not isinstance(cur_node, _Dummy) and cur_node.__repr__() not in visited:
                total_cost += cur_node.get_p2p_refined_cost(cur_node.duration)
                visited.append(cur_node.__repr__())
            for child_id in self.critical_dag_aon.complete_dag.successors(cur_id):
                q.put(child_id)

        return total_cost
    
    def calculate_total_time(self) -> float:
        critical_path = self.critical_dag_aon.get_critical_path()
        total_time: float = 0.0
        for inst in critical_path:
            total_time += inst.actual_duration
        return total_time
    
    def assign_frequency(self) -> list[list[int]]:
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
                # start binary search
                left = 0
                right = len(cur_node.time_costs) - 1
                while left < right:
                    mid = (left + right) // 2
                    # if there is an exact match, or we are at the head/end of the list, we are done
                    if abs(cur_node.time_costs[mid][0] - cur_node.duration) < 1e-5 or mid == 0 or mid == len(cur_node.time_costs) - 1:
                        cur_node.frequency = cur_node.time_costs[mid][2]
                        break
                    elif cur_node.time_costs[mid][0] < cur_node.duration:
                        if cur_node.time_costs[mid-1][0] > cur_node.duration:
                            # we are between two points, choose the one with shorter duration since we are solving deadline problem
                            cur_node.frequency = cur_node.time_costs[mid][2]
                            # mid_duration = (cur_node.time_costs[mid][0] + cur_node.time_costs[mid-1][0]) / 2
                            # if mid_duration < cur_node.duration:
                            #     cur_node.frequency = cur_node.time_costs[mid-1][2]
                            # else:
                            #     cur_node.frequency = cur_node.time_costs[mid][2]
                            break
                        right = mid
                    elif cur_node.time_costs[mid][0] > cur_node.duration:
                        if cur_node.time_costs[mid+1][0] < cur_node.duration:
                            # we are between two points, choose the one with shorter duration since we are solving deadline problem
                            cur_node.frequency = cur_node.time_costs[mid+1][2]
                            # mid_duration = (cur_node.time_costs[mid][0] + cur_node.time_costs[mid+1][0]) / 2
                            # if mid_duration < cur_node.duration:
                            #     cur_node.frequency = cur_node.time_costs[mid][2]
                            # else:
                            #     cur_node.frequency = cur_node.time_costs[mid+1][2]
                            # break
                        left = mid + 1
                                   
            if cur_node.stage_id not in stage_view:
                stage_view[cur_node.stage_id] = [cur_node]
            else:
                stage_view[cur_node.stage_id].append(cur_node)

        # # if in the next iteration the duration will be smaller than min duration, assign the highest frequency possible
        # changed = False
        # for stage_id, insts in stage_view.items():
        #     insts: list[Instruction] = stage_view[stage_id]
        #     for inst in insts:
        #         if inst.duration - self.unit_scale < inst.min_duration:
        #             logging.info(f"{inst.__repr__()} will exceed min duration in the next iteration, assign highest frequency {inst.time_costs[-1][2]} instead of {inst.frequency}")
        #             inst.frequency = inst.time_costs[-1][2]
        #             inst.duration = inst.time_costs[-1][0]
        #             changed = True
        
        # # We need to update the total duration and cost if any stage changes
        # if changed:
        #     self.critical_dag_aon.clear_annotations()
        #     self.critical_dag_aon.annotate_nodes()
        #     for inst in self.critical_dag_aon.insts:
        #         inst.actual_start = inst.earliest_start
        #         inst.actual_finish = inst.earliest_finish    

        total_freqs: list[list[int]] = []
        logging.info(f"Iteration {self.iteration} outputing frequency assignment...")
        for stage_id in sorted(stage_view.keys()):
            logging.info(f"Stage {stage_id} frequency assignment ")
            insts: list[Instruction] = stage_view[stage_id]
            freqs: list[int] = []
            reprs: list[int] = []
            for inst in insts:
                assert(inst.frequency != -1)
                freqs.append(inst.frequency)
                reprs.append(inst.__repr__())
            logging.info(f"Freqs: {freqs}")
            logging.info(f"Reprs: {reprs}")
            total_freqs.append(freqs)

        with open(os.path.join(self.output_dir, f"freqs_{self.iteration:04d}.py"), "w+") as f:
            f.write(repr(total_freqs)+'\n')
            

        return total_freqs

    def draw_aoa_graph(self, path: str) -> None:
        plt.figure(figsize=(20, 20))
        pos = nx.circular_layout(self.critical_dag_aoa)
        nx.draw(self.critical_dag_aoa, pos, with_labels=True, font_weight='bold')
        node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
        nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)
        # edge_labels = nx.get_edge_attributes(self.critical_dag_aoa, "weight")
        # nx.draw_networkx_edge_labels(self.critical_dag_aoa, pos, edge_labels=edge_labels)
        plt.tight_layout()
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_capacity_graph(self, path: str) -> None:
        # get all edges that are critical
        # filtered_critical_pairs = self.get_critical_pairs()
        # # remove all other edges not in the critical pairs
        # filtered_edges = []
        # for inst1, inst2 in filtered_critical_pairs:
        #     filtered_edges.append((self.critical_dag_aon.inst_id_map[repr(inst1)], self.critical_dag_aon.inst_id_map[repr(inst2)]))
        # remove_edges = []
        # for edge in self.capacity_graph.edges:
        #     if edge not in filtered_edges:
        #         remove_edges.append(edge)
        # self.capacity_graph.remove_edges_from(remove_edges)

        plt.figure(figsize=(30, 30))
        pos = nx.circular_layout(self.capacity_graph)
        nx.draw(self.capacity_graph, pos, with_labels=True, font_weight='bold')
        # node_labels = nx.get_node_attributes(self.critical_dag_aoa, "repr")
        # nx.draw_networkx_labels(self.critical_dag_aoa, pos, labels=node_labels)


        # set the attribute of edge as a combination of lb and ub, and round lb and ub to 2 decimal places
        for edge in self.capacity_graph.edges:
            self.capacity_graph.edges[edge]["label"] = f"{repr(self.capacity_graph.edges[edge]['inst'])}:({round(self.capacity_graph.edges[edge]['lb'], 2)}, {round(self.capacity_graph.edges[edge]['ub'], 2)}), {round(self.capacity_graph.edges[edge]['weight'], 2)}"

        edge_labels = nx.get_edge_attributes(self.capacity_graph, "label")
        nx.draw_networkx_edge_labels(self.capacity_graph, pos, edge_labels=edge_labels)
        # plt.tight_layout()
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_pipeline_graph(self, path: str, draw_time_axis: bool = False) -> None:
        """Draw the pipeline on the given Axes object."""
        fig, ax = plt.subplots(figsize=(12, 4), tight_layout=True)
        ax.set_xlim(0, 58)
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

    def draw_pareto_frontier(self, path: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
        ax.plot(self.times, self.costs)
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        fig.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_critical_path(self, ax: Axes) -> None:
        """Draw the critical path of the DAG on the given Axes object."""
        # critical_path = self.get_critical_path()

        # get all pairs of instructions in the critical path defined by self.critical_dag_aon by BFS
        filtered_critical_pairs = self.get_critical_pairs()

        for inst1, inst2 in filtered_critical_pairs:
            ax.plot(
                [
                    (inst1.actual_start + inst1.actual_finish) / 2,
                    (inst2.actual_start + inst2.actual_finish) / 2,
                ],
                [inst1.stage_id + 0.75, inst2.stage_id + 0.75],
                **self.line_args,
            )

    def get_critical_pairs(self) -> list[tuple[Instruction, Instruction]]:
        # get all pairs of instructions in the critical path defined by self.critical_dag_aon by BFS
        critical_pairs = []
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.entry_id)
        visited: set[int] = set()
        while not q.empty():
            cur_id = q.get()
            if cur_id in visited:
                continue
            visited.add(cur_id)

            for succ_id in self.critical_dag_aon.dag.successors(cur_id):
                q.put(succ_id)
                if cur_id != self.entry_id and succ_id != self.exit_id:
                    critical_pairs.append((self.critical_dag_aon.dag.nodes[cur_id]["inst"], self.critical_dag_aon.dag.nodes[succ_id]["inst"]))

        # do some ad hoc filtering to remove some errornous critical pairs
        filtered_critical_pairs = []
        for inst1, inst2 in critical_pairs:
            if inst1.stage_id == inst2.stage_id and abs(inst1.actual_finish - inst2.actual_start) > 1e-5:
                continue
            filtered_critical_pairs.append((inst1, inst2))

        return filtered_critical_pairs

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
        logging.info("current: ", cur_id)
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