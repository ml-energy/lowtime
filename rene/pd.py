"""A time-cost trade-off solver based on the PD-algorithm."""


from __future__ import annotations

import copy
import logging
from pathlib import Path
from queue import SimpleQueue
from collections import deque
from collections.abc import Generator

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from networkx.algorithms.flow import edmonds_karp  # type: ignore

from rene.constants import (
    FP_ERROR,
    DEFAULT_RECTANGLE_ARGS,
    DEFAULT_ANNOTATION_ARGS,
    DEFAULT_LINE_ARGS,
)
from rene.dag import ReneDAGOld
from rene.perseus.instruction import Instruction
from rene.operation import DummyOperation
from rene.graph_utils import aon_dag_to_aoa_dag


class PhillipsDessouky:
    """Implements the Phillips-Dessouky algorithm for the time-cost tradeoff problem."""

    def __init__(self, dag: nx.DiGraph) -> None:
        """Initialize the solver.

        Args:
            dag: A DAG with the source and sink node IDs annotated respectively as
                `dag.graph["source_node"]` and `dag.graph["sink_node"]`.
        """
        # Run checks on the DAG and cache some properties.
        # Check: It's a DAG.
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("The graph should be a Directed Acyclic Graph.")

        # Check: Only one source node that matches annotation.
        if (source_node := dag.graph.get("source_node")) is None:
            raise ValueError("The graph should have a `source_node` attribute.")
        source_node_candidates = []
        for node_id, in_degree in dag.in_degree():
            if in_degree == 0:
                source_node_candidates.append(node_id)
        if len(source_node_candidates) == 0:
            raise ValueError(
                "Found zero nodes with in-degree 0. Cannot determine source node."
            )
        if len(source_node_candidates) > 1:
            raise ValueError(
                f"Expecting only one source node, found {source_node_candidates}."
            )
        if (detected_source_node := source_node_candidates[0]) != source_node:
            raise ValueError(
                f"Detected source node ({detected_source_node}) does not match "
                f"the annotated source node ({source_node})."
            )

        # Check: Only one sink node that matches annotation.
        if (sink_node := dag.graph.get("sink_node")) is None:
            raise ValueError("The graph should have a `sink_node` attribute.")
        sink_node_candidates = []
        for node_id, out_degree in dag.out_degree():
            if out_degree == 0:
                sink_node_candidates.append(node_id)
        if len(sink_node_candidates) == 0:
            raise ValueError(
                "Found zero nodes with out-degree 0. Cannot determine sink node."
            )
        if len(sink_node_candidates) > 1:
            raise ValueError(
                f"Expecting only one sink node, found {sink_node_candidates}."
            )
        if (detected_sink_node := sink_node_candidates[0]) != sink_node:
            raise ValueError(
                f"Detected sink node ({detected_sink_node}) does not match "
                f"the annotated sink node ({sink_node})."
            )

        self.aon_dag = dag

    def run(self) -> Generator[nx.DiGraph, None, None]:
        """Run the algorithm and yield a DAG after each iteration.

        The solver will not deepcopy operations on the DAG but rather in-place modify them
        for performance reasons. The caller should deepcopy the DAG if needed before running
        the next iteration.
        """
        # Convert the original activity-on-node DAG to activity-on-arc DAG form.
        # AOA DAGs are purely internal. All public input and output of this class
        # should be in AON form.
        aoa_dag = aon_dag_to_aoa_dag(self.aon_dag, attr_name="op")

        # Iteratively reduce the execution time of the DAG.
        while True:
            # TODO(JW): More than the DAG should probably be returned.
            dag = self.run_one_iteration(aoa_dag)
            if dag is not None:
                # We directly modify operation attributes in the DAG, so after we
                # ran one iteration, the AON DAG holds updated attributes.
                yield self.aon_dag
            else:
                break

    def run_one_iteration(self, aoa_dag: nx.DiGraph) -> nx.DiGraph | None:
        """Reduce the execution time of the DAG by 1 while increasing cost minimally.
        
        Returns:
            The new (reduced duration) DAG or None if no further reduction is possible.
        """
        critical_dag = self.to_critical_dag(aoa_dag)


class PDSolver:
    """The PD solver class for time-cost trade-off problem.

    Takes a critical DAG as input and iteratively run the PD algorithm to update the instruction durations.
    """

    def __init__(
        self,
        rene_dag: ReneDAGOld,
        output_dir: Path,
    ) -> None:
        """Initialize the PD solver.

        Arguments:
            rene_dag: a ReneDAG object
            output_dir: output directory for figures and frequency assignments
        """
        self.iteration: int = 0
        self.output_dir = output_dir
        self.rene_dag: ReneDAGOld = rene_dag
        # self.critical_dag_aon: CriticalDAG = critical_dag
        self.node_id: int = self.rene_dag.node_id
        self.annotation_args = DEFAULT_ANNOTATION_ARGS
        self.rectangle_args = DEFAULT_RECTANGLE_ARGS
        self.line_args = DEFAULT_LINE_ARGS
        self.entry_id: int = 0
        self.exit_id: int = 1

        # for Pareto frontier
        self.costs: list[float] = []
        self.refined_costs: list[float] = []
        self.times: list[float] = []

    def aon_to_aoa(self, rene_dag: ReneDAGOld) -> nx.DiGraph:
        """Convert the ReneDAG.dag in Activity-on-Node (AON) form to a dag in Activity-on-Arc (AOA) form.

        Arguments:
            rene_dag: A ReneDAG object

        Returns:
            A nx.DiGraph DAG representing ReneDAG in AOA form
        """
        # TODO: crash dummy nodes for optimization
        # do a BFS to split all nodes and reconnect
        dag: nx.DiGraph = nx.DiGraph(rene_dag.dag)
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
            # Remove current node
            dag.remove_node(cur_id)
            # Split node
            left_id = self.node_id
            right_id = left_id + 1
            dag.add_node(left_id, inst=cur_inst, repr=repr(cur_inst))
            dag.add_node(right_id, inst=cur_inst, repr=repr(cur_inst))
            # Create activity-on-edge
            dag.add_edge(left_id, right_id, weight=cur_inst.duration, inst=cur_inst)
            if cur_inst is rene_dag.entry_node:
                self.entry_id = left_id
            elif cur_inst is rene_dag.exit_node:
                self.exit_id = right_id

            self.node_id += 2
            # Reconnect with predecessors and successors
            for pred_id in pred_ids:
                dag.add_edge(
                    pred_id,
                    left_id,
                    weight=0.0,
                    inst=_Dummy(
                        -1, -1, duration=0, min_duration=0, max_duration=float("inf")
                    ),
                )
            for succ_id in succ_ids:
                dag.add_edge(
                    right_id,
                    succ_id,
                    weight=0.0,
                    inst=_Dummy(
                        -1, -1, duration=0, min_duration=0, max_duration=float("inf")
                    ),
                )
            targets.remove(cur_id)

        return dag

    def generate_capacity_graph(
        self, aoa_dag: nx.DiGraph
    ) -> tuple[nx.DiGraph, int, int]:
        """Generate the capacity graph from the critical AOA graph.

        Arguments:
            aoa_dag: A critical AOA graph

        Returns:
            A tuple of (capacity graph, entry node id, exit node id)
        """
        cap_graph: nx.DiGraph = nx.DiGraph(aoa_dag)
        # Relabel all nodes
        node_ids: list[int] = list(cap_graph.nodes)
        mapping: dict = {}
        index = 0
        for node_id in node_ids:
            mapping[node_id] = index
            index += 1
        cap_graph = nx.relabel_nodes(cap_graph, mapping)

        # capa_dict = {}
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
                lb = 0.0
                ub = 0.0
                if isinstance(cur_inst, _Dummy) or (
                    abs(cur_inst.max_duration - cur_inst.duration) < FP_ERROR
                    and abs(cur_inst.min_duration - cur_inst.duration) < FP_ERROR
                ):
                    lb = 0.0
                    ub = 10000.0
                elif cur_inst.duration - 1 < cur_inst.min_duration:
                    lb = cur_inst.get_derivative(
                        cur_inst.duration, cur_inst.duration + 1
                    )
                    ub = 10000.0
                elif cur_inst.duration + 1 > cur_inst.max_duration:
                    lb = 0.0
                    ub = cur_inst.get_derivative(
                        cur_inst.duration, cur_inst.duration - 1
                    )
                else:
                    lb = cur_inst.get_derivative(
                        cur_inst.duration, cur_inst.duration + 1
                    )
                    ub = (
                        cur_inst.get_derivative(
                            cur_inst.duration, cur_inst.duration - 1
                        )
                        + FP_ERROR
                    )
                cap_graph[cur_id][succ_id]["lb"] = lb // FP_ERROR * FP_ERROR
                cap_graph[cur_id][succ_id]["ub"] = ub // FP_ERROR * FP_ERROR

        return cap_graph, mapping[self.entry_id], mapping[self.exit_id]

    def search_path_bfs(
        self, graph: nx.DiGraph, s: int, t: int
    ) -> tuple[dict[int, bool], dict[int, int]]:
        """Search path from s to t using BFS search, return a tuple of visited and parents.

        Args:
            graph:  The graph to search on.
            s: The source node id.
            t: The target node id.

        Returns:
            A tuple of visited indices and parents.
        """
        parents: dict[int, int] = {}
        visited: dict[int, bool] = {}
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
                if (
                    not visited[child_id]
                    and abs(graph[cur_id][child_id]["weight"]) > FP_ERROR
                ):
                    parents[child_id] = cur_id
                    q.put(child_id)

        return (visited, parents)

    def search_path_dfs(
        self, graph: nx.DiGraph, s: int, t: int
    ) -> tuple[dict[int, bool], dict[int, int]]:
        """Search path from s to t using DFS search, return a tuple of visited and parents.

        Args:
            graph: The graph to search on.
            s: The source node id.
            t: The target node id.

        Returns:
            A tuple of visited indices and parents.
        """
        parents: dict[int, int] = {}
        visited: dict[int, bool] = {}
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
                if (
                    not visited[child_id]
                    and abs(graph[cur_id][child_id]["weight"]) > FP_ERROR
                ):
                    parents[child_id] = cur_id
                    q.append(child_id)

        return (visited, parents)

    def find_min_cut(
        self, residual_graph: nx.DiGraph, source_id: int, sink_id: int
    ) -> tuple[set[int], set[int]]:
        """Find min cut given a residual graph.

        Arguments:
            residual_graph: A residual graph annotated with flow on edges
            source_id: Start node id
            sink_id: Sink node id

        Returns:
            s_set, t_set: Two sets of nodes in the min cut
        """
        # Find the cut:
        visited, _ = self.search_path_dfs(residual_graph, source_id, sink_id)
        s_set: set[int] = set()
        t_set: set[int] = set()
        for i in range(len(visited)):
            if visited[i] is True:
                s_set.add(i)
            else:
                t_set.add(i)

        return (s_set, t_set)

    def find_max_flow_bounded(  # noqa: PLR0912
        self, graph: nx.DiGraph, source_id: int, sink_id: int
    ) -> nx.DiGraph | None:
        """Find max flow given a double-sides capacity bounded graph.

        Each edge in the graph has a double-side bounded capacity [lb, ub].

        Arguments:
            graph: a DAG with annoated capacity on edges
            source_id: start node id
            sink_id: sink node id

        Returns:
            residual_graph | None: a residual graph annotated with flow on edges,
            return None if no feasible flow exists
        """
        unbound_graph: nx.DiGraph = nx.DiGraph(graph)
        s_prime = self.node_id
        unbound_graph.add_node(s_prime)
        t_prime = self.node_id + 1
        unbound_graph.add_node(t_prime)
        # Step 1: For every edge in the original graph, modify the capacity to be upper bound - lower bound
        for u, v in unbound_graph.edges:
            unbound_graph[u][v]["capacity"] = (
                unbound_graph[u][v]["ub"] - unbound_graph[u][v]["lb"]
            )

        # Step 2: Add edges from s_prime to all nodes in original graph
        # capacity = sum of all lower bounds of parents of current node
        for node_id in unbound_graph.nodes:
            capacity = 0
            for parent_id in unbound_graph.predecessors(node_id):
                capacity += unbound_graph[parent_id][node_id]["lb"]
            unbound_graph.add_edge(s_prime, node_id, capacity=capacity, inst=None)

        # Step 3: Add edges from all nodes in original graph to t_prime
        # capacity = sum of all lower bounds of children of current node
        for node_id in unbound_graph.nodes:
            if node_id == s_prime:
                continue
            capacity = 0
            for child_id in unbound_graph.successors(node_id):
                capacity += unbound_graph[node_id][child_id]["lb"]
            unbound_graph.add_edge(node_id, t_prime, capacity=capacity, inst=None)

        # Step 4: Add an edge from t to s with infinite weight
        unbound_graph.add_edge(sink_id, source_id, capacity=float("inf"), inst=None)

        # Step 5: Find min cut on the unbound graph
        flow_value, flow_dict = nx.maximum_flow(
            unbound_graph, s_prime, t_prime, capacity="capacity", flow_func=edmonds_karp
        )

        # Step 6: Check if residual graph is saturated
        for node_id in unbound_graph.successors(s_prime):
            if (
                abs(
                    flow_dict[s_prime][node_id]
                    - unbound_graph[s_prime][node_id]["capacity"]
                )
                > FP_ERROR
            ):
                logging.info(
                    "Edge %s->%s has weight %s but capcaity is %s",
                    s_prime,
                    node_id,
                    flow_dict[s_prime][node_id],
                    unbound_graph[s_prime][node_id]["capacity"],
                )
                raise RuntimeError(
                    "Residual graph is not saturated at the new source, no flow in the original DAG!"
                )
        for node_id in unbound_graph.predecessors(t_prime):
            if (
                abs(
                    flow_dict[node_id][t_prime]
                    - unbound_graph[node_id][t_prime]["capacity"]
                )
                > FP_ERROR
            ):
                logging.info(
                    "Edge %s->%s has weight %s",
                    node_id,
                    t_prime,
                    flow_dict[node_id][t_prime],
                )
                raise RuntimeError(
                    "Residual graph is not saturated at the new sink, no flow in the original DAG!"
                )

        # Step 7: Retreive the flow for the original graph
        for u, v in graph.edges:
            # f'(u,v) = f(u,v) - lb(u,v)
            graph[u][v]["weight"] = flow_dict[u][v] + graph[u][v]["lb"]

        # Step 8: Modify capacity of the residual graph
        residual_graph: nx.DiGraph = nx.DiGraph(graph)
        edge_pending: list[tuple] = []
        for u, v in residual_graph.edges:
            residual_graph[u][v]["capacity"] = (
                residual_graph[u][v]["ub"] - residual_graph[u][v]["weight"]
            )
            if u in residual_graph.successors(v):
                residual_graph[v][u]["capacity"] = (
                    residual_graph[u][v]["weight"] - residual_graph[u][v]["lb"]
                )
            else:
                edge_pending.append(
                    (
                        v,
                        u,
                        residual_graph[u][v]["weight"] - residual_graph[u][v]["lb"],
                        residual_graph[u][v]["inst"],
                    )
                )

        for u, v, capacity, inst in edge_pending:
            residual_graph.add_edge(u, v, capacity=capacity, inst=inst)

        # Step 9: Refine the feasible flow found to the maximum flow
        try:
            # cut_value, partition = nx.minimum_cut(residual_graph, self.entry_id, self.exit_id, capacity="capacity")
            flow_value, flow_dict = nx.maximum_flow(
                residual_graph,
                source_id,
                sink_id,
                capacity="capacity",
                flow_func=edmonds_karp,
            )

            # Add additional flow we get to the original graph
            # XXX(JW): I think we can directly use edmonds_karp to skip following lines
            #          and directly get the residual graph.
            for u, v in graph.edges:
                graph[u][v]["weight"] += flow_dict[u][v]
                graph[u][v]["weight"] -= flow_dict[v][u]

            # Construct the new residual graph.
            new_residual = nx.DiGraph(graph)
            add_candidates = []
            for u, v in new_residual.edges:
                new_residual[u][v]["weight"] = graph[u][v]["ub"] - graph[u][v]["weight"]
                add_candidates.append((v, u, graph[u][v]["weight"] - graph[u][v]["lb"]))
            for u, v, weight in add_candidates:
                new_residual.add_edge(u, v, weight=weight)

        except nx.NetworkXUnbounded:
            return None

        return new_residual

    def get_critical_aoa_dag(
        self, aoa_dag: nx.DiGraph, exit_node: Instruction
    ) -> nx.DiGraph:
        """Run Critical Path Method on the AOA dag to get the critical path.

        Arguments:
            aoa_dag: an AOA DAG
            exit_node: the exit node of the AOA DAG

        Returns:
            A critical AOA DAG, which is a subgraph of the original AOA DAG,
            with only critical edges
        """
        # Step 1: Clear all annotations
        for edge in aoa_dag.edges:
            cur_inst: Instruction = aoa_dag.edges[edge]["inst"]
            cur_inst.earliest_start = 0.0
            cur_inst.latest_start = float("inf")
            cur_inst.earliest_finish = 0.0
            cur_inst.latest_finish = float("inf")

        # Step 2: Run forward pass to get earliest start and earliest finish, note that instructions are on the edges
        for node_id in nx.topological_sort(aoa_dag):
            for succ in aoa_dag.successors(node_id):
                cur_inst = aoa_dag[node_id][succ]["inst"]

                for next_succ in aoa_dag.successors(succ):
                    next_inst: Instruction = aoa_dag[succ][next_succ]["inst"]

                    next_inst.earliest_start = max(
                        next_inst.earliest_start,
                        cur_inst.earliest_finish,
                    )
                    next_inst.earliest_finish = (
                        next_inst.earliest_start + next_inst.duration
                    )

        # Step 3: Run backward pass to get latest start and latest finish, note that instructions are on the edges
        exit_node.latest_finish = exit_node.earliest_finish
        exit_node.latest_start = exit_node.latest_finish - exit_node.duration
        for node_id in list(reversed(list(nx.topological_sort(aoa_dag)))):
            for pred in aoa_dag.predecessors(node_id):
                cur_inst = aoa_dag[pred][node_id]["inst"]

                for next_pred in aoa_dag.predecessors(pred):
                    parent_inst: Instruction = aoa_dag[next_pred][pred]["inst"]

                    parent_inst.latest_start = min(
                        parent_inst.latest_start,
                        cur_inst.latest_start - parent_inst.duration,
                    )
                    parent_inst.latest_finish = (
                        parent_inst.latest_start + parent_inst.duration
                    )

        # Step 4: Remove all edges that are not on the critical path, note that instructions are on the edges
        critical_dag: nx.DiGraph = nx.DiGraph(aoa_dag)
        for node_id in nx.topological_sort(aoa_dag):
            for succ in aoa_dag.successors(node_id):
                cur_inst = aoa_dag[node_id][succ]["inst"]
                if abs(cur_inst.slack) > FP_ERROR:
                    critical_dag.remove_edge(node_id, succ)

        for node_id in nx.topological_sort(critical_dag):
            for succ in critical_dag.successors(node_id):
                cur_inst = critical_dag[node_id][succ]["inst"]

        return critical_dag

    def run_one_iteration(self) -> ReneDAGOld | None:
        """Run one iteration of the PD algorithm.

        Returns:
            ReneDAG | None: The updated ReneDAG after one iteration of the PD algorithm,
                or None if the algorithm completes
        """
        # Step 1: Get critical AOA dag from Rene AOA dag
        self.critical_aoa_dag = self.get_critical_aoa_dag(
            self.rene_dag_aoa, self.rene_dag.exit_node
        )

        # Step 2: Generate capacity graph
        self.capacity_graph, cap_entry_id, cap_exit_id = self.generate_capacity_graph(
            self.critical_aoa_dag
        )

        # Step 3: Run double side bounded max flow algorithm on capacity graph,
        # After this the capacity graph will be annoated with the flow
        residual_graph = self.find_max_flow_bounded(
            self.capacity_graph, cap_entry_id, cap_exit_id
        )
        if residual_graph is None:
            return None
        #  Step 4: Find the min cut on the residual graph
        s_set, t_set = self.find_min_cut(residual_graph, cap_entry_id, cap_exit_id)
        # Step 5: Reduce the duration by cutting crossing edges between s_set and t_set
        cost_change = self.reduce_duration(s_set, t_set)
        if cost_change == float("inf") or abs(cost_change) < FP_ERROR:
            return None
        # the ReneDAG must be re-scheduled after the duration is reduced
        self.rene_dag.scheduled = False

        # Step 6: Assign frequency to each instruction, solve for the total cost and the total time
        total_cost, refined_cost = self.rene_dag.get_total_cost()
        total_time = self.rene_dag.get_total_time()

        logging.info("Iteration %s: cost change %s", self.iteration, cost_change)
        logging.info("Iteration %s: total cost %s", self.iteration, total_cost)
        logging.info("Iteration %s: refined cost %s", self.iteration, refined_cost)
        logging.info(
            "Iteration %s: total time %s",
            self.iteration,
            total_time,
        )
        # Step 7: Do some bookkeeping at tne end of the iteration
        self.costs.append(total_cost)
        self.refined_costs.append(refined_cost)
        self.times.append(total_time)
        self.iteration += 1

        return copy.deepcopy(self.rene_dag)

    def run(self) -> Generator[ReneDAGOld, None, None]:
        """Run the PD algorithm iteratively to solve for the Pareto optimal schedule for each time breakpoint.

        This is the main workflow of the algorithm.

        Yields:
            The updated ReneDAG after each iteration of the PD algorithm.
        """
        # Step 0: Convert Rene AON dag to Rene AOA dag
        self.rene_dag_aoa: nx.DiGraph = self.aon_to_aoa(self.rene_dag)

        while True:
            new_rene_dag = self.run_one_iteration()
            if new_rene_dag is None:
                # The PD algorithm finishes
                logging.info(
                    "%%% The PD algorithm finishes, generating final pipeline and Pareto frontier... %%%"
                )
                self.draw_pareto_frontier(self.output_dir / "Pareto_frontier.png")
                break
            else:
                yield new_rene_dag

    def reduce_duration(self, s: set[int], t: set[int]) -> float:  # noqa: PLR0912
        """Reduce the duration of forward edges from s to t and increase the duration of backward edges from t to s.

        Arguments:
            s: Set of nodes in s set
            t: Set of nodes in t set

        Returns:
            The cost change
        """
        reduce_edges: list[tuple[int, int]] = list()
        increase_edges: list[tuple[int, int]] = list()
        for node_id in s:
            for child_id in list(self.capacity_graph.successors(node_id)):
                if child_id in t:
                    reduce_edges.append((node_id, child_id))

        for node_id in t:
            for child_id in list(self.capacity_graph.successors(node_id)):
                if child_id in s:
                    increase_edges.append((node_id, child_id))

        logging.info("Iteration %s: reduce edges %s", self.iteration, reduce_edges)
        logging.info("Iteration %s: increase edges %s", self.iteration, increase_edges)
        if len(reduce_edges) == 0:
            return 0.0

        cost_change: float = 0.0

        increase_insts: list[str] = list()
        reduce_insts: list[str] = list()

        for u, v in reduce_edges:
            inst: Instruction = self.capacity_graph[u][v]["inst"]
            reduce_insts.append(repr(inst))
            if inst.duration - 1 < inst.min_duration or isinstance(inst, _Dummy):
                return float("inf")
            cost_change += inst.get_derivative(inst.duration, inst.duration - 1) * 1
            inst.duration -= 1
            logging.info("Reduced duration of %s to %s", repr(inst), inst.duration)

        for u, v in increase_edges:
            inst = self.capacity_graph[u][v]["inst"]
            logging.info("Increase edge: [%s, %s] %s", u, v, repr(inst))
            increase_insts.append(repr(inst))
            # Notice: dummy edge is always valid for increasing duration
            if isinstance(inst, _Dummy):
                inst.duration += 1
                logging.info(
                    "Increase edge is dummy: [%s, %s] %s, duration: %s",
                    u,
                    v,
                    repr(inst),
                    inst.duration,
                )
            elif inst.duration + 1 > inst.max_duration:
                return float("inf")
            else:
                cost_change -= inst.get_derivative(inst.duration, inst.duration + 1) * 1
                inst.duration += 1
                logging.info(
                    "Increase edge is non-dummy: [%s, %s] %s", u, v, repr(inst)
                )
        logging.info("Iteration %s: reduce insts %s", self.iteration, reduce_insts)
        logging.info("Iteration %s: increase insts %s", self.iteration, increase_insts)
        self.rene_dag.changed = True
        return cost_change

    def draw_aoa_graph(self, path: str) -> None:
        """Draw the AOA graph to the given path.

        Arguments:
            path: The path to save the graph.
        """
        plt.figure(figsize=(20, 20))
        pos = nx.circular_layout(self.critical_aoa_dag)
        nx.draw(self.critical_aoa_dag, pos, with_labels=True, font_weight="bold")
        node_labels = nx.get_node_attributes(self.critical_aoa_dag, "repr")
        nx.draw_networkx_labels(self.critical_aoa_dag, pos, labels=node_labels)
        # edge_labels = nx.get_edge_attributes(self.critical_aoa_dag, "weight")
        # nx.draw_networkx_edge_labels(self.critical_aoa_dag, pos, edge_labels=edge_labels)
        plt.tight_layout()
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_capacity_graph(self, path: str) -> None:
        """Draw the capacity graph to the given path.

        Arguments:
            path: The path to save the graph.
        """
        plt.figure(figsize=(30, 30))
        pos = nx.circular_layout(self.capacity_graph)
        nx.draw(self.capacity_graph, pos, with_labels=True, font_weight="bold")

        # set the attribute of edge as a combination of lb and ub, and round lb and ub to 2 decimal places
        for edge in self.capacity_graph.edges:
            if type(self.capacity_graph.edges[edge]["inst"]) != _Dummy:
                self.capacity_graph.edges[edge][
                    "label"
                ] = f"{repr(self.capacity_graph.edges[edge]['inst'])}: \
                    ({round(self.capacity_graph.edges[edge]['lb'], 2)}, \
                    {round(self.capacity_graph.edges[edge]['ub'], 2)})"

        edge_labels = nx.get_edge_attributes(self.capacity_graph, "label")
        nx.draw_networkx_edge_labels(self.capacity_graph, pos, edge_labels=edge_labels)
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()

    def draw_pareto_frontier(self, path: Path) -> None:
        """Draw the pareto frontier to the given path.

        Arguments:
            path: The path to save the graph.
        """
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
        ax.plot(self.times, self.costs, label="cost")
        ax.plot(self.times, self.refined_costs, label="refined cost")
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        ax.legend()
        fig.savefig(path, format="PNG")
        plt.clf()
        plt.close()
