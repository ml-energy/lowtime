"""A time-cost trade-off solver based on the PD-algorithm."""


from __future__ import annotations

import sys
import copy
import logging
from pathlib import Path
from queue import SimpleQueue
from collections import deque
from collections.abc import Generator

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from networkx.algorithms.flow import edmonds_karp  # type: ignore
from attrs import define

from rene.constants import (
    FP_ERROR,
    DEFAULT_RECTANGLE_ARGS,
    DEFAULT_ANNOTATION_ARGS,
    DEFAULT_LINE_ARGS,
)
from rene.dag import ReneDAGOld
from rene.perseus.instruction import Instruction
from rene.operation import DummyOperation, Operation
from rene.graph_utils import aon_dag_to_aoa_dag, get_total_time, get_total_cost
from rene.exceptions import ReneFlowError

logger = logging.getLogger(__name__)


@define
class IterationResult:
    """A POD to hold the results for one PD iteration."""
    dag: nx.DiGraph
    cost_change: float


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

    def run(self) -> Generator[IterationResult, None, None]:
        """Run the algorithm and yield a DAG after each iteration.

        The solver will not deepcopy operations on the DAG but rather in-place modify them
        for performance reasons. The caller should deepcopy the DAG if needed before running
        the next iteration.
        """
        logger.info("Starting Phillips-Dessouky solver.")

        # Convert the original activity-on-node DAG to activity-on-arc DAG form.
        # AOA DAGs are purely internal. All public input and output of this class
        # should be in AON form.
        aoa_dag = aon_dag_to_aoa_dag(self.aon_dag, attr_name="op")

        # Finding the critical DAG is the first step of each iteration, but that's
        # also useful for computing the current execution time of the DAG *before*
        # running each iteration. So we run this step here and do `to_critical_dag`
        # again end of the loop.
        critical_dag = self.to_critical_dag(aoa_dag)

        logger.info("Before PD iteration")
        logger.info("Total quantized time: %d", get_total_time(critical_dag))
        logger.info("Total cost: %f", get_total_cost(aoa_dag, mode="edge"))
        logger.debug("Number of nodes: %d", critical_dag.number_of_nodes())
        logger.debug("Number of edges: %d", critical_dag.number_of_edges())
        non_dummy_ops = [
            attr["op"]
            for _, _, attr in critical_dag.edges(data=True)
            if not attr["op"].is_dummy
        ]
        logger.debug("Number of non-dummy operations: %d", len(non_dummy_ops))
        logger.debug(
            "Sum of non-dummy durations: %d", sum(op.duration for op in non_dummy_ops)
        )

        # Iteratively reduce the execution time of the DAG.
        for iteration in range(sys.maxsize):
            logger.info(">>> Iteration %d", iteration)

            capacity_dag = self.annotate_capacities(critical_dag)
            logger.debug(
                "Total lb value: %f",
                sum([capacity_dag[u][v]["lb"] for u, v in capacity_dag.edges]),
            )
            logger.debug(
                "Total ub value: %f",
                sum([capacity_dag[u][v]["ub"] for u, v in capacity_dag.edges]),
            )

            try:
                s_set, t_set = self.find_min_cut(capacity_dag)
            except ReneFlowError as e:
                logger.info("Could not find minimum cut: %s", e.message)
                logger.info("Terminating PD iteration.")
                break

            cost_change = self.reduce_durations(capacity_dag, s_set, t_set)
            if cost_change == float("inf") or abs(cost_change) < FP_ERROR:
                logger.info("No further time reduction possible.")
                logger.info("Terminating PD iteration.")
                break

            # We directly modify operation attributes in the DAG, so after we
            # ran one iteration, the AON DAG holds updated attributes.
            yield IterationResult(dag=self.aon_dag, cost_change=cost_change)

            logger.info("Cost change: %f", cost_change)

            critical_dag = self.to_critical_dag(aoa_dag)
            logger.info("Total quantized time: %f", get_total_time(critical_dag))
            logger.info("Total cost: %f", get_total_cost(aoa_dag, mode="edge"))
            logger.debug("Number of nodes: %d", critical_dag.number_of_nodes())
            logger.debug("Number of edges: %d", critical_dag.number_of_edges())
            non_dummy_ops = [
                attr["op"]
                for _, _, attr in critical_dag.edges(data=True)
                if not attr["op"].is_dummy
            ]
            logger.debug("Number of non-dummy operations: %d", len(non_dummy_ops))
            logger.debug(
                "Sum of non-dummy durations: %d", sum(op.duration for op in non_dummy_ops)
            )

    def reduce_durations(
        self, capacity_dag: nx.DiGraph, s_set: set[int], t_set: set[int]
    ) -> float:
        """Modify operation durations to reduce the DAG execution time by 1."""
        speed_up_edges: list[Operation] = []
        for node_id in s_set:
            for child_id in list(capacity_dag.successors(node_id)):
                if child_id in t_set:
                    op: Operation = capacity_dag[node_id][child_id]["op"]
                    speed_up_edges.append(op)

        slow_down_edges: list[Operation] = []
        for node_id in t_set:
            for child_id in list(capacity_dag.successors(node_id)):
                if child_id in s_set:
                    op: Operation = capacity_dag[node_id][child_id]["op"]
                    slow_down_edges.append(op)

        if not speed_up_edges:
            logger.info("No speed up candidate operations.")
            return 0.0

        cost_change = 0.0

        # Reduce the duration of edges (speed up) by quant_time 1.
        for op in speed_up_edges:
            if op.is_dummy:
                logger.info("Cannot speed up dummy operation.")
                return float("inf")
            if op.duration - 1 < op.min_duration:
                logger.info("Operation %s has reached the limit of speed up", op)
                return float("inf")
            cost_change += abs(op.get_cost(op.duration - 1) - op.get_cost(op.duration))
            logger.info("Sped up %s to %d", op, op.duration - 1)
            op.duration -= 1

        # Increase the duration of edges (slow down) by quant_time 1.
        for op in slow_down_edges:
            # Dummy edges can always be slowed down.
            if op.is_dummy:
                logger.info("Slowed down %s to %d", op, op.duration + 1)
                op.duration += 1
            elif op.duration + 1 > op.max_duration:
                logger.info("Operation %s has reached the limit of slow down", op)
                return float("inf")
            else:
                cost_change += abs(op.get_cost(op.duration) - op.get_cost(op.duration + 1))
                logger.info("Slowed down %s to %d", op, op.duration + 1)
                op.duration += 1

        return cost_change

    def find_min_cut(self, capacity_dag: nx.DiGraph) -> tuple[set[int], set[int]]:
        """Find the min cut of the capacity DAG.

        Assumptions:
            - The capacity DAG is in AOA form.
            - The capacity DAG has been annotated with `lb` and `ub` attributes on edges,
                representing the lower and upper bounds of the flow on the edge.

        Returns:
            A tuple of (s_set, t_set) where s_set is the set of nodes on the source side
            of the min cut and t_set is the set of nodes on the sink side of the min cut.
            Returns None if no feasible flow exists.

        Raises:
            ReneFlowError: When no feasible flow exists.
        """
        source_node = capacity_dag.graph["source_node"]
        sink_node = capacity_dag.graph["sink_node"]

        # In order to solve max flow on edges with both lower and upper bounds,
        # we first need to convert it to another DAG that only has upper bounds.
        unbound_dag = nx.DiGraph(capacity_dag)

        # For every edge, capacity = ub - lb.
        for _, _, edge_attrs in unbound_dag.edges(data=True):
            edge_attrs["capacity"] = edge_attrs["ub"] - edge_attrs["lb"]

        # Add a new node s', which will become the new source node.
        # We constructed the AOA DAG, so we know that node IDs are integers.
        node_ids: list[int] = list(unbound_dag.nodes)
        s_prime_id = max(node_ids) + 1
        unbound_dag.add_node(s_prime_id)

        # For every node u in the original graph, add an edge (s', u) with capacity
        # equal to the sum of all lower bounds of u's parents.
        for node_id in capacity_dag.nodes:
            capacity = 0.0
            for pred_id in capacity_dag.predecessors(node_id):
                capacity += capacity_dag[pred_id][node_id]["lb"]
            # print(capacity)
            unbound_dag.add_edge(s_prime_id, node_id, capacity=capacity)

        # Add a new node t', which will become the new sink node.
        t_prime_id = s_prime_id + 1
        unbound_dag.add_node(t_prime_id)

        # For every node u in the original graph, add an edge (u, t') with capacity
        # equal to the sum of all lower bounds of u's children.
        for node_id in capacity_dag.nodes:
            capacity = 0.0
            for succ_id in capacity_dag.successors(node_id):
                capacity += capacity_dag[node_id][succ_id]["lb"]
            # print(capacity)
            unbound_dag.add_edge(node_id, t_prime_id, capacity=capacity)

        logger.debug("Unbound DAG")
        logger.debug("Number of nodes: %d", unbound_dag.number_of_nodes())
        logger.debug("Number of edges: %d", unbound_dag.number_of_edges())
        logger.debug(
            "Sum of capacities: %f",
            sum(attr["capacity"] for _, _, attr in unbound_dag.edges(data=True)),
        )

        # Add an edge from t to s with infinite weight.
        unbound_dag.add_edge(
            sink_node,
            source_node,
            capacity=float("inf"),
        )

        # We're done with constructing the DAG with only flow upper bounds.
        # Find the maximum flow on this DAG.
        try:
            flow_value, flow_dict = nx.maximum_flow(
                unbound_dag,
                s_prime_id,
                t_prime_id,
                capacity="capacity",
                flow_func=edmonds_karp,
            )
        except nx.NetworkXUnbounded:
            raise ReneFlowError("ERROR: Infinite flow for unbounded DAG.")

        logger.debug("After first max flow")
        total_flow = 0.0
        for d in flow_dict.values():
            for flow in d.values():
                total_flow += flow
        logger.debug("Sum of all flow values: %d", total_flow)

        # Check if residual graph is saturated. If so, we have a feasible flow.
        for node_id in unbound_dag.successors(s_prime_id):
            if (
                abs(
                    # unbound_res_graph[s_prime_id][node_id]["flow"]
                    flow_dict[s_prime_id][node_id]
                    - unbound_dag[s_prime_id][node_id]["capacity"]
                )
                > FP_ERROR
            ):
                logger.error(
                    "s' -> %s unsaturated (flow: %s, capacity: %s)",
                    node_id,
                    flow_dict[s_prime_id][node_id],
                    unbound_dag[s_prime_id][node_id]["capacity"],
                )
                raise ReneFlowError("ERROR: Max flow on unbounded DAG didn't saturate.")
        for node_id in unbound_dag.predecessors(t_prime_id):
            if (
                abs(
                    # unbound_res_graph[node_id][t_prime_id]["flow"]
                    flow_dict[node_id][t_prime_id]
                    - unbound_dag[node_id][t_prime_id]["capacity"]
                )
                > FP_ERROR
            ):
                logger.error(
                    "%s -> t' unsaturated (flow: %s, capacity: %s)",
                    node_id,
                    flow_dict[node_id][t_prime_id],
                    unbound_dag[node_id][t_prime_id]["capacity"],
                )
                raise ReneFlowError("ERROR: Max flow on unbounded DAG didn't saturate.")

        # We have a feasible flow. Construct a new residual graph with the same
        # shape as the capacity DAG so that we can find the min cut.
        # TODO(JW): From here, it's basically:
        # 1. Retrieve the flow amounts to the original capacity graph, where for
        #    each edge u -> v, the flow amount is `weight = flow + lb`.
        # 2. Construct a new residual graph (same shape as capacity DAG) with
        #    u -> v capacity `ub - weight` (=`ub - flow - lb`) and v -> u capacity
        #    `weight - lb` (=`flow`).
        # 3. Run max flow on the new residual graph from 1.
        # 4. Find the s-t cut induced by the maximum flow from 2.
        # Steps 1 and 2 can be done in one step by directly converting the unbounded
        # residual graph into a new one with u -> v capacity `ub - flow - lb` and
        # v -> u capacity `flow`.
        # Steps 3 and 4 can be done in one step with `nx.minimum_cut`. Especially,
        # various `flow_func`s like `edmonds_karp` support the `residual_graph`
        # keyword argument, which is exactly what we need.

        # TODO(JW): Step 1 above.
        for u, v in capacity_dag.edges:
            # f'(u,v) = f(u,v) - lb(u,v)
            capacity_dag[u][v]["weight"] = flow_dict[u][v] + capacity_dag[u][v]["lb"]

        # TODO(JW): Step 2 above.
        residual_graph = nx.DiGraph(capacity_dag)
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
                    )
                )
        for u, v, capacity in edge_pending:
            residual_graph.add_edge(u, v, capacity=capacity)

        # TODO(JW): Step 3 above.
        try:
            _, flow_dict = nx.maximum_flow(
                residual_graph,
                source_node,
                sink_node,
                capacity="capacity",
                flow_func=edmonds_karp,
            )
        except nx.NetworkXUnbounded:
            raise ReneFlowError("ERROR: Infinite flow on capacity residual graph.")

        # Add additional flow we get to the original graph
        for u, v in capacity_dag.edges:
            capacity_dag[u][v]["weight"] += flow_dict[u][v]
            capacity_dag[u][v]["weight"] -= flow_dict[v][u]

        # Construct the new residual graph.
        new_residual = nx.DiGraph(capacity_dag)
        add_candidates = []
        for u, v in new_residual.edges:
            new_residual[u][v]["weight"] = (
                capacity_dag[u][v]["ub"] - capacity_dag[u][v]["weight"]
            )
            add_candidates.append(
                (v, u, capacity_dag[u][v]["weight"] - capacity_dag[u][v]["lb"])
            )
        for u, v, weight in add_candidates:
            new_residual.add_edge(u, v, weight=weight)

        logger.debug("New residual graph")
        logger.debug("Number of nodes: %d", new_residual.number_of_nodes())
        logger.debug("Number of edges: %d", new_residual.number_of_edges())
        logger.debug(
            "Sum of weights: %f",
            sum(attr["weight"] for _, _, attr in new_residual.edges(data=True)),
        )

        # TODO(JW): Step 4 above.
        visited, _ = self.search_path_dfs(new_residual, source_node, sink_node)
        s_set: set[int] = set()
        t_set: set[int] = set()
        for i in range(len(visited)):
            if visited[i]:
                s_set.add(i)
            else:
                t_set.add(i)

        logger.debug("Minimum s-t cut")
        logger.debug("Size of s set: %d", len(s_set))
        logger.debug("Size of t set: %d", len(t_set))

        return (s_set, t_set)

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

    def annotate_capacities(self, critical_dag: nx.DiGraph) -> nx.DiGraph:
        """Annotate the critical DAG with flow capacities.

        Returns:
            A shallow copy of the input critical DAG with each edge annotated with
            `lb` and `ub` attributes, each representing flow lower and upper bounds.
        """
        # XXX(JW): Do we need to shallow copy? Why not add lb and ub directly to
        # capacity_dag edges?
        capacity_dag = nx.DiGraph(critical_dag)

        # XXX(JW): Why not actual float("inf")?
        inf = 10000.0
        for _, _, edge_attr in capacity_dag.edges(data=True):
            op: Operation = edge_attr["op"]
            # Dummy operations don't constrain the flow.
            if op.is_dummy:
                lb, ub = 0.0, inf
            # Special case when the operation has only one execution option.
            elif op.duration == op.min_duration == op.max_duration:
                lb, ub = 0.0, inf
            # Typical non-dummy operation.
            else:
                # Can this operation be slowed down?
                if op.duration + 1 <= op.max_duration:
                    # Absolute right subgradient.
                    lb = abs(op.get_cost(op.duration) - op.get_cost(op.duration + 1))
                else:
                    lb = 0.0
                # Can this operation be sped up?
                if op.duration - 1 >= op.min_duration:
                    # Absolute left subgradient.
                    ub = abs(op.get_cost(op.duration - 1) - op.get_cost(op.duration))
                else:
                    ub = inf

            # XXX(JW): Why is this rouding needed?
            edge_attr["lb"] = lb // FP_ERROR * FP_ERROR
            edge_attr["ub"] = ub // FP_ERROR * FP_ERROR

        return capacity_dag

    def to_critical_dag(self, aoa_dag: nx.DiGraph) -> nx.DiGraph:
        """Convert the AOA DAG to a critical AOA DAG where only critical edges remain."""
        # Clear all earliest/latest start/end times.
        for _, _, edge_attrs in aoa_dag.edges(data=True):
            operation: Operation = edge_attrs["op"]
            operation.reset_times()

        # Run the forward pass to set earliest start/end times.
        # TODO(JW): I think this implementation is strange. Why don't we just
        # go through (u, v) with `nx.bfs_edge` and find the latest earliest_finish
        # time among u's predecessor edges? Similar applies for the backward pass.
        for node_id in nx.topological_sort(aoa_dag):
            for succ_id in aoa_dag.successors(node_id):
                cur_op: Operation = aoa_dag[node_id][succ_id]["op"]

                for succ_succ_id in aoa_dag.successors(succ_id):
                    next_op: Operation = aoa_dag[succ_id][succ_succ_id]["op"]

                    next_op.earliest_start = max(
                        next_op.earliest_start,
                        cur_op.earliest_finish,
                    )
                    next_op.earliest_finish = next_op.earliest_start + next_op.duration

        # Run the backward pass to set latest start/end times.
        # XXX(JW): The original implementation only worked because there is guaranteed
        # to be only one predecessor of the sink node because it was split into two nodes
        # when converting to AOA form. Rather, the latest finish time of the entire AOA
        # DAG should be derived by taking the max among all earliest_finish times of the
        # edges that have the sink node as their target.
        exit_node = aoa_dag.graph["sink_node"]
        exit_edge_source_candidates = list(aoa_dag.predecessors(exit_node))
        assert len(exit_edge_source_candidates) == 1
        exit_edge_op: Operation = aoa_dag[exit_edge_source_candidates[0]][exit_node][
            "op"
        ]
        exit_edge_op.latest_finish = exit_edge_op.earliest_finish
        exit_edge_op.latest_start = exit_edge_op.latest_finish - exit_edge_op.duration
        for node_id in reversed(list(nx.topological_sort(aoa_dag))):
            for pred_id in aoa_dag.predecessors(node_id):
                cur_op: Operation = aoa_dag[pred_id][node_id]["op"]

                for pred_pred_id in aoa_dag.predecessors(pred_id):
                    prev_op: Operation = aoa_dag[pred_pred_id][pred_id]["op"]

                    prev_op.latest_start = min(
                        prev_op.latest_start,
                        cur_op.latest_start - prev_op.duration,
                    )
                    prev_op.latest_finish = prev_op.latest_start + prev_op.duration

        # Remove all edges that are not on the critical path.
        critical_dag = nx.DiGraph(aoa_dag)
        # XXX(JW): I don't know why we should do a topological sort here. Why not just
        # go through all edges like:
        # for u, v, edge_attrs in aoa_dag.edges(data=True):
        #     op: Operation = edge_attrs["op"]
        #     if op.earliest_finish != op.latest_finish:
        #         critical_dag.remove_edge(u, v)
        for node_id in nx.topological_sort(aoa_dag):
            for succ_id in aoa_dag.successors(node_id):
                cur_op: Operation = aoa_dag[node_id][succ_id]["op"]
                if cur_op.latest_finish != cur_op.earliest_finish:
                    critical_dag.remove_edge(node_id, succ_id)

        # Copy over source and sink node IDs.
        source_id = critical_dag.graph["source_node"] = aoa_dag.graph["source_node"]
        sink_id = critical_dag.graph["sink_node"] = aoa_dag.graph["sink_node"]
        if source_id not in critical_dag and source_id in aoa_dag:
            raise RuntimeError(
                "Source node was removed from the DAG when getting critical DAG."
            )
        if sink_id not in critical_dag and sink_id in aoa_dag:
            raise RuntimeError(
                "Sink node was removed from the DAG when getting critical DAG."
            )

        return critical_dag


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
        # XXX(JW): Constructing the `flow_dict` needs an extra step of going through all
        # the edges in the graph. We can directly use `edmonds_karp` to get the residual
        # graph, from which we can check if the residual graph is saturated and reconstruct
        # flow values on original graph (i.e., adding `lb` to current flow).
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
        logging.info("Iteration %s: total time %s", self.iteration, total_time)
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
