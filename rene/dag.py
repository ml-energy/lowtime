"""Defines the DAG of instructions and algorithms to define their execution times."""

from __future__ import annotations

import inspect
import itertools
import logging
import typing
from pathlib import Path
from typing import Iterable, Sequence, Type, Callable, Generator, Literal
from queue import SimpleQueue
from collections import deque

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np

from rene.constants import FP_ERROR
from rene.instruction import Instruction, InstructionType, Forward, Backward, _Dummy


def forward_dep(inst1: Forward, inst2: Forward) -> bool:
    """Dependency rule between Forward instructions.

    Forward(stage i, microbatch j) -> Forward(stage i+1, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id + 1 == inst2.stage_id
    )


def backward_dep(inst1: Backward, inst2: Backward) -> bool:
    """Dependency rule between Backward instructions.

    Backward(stage i+1, microbatch j) -> Backward(stage i, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id == inst2.stage_id + 1
    )


class ReneDAG:
    """DAG of instructions and analysis methods, represented in Activity-on-Node form (AON)."""

    def __init__(  # noqa: PLR0913
        self,
        schedule_type: Callable[[int, int, int], Iterable[Instruction]],
        num_stages: int,
        num_micro_batches: int,
        time_costs: dict[Type[Instruction], dict[int, list[tuple[float, float, int]]]],
        unit_time: float,
        dependency_rules: Sequence[Callable[..., bool]] = [forward_dep, backward_dep],
        output_dir: Path | None = None,
        fit_method: Literal["linear", "piecewise-linear", "exponential"] = "linear",
        p2p_power: float = 0.0,
        initial_guess: dict[Type[Instruction], dict[int, list[float]]] = None,  # type: ignore
    ) -> None:
        """Instantiate instructions and construct the DAG.

        Arguments:
            schedule_type: A callable that returns an iterable of pipeline instructions.
                Can also be a subclass of `rene.schedule.PipeSchedule` like `Synchronous1F1B`.
            num_stages: The number of pipeline stages.
            num_micro_batches: The number of micro batches in the pipeline.
            time_costs: A dict that maps inst type to a dict of stage_id: list of (duration, cost, frequency) tuples.
            unit_time: The unit time of the time cost data.
            dependency_rules: A list of functions that define the dependency between instructions.
            output_dir: Output directory for figures. Pass `None` to disable outputs.
            fit_method: The method to fit the time cost data. Can be "linear" or "piecewise-linear" or "exponential".
            p2p_power: The power consumption of blocking on p2p communication between GPUs.
            initial_guess: A dict mapping inst type to a dict of stage_id:
                list of initial guess for parameters of "exponential" fit.

        ## Dependency rules

        ```python
        def forward_dep(inst1: Forward, inst2: Forward) -> bool:
            return inst1.micro_batch_id == inst2.micro_batch_id and inst1.stage_id + 1 == inst2.stage_id
        ```

        Dependency rules must be a type-annotated function that takes two arguments and returns `True`
        when there should be an dependency edge from `inst1` to `inst2` in the instruction DAG and
        `False` otherwise. Try to make this generate the minimum number of edges, since it affects
        the time complexity of critical path analysis (which is basically 2 * BFS + 1 * DFS).

        The two arguments must each be a subclass of `Instruction`, e.g. `Forward` and `Backward`.
        Then, `ReneDAG` will insepct the type annotations and only call `forward_dep` for
        `(inst1, inst2)` pairs where `isinstance(inst1, Forward)` and `isinstance(inst2, Forward)`,
        for example.
        """
        logging.info("Initializing ReneDAG...")
        self.schedule_type = schedule_type
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.time_costs = time_costs
        self.dependency_rules = dependency_rules
        self.scheduled = False
        self.fit_method = fit_method
        self.output_dir = output_dir

        # For decoupled instruction energy
        self.p2p_power = p2p_power

        # For graph generation
        self.node_id: int = 0
        self.inst_map: dict[str, Instruction] = {}
        self.inst_id_map: dict[str, int] = {}
        self._dag: nx.DiGraph = nx.DiGraph()
        self._critical_dag: nx.DiGraph = nx.DiGraph()
        self._insts: list[Instruction] = []
        self.changed = True

        # For interpolation
        self.coeffs_dict: dict[Type[Instruction], dict[int, np.ndarray]] = {}
        self.initial_guess: dict[
            Type[Instruction], dict[int, list[float]]
        ] = initial_guess

        # For frequency assignment
        self.stage_view: dict[int, list[Instruction]] = {}

        # For time recalculation
        self.unit_time = unit_time

        # Sanity check.
        if self.time_costs is not None:
            for meta_dic in self.time_costs.values():
                if len(meta_dic) != num_stages:
                    raise ValueError(
                        "`len(time_costs[instruction_type])` should be the same as `num_stages`"
                    )

        # Check the signature of depndency rules.
        for rule in self.dependency_rules:
            if not inspect.isfunction(rule):
                raise ValueError("Dependency rules should be a function.")
            type_hints = typing.get_type_hints(rule)
            if "return" in type_hints:
                type_hints.pop("return")
            if len(type_hints) != 2:  # noqa
                raise ValueError(
                    "Dependency rules must have exactly two type-annotated arguments."
                )
            for type_hint in type_hints.values():
                if not isinstance(type_hint, InstructionType):
                    raise ValueError(
                        f"Unexpected instruction type '{type_hint}'. "
                        f"Should be one of {InstructionType.subclass_names}"
                    )

        # Preprocess the time costs data
        for stage_ind in range(self.num_stages):
            for inst_type in [Forward, Backward]:
                # Sort the (duration, cost, frequency) tuple by reverse duration
                self.time_costs[inst_type][stage_ind] = sorted(
                    self.time_costs[inst_type][stage_ind], reverse=True
                )
                # Sanity check
                if len(self.time_costs[inst_type][stage_ind]) == 0:
                    raise ValueError(
                        f"No time-cost data for inst type '{inst_type}' at stage '{stage_ind}'. "
                    )
        # Initialize the DAG.
        self.init_dag()

    def _is_dependent(self, inst1: Instruction, inst2: Instruction) -> bool:
        """Check if there is a dependency from `inst1` to `inst2`.

        Checks the function type annotation and only call rules that
        have consistent type annotations with the types of `inst1` and `inst2`.
        """
        for rule in self.dependency_rules:
            type_hints = typing.get_type_hints(rule)
            if "return" in type_hints:
                type_hints.pop("return")
            if all(
                isinstance(inst, type)
                for inst, type in zip([inst1, inst2], type_hints.values())  # noqa
            ):
                result = rule(inst1, inst2)
                if not isinstance(result, bool):
                    raise RuntimeError("Dependency rule returned a non-boolean value.")
                if result:
                    return True
        return False

    def init_dag(self) -> None:  # noqa: PLR0912, PLR0915
        """Initialize the DAG."""
        # TODO: Eventually move the construction of the DAG out of this class.
        # The current code implicitly creates `Instruction`s based on the order of instruction generated by
        # `schedule_type` and populates time-cost values as they are created, but some users will want to
        # create `Instruction`s and assign time-cost values themselves.
        # There should be two ways to generate the DAG: (1) manual, (2) with an helper for well-known
        # pipeline schedules (perhaps put the helper inside `Synchronous1F1B`).

        # Introduce dummy entry and exit nodes for analysis convenience.
        # Assign node_id 0 to entry_node and node_id 1 to exit_node
        self.entry_node = _Dummy(-1, -1, duration=0.0, repr="Entry")
        self.entry_id = self.node_id
        self._dag.add_node(self.node_id, inst=self.entry_node)
        self.inst_map[repr(self.entry_node)] = self.entry_node
        self.inst_id_map[repr(self.entry_node)] = self.node_id
        self.exit_node = _Dummy(-1, -1, duration=0.0, repr="Exit")
        self.exit_id = self.node_id + 1
        self._dag.add_node(self.exit_id, inst=self.exit_node)
        self.inst_map[repr(self.exit_node)] = self.exit_node
        self.inst_id_map[repr(self.exit_node)] = self.exit_id
        self.node_id += 2

        # Generate instructions from `PipelineSchedule` and pipeline configurations.
        logging.info("Generate instructions and creating the DAG...")
        for stage_ind in range(self.num_stages):
            stage = self.schedule_type(
                self.num_stages, self.num_micro_batches, stage_ind
            )
            prev_inst = None
            self.stage_view[stage_ind] = []
            for inst in stage:
                # Get the time cost for this instruction
                inst.time_costs = self.time_costs[type(inst)][stage_ind]

                # Pick longest duration by default, as default schedule policy is "eager"
                inst.duration = self.time_costs[type(inst)][stage_ind][0][0]
                # Set min/max duration for each instruction
                inst.max_duration = self.time_costs[type(inst)][stage_ind][0][0]
                inst.min_duration = self.time_costs[type(inst)][stage_ind][-1][0]

                # Set the output directory for each instruction
                inst.output_dir = self.output_dir

                # Input some info needed by p2p blocking energy reduction
                inst.num_stages = self.num_stages
                inst.p2p_power = self.p2p_power

                # Set unit time for each instruction
                inst.unit_time = self.unit_time

                # Do interpolation here

                # check if initial_guess is an empty dict, if not, use the initial guess
                if self.initial_guess:
                    inst.initial_guess = self.initial_guess[type(inst)][inst.stage_id]

                if type(inst) not in self.coeffs_dict:
                    self.coeffs_dict[type(inst)] = {}
                if inst.stage_id not in self.coeffs_dict[type(inst)]:
                    self.coeffs_dict[type(inst)][inst.stage_id] = inst.fit(
                        self.fit_method
                    )
                else:
                    inst.fit_coeffs = self.coeffs_dict[type(inst)][inst.stage_id]
                    inst.fit_method = self.fit_method
                # inst.unit_cost = abs(inst.k)

                self._insts.append(inst)
                self.stage_view[stage_ind].append(inst)

                # add a new node
                if repr(inst) not in self.inst_map:
                    self._dag.add_node(self.node_id, inst=inst)
                    self.inst_map[repr(inst)] = inst
                    self.inst_id_map[repr(inst)] = self.node_id
                    self.node_id += 1

                if prev_inst is not None:
                    self._dag.add_edge(
                        self.inst_id_map[repr(prev_inst)],
                        self.inst_id_map[repr(inst)],
                    )
                prev_inst = inst
            prev_inst = None

        # Define dependencies by the dependency rules passed in.
        for inst1, inst2 in itertools.product(self._insts, self._insts):
            if self._is_dependent(inst1, inst2):
                self._dag.add_edge(
                    self.inst_id_map[repr(inst1)], self.inst_id_map[repr(inst2)]
                )

        for node in self._insts:
            # Add edges from entry_node to all nodes without predecessors.
            if self._dag.in_degree(self.inst_id_map[repr(node)]) == 0:
                self._dag.add_edge(
                    self.inst_id_map[repr(self.entry_node)],
                    self.inst_id_map[repr(node)],
                )
            # Add edges from all nodes without successors to exit_node.
            if self._dag.out_degree(self.inst_id_map[repr(node)]) == 0:
                self._dag.add_edge(
                    self.inst_id_map[repr(node)],
                    self.inst_id_map[repr(self.exit_node)],
                )

    @property
    def insts(self) -> Generator[Instruction, None, None]:
        """Yield non-dummy instructions."""
        yield from filter(lambda inst: not isinstance(inst, _Dummy), self._insts)

    @property
    def dag(self) -> nx.DiGraph:
        """Return a networkx graph representation of the dag."""
        return self._dag

    def schedule(self, algo: Literal["eager", "lazy"] = "eager") -> None:
        """Determine the actual start/finish times of all instructions.

        Available algorithms:
            eager: Whenever I can execute, I immediately execute.
            lazy: I postpone execution as much as possible.
        """
        self.scheduled = True
        if algo == "eager":
            for inst in self.insts:
                inst.actual_start = inst.earliest_start
                inst.actual_finish = inst.earliest_finish
        elif algo == "lazy":
            for inst in self.insts:
                inst.actual_start = inst.latest_start
                inst.actual_finish = inst.latest_finish
        else:
            raise NotImplementedError(
                f"Scheduling algorithm '{algo}' is not implemented"
            )

    def annotate_nodes(self) -> None:
        """Annotate earliest/latest start/finish times in nodes."""
        logging.info("Annotating nodes with earliest/latest start/finish times...")
        # Forward computation: Assign earliest start and finish times
        self.entry_node.earliest_start = 0.0
        self.entry_node.earliest_finish = 0.0
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[repr(self.entry_node)])

        visited: dict[int, bool] = {}
        for node_id in self._dag.nodes:
            visited[node_id] = False
        while not q.empty():
            node_id = q.get()
            if visited[node_id]:
                continue
            visited[node_id] = True
            node: Instruction = self._dag.nodes[node_id]["inst"]
            for child_id in self._dag.successors(node_id):
                child: Instruction = self._dag.nodes[child_id]["inst"]
                if child.earliest_start < node.earliest_finish:
                    visited[child_id] = False
                    child.earliest_start = node.earliest_finish
                child.earliest_finish = child.earliest_start + child.duration

                # child.earliest_start = max(child.earliest_start, node.earliest_finish)
                q.put(child_id)

        # Backward computation: Assign latest start and finish times
        # Exit node has duration 0, so latest finish and latest start should be the same.
        self.exit_node.latest_finish = self.exit_node.earliest_start
        self.exit_node.latest_start = self.exit_node.earliest_start
        q.put(self.inst_id_map[repr(self.exit_node)])

        for node_id in self._dag.nodes:
            visited[node_id] = False
        while not q.empty():
            node_id = q.get()
            if visited[node_id]:
                continue
            visited[node_id] = True
            node = self._dag.nodes[node_id]["inst"]
            for parent_id in self._dag.predecessors(node_id):
                parent: Instruction = self._dag.nodes[parent_id]["inst"]
                if parent.latest_start > node.latest_start - parent.duration:
                    visited[parent_id] = False
                    parent.latest_start = node.latest_start - parent.duration
                parent.latest_finish = parent.latest_start + parent.duration

                q.put(parent_id)

    def clear_annotations(self) -> None:
        """Clear all annotations in nodes."""
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[repr(self.entry_node)])

        visited: list[int] = []
        while not q.empty():
            cur_id: int = q.get()
            if cur_id in visited:
                continue
            visited.append(cur_id)
            cur_node: Instruction = self._dag.nodes[cur_id]["inst"]
            cur_node.earliest_start = 0.0
            cur_node.latest_start = float("inf")
            cur_node.earliest_finish = 0.0
            cur_node.latest_finish = float("inf")
            for child_id in self._dag.successors(cur_id):
                q.put(child_id)

    def get_critical_path(self) -> list[Instruction]:
        """Return a single critical path of the DAG."""
        critical_dag = self.get_critical_dag()
        critical_path: list[Instruction] = []
        q: deque[int] = deque()
        # do a DFS to get the critical path
        q.append(self.inst_id_map[repr(self.entry_node)])
        visited: list[int] = list()
        while len(q) > 0:
            cur_id = q.pop()
            visited.append(cur_id)
            critical_path.append(critical_dag.nodes[cur_id]["inst"])
            if cur_id == self.inst_id_map[repr(self.exit_node)]:
                break
            for succ_id in critical_dag.successors(cur_id):
                if succ_id not in visited:
                    q.append(succ_id)

        # Slice out entry and exit nodes
        return list(filter(lambda node: not isinstance(node, _Dummy), critical_path))

    def get_critical_dag(self) -> nx.DiGraph:
        """Update the critical DAG, which is a subgraph of self.complete_dag."""
        if self.changed:
            self.changed = False
            self.clear_annotations()
            self.annotate_nodes()
            critical_dag: nx.DiGraph = nx.DiGraph(self._dag)
            # Start to construct critical path graph, in AON format
            # This is different than get_critical_path() in ReneDAG as it allows multiple critcal paths
            q: SimpleQueue[int] = SimpleQueue()
            q.put(self.inst_id_map[repr(self.entry_node)])

            critical_ids: list[int] = []
            visited: list[int] = []
            logging.info("Updating critical dag...")
            while not q.empty():
                node_id = q.get()
                if node_id in visited:
                    continue
                visited.append(node_id)
                node: Instruction = self._dag.nodes[node_id]["inst"]
                if (
                    abs(node.latest_finish - node.earliest_start - node.duration)
                    < FP_ERROR
                    and node_id not in critical_ids
                ):
                    critical_ids.append(node_id)
                for child_id in self._dag.successors(node_id):
                    q.put(child_id)
            # Remove all non-critical nodes
            for i in range(0, self.node_id):
                if i not in critical_ids:
                    critical_dag.remove_node(i)

            self._critical_dag = critical_dag
        return self._critical_dag

    def get_total_cost(self) -> tuple[float, float]:
        """Get the total cost of the current pipeline.

        Returns:
            total_cost: The total cost
            refined_cost: The total cost refined with p2p blocking energy
        """
        q: SimpleQueue[int] = SimpleQueue()
        q.put(0)
        visited: list[str] = list()
        total_cost: float = 0.0

        while not q.empty():
            cur_id: int = q.get()
            cur_node: Instruction = self.dag.nodes[cur_id]["inst"]
            if repr(cur_node) in visited:
                continue
            visited.append(repr(cur_node))
            if not isinstance(cur_node, _Dummy):
                total_cost += cur_node.get_cost(cur_node.duration)
            for child_id in self.dag.successors(cur_id):
                q.put(child_id)

        # Refine the total cost with p2p blocking energy
        refined_cost = (
            total_cost + self.num_stages * self.get_total_time() * self.p2p_power
        )

        return (total_cost, refined_cost)

    def get_total_time(self) -> float:
        """Compute the total execution time of the current pipeline."""
        critical_path = self.get_critical_path()
        return sum(inst.duration for inst in critical_path) * self.unit_time

    def get_freq_assignment(self) -> list[list[int]]:
        """Assign frequency to each instruction in the pipeline based on the duration.

        Returns:
            total_freqs: The frequency assignment, indexed by stage_id
        """
        # Do binary search on inst.time_costs, list of (duration, cost, frequency) tuples, sorted by reverse duration
        for insts in self.stage_view.values():
            for inst in insts:
                # max/min duration should be common case
                if abs(inst.time_costs[0][0] - inst.duration) < FP_ERROR:
                    inst.frequency = inst.time_costs[0][2]
                elif abs(inst.time_costs[-1][0] - inst.duration) < FP_ERROR:
                    inst.frequency = inst.time_costs[-1][2]
                else:
                    inst.frequency = self.binary_search_frequency(inst)

        total_freqs: list[list[int]] = []
        for stage_id in sorted(self.stage_view.keys()):
            insts = self.stage_view[stage_id]
            freqs: list[int] = []
            reprs: list[str] = []
            for inst in insts:
                assert inst.frequency != -1
                freqs.append(inst.frequency)
                reprs.append(repr(inst))
            total_freqs.append(freqs)

        return total_freqs

    def binary_search_frequency(self, cur_node: Instruction) -> int:
        """Binary search the frequency based on the duration.

        Arguments:
            cur_node: The instruction to search

        Returns:
            frequency: The frequency
        """
        # start binary search
        left = 0
        right = len(cur_node.time_costs) - 1
        frequency = -1

        while left < right:
            mid = (left + right) // 2
            # if there is an exact match, or we are at the head/end of the list, we are done
            if (
                abs(cur_node.time_costs[mid][0] - cur_node.duration) < FP_ERROR
                or mid == 0
                or mid == len(cur_node.time_costs) - 1
            ):
                frequency = cur_node.time_costs[mid][2]
                break
            elif cur_node.time_costs[mid][0] < cur_node.duration:
                if cur_node.time_costs[mid - 1][0] > cur_node.duration:
                    # we are between two points, choose one with shorter duration since it is deadline problem
                    frequency = cur_node.time_costs[mid][2]
                    break
                right = mid
            elif cur_node.time_costs[mid][0] > cur_node.duration:
                if cur_node.time_costs[mid + 1][0] < cur_node.duration:
                    # we are between two points, choose one with shorter duration since it is deadline problem
                    frequency = cur_node.time_costs[mid + 1][2]
                    break
                left = mid + 1

        if frequency == -1:
            raise Exception(f"Cannot find frequency for {repr(cur_node)}")
        else:
            return frequency

    def draw_aon_graph(self, path: str) -> None:
        """Draw the graph in Activity-on-Node form (AON).

        Arguments:
            path: Path to save the graph.
        """
        pos = nx.spring_layout(self.dag)
        nx.draw(self.dag, pos, with_labels=True, font_weight="bold")
        labels = nx.get_node_attributes(self.dag, "repr")
        nx.draw_networkx_labels(self.dag, pos, labels=labels)
        plt.tight_layout()
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()
