"""Defines the DAG of instructions and algorithms to define their execution times."""

from __future__ import annotations

import inspect
import itertools
import logging
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure # type: ignore
from matplotlib.ticker import FormatStrFormatter  # type: ignore
from typing import Iterable, Sequence, Type, Callable, Generator, Literal
from queue import SimpleQueue
from collections import deque
import typing

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
    """DAG of instructions and analysis methods, represented in Activity-on-Node form (AON). """

    # pylint: disable=dangerous-default-value,too-many-branches
    def __init__(
        self,
        schedule_type: Callable[[int, int, int], Iterable[Instruction]],
        num_stages: int,
        num_micro_batches: int,
        time_costs: dict[Type[Instruction], dict[int, list[tuple]]],
        dependency_rules: Sequence[Callable[..., bool]] = [forward_dep, backward_dep],
        output_dir: str = "",
        fit_method: str = "linear",
    ) -> None:
        """Instantiate instructions and construct the DAG.

        Arguments:
            schedule_type: A callable that returns an iterable of pipeline instructions.
                Can also be a subclass of `rene.schedule.PipeSchedule` like `Synchronous1F1B`.
            num_stages: The number of pipeline stages.
            num_micro_batches: The number of micro batches in the pipeline.
            time_costs: A dict that maps instruction type to a dict of stage_id : list of (duration, cost, frequency) tuples.
            output_dir: output directory for figures
            dependency_rules: A list of functions that define the dependency between instructions.
            fit_method: The method to fit the time cost data. Can be "linear" or "piecewise-linear".

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

        # For graph generation
        self.node_id: int = 0
        self.inst_map: dict[str, Instruction] = dict()
        self.inst_id_map: dict[str, int] = dict()
        self._dag: nx.DiGraph = nx.DiGraph()

        # For interpolation caching
        self.coeffs_dict: dict[Type[Instruction], dict[int, np.ndarray]] = dict()

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
            if len(type_hints) != 2:
                raise ValueError(
                    "Dependency rules must have exactly two type-annotated arguments."
                )
            for type_hint in type_hints.values():
                if not isinstance(type_hint, InstructionType):
                    raise ValueError(
                        f"Unexpected instruction type '{type_hint}'. "
                        f"Should be one of {InstructionType.subclass_names}"
                    )
                
        # Introduce dummy entry and exit nodes for analysis convenience.
        # Assign node_id 0 to entry_node and node_id 1 to exit_node
        self.entry_node = _Dummy(-1, -1, duration=0.0, repr="Entry")
        self._dag.add_node(self.node_id, inst=self.entry_node)
        self.inst_map[self.entry_node.__repr__()] = self.entry_node
        self.inst_id_map[self.entry_node.__repr__()] = self.node_id
        self.node_id += 1
        self.exit_node = _Dummy(-1, -1, duration=0.0, repr="Exit")
        self._dag.add_node(self.node_id, inst=self.exit_node)
        self.inst_map[self.exit_node.__repr__()] = self.exit_node
        self.inst_id_map[self.exit_node.__repr__()] = self.node_id
        self.node_id += 1

        # Generate instructions from `PipelineSchedule` and pipeline configurations.
        logging.info("Generate instructions and creating the DAG...")
        self._insts: list[Instruction] = []
        for stage_ind in range(self.num_stages):
            stage = self.schedule_type(
                self.num_stages, self.num_micro_batches, stage_ind
            )
            prev_inst = None
            for inst in stage:
                # When `durations` is `None`, it means that `inst` already has `duration` set.
                # if self.durations is not None:
                #     inst.duration = self.durations[type(inst)][inst.stage_id]

                # Sort the (duration, cost, frequency) tuple by reverse duration
                self.time_costs[type(inst)][stage_ind] = sorted(time_costs[type(inst)][stage_ind], reverse=True)
                inst.time_costs = self.time_costs[type(inst)][stage_ind]
                # Sanity check
                if (len(self.time_costs[type(inst)][stage_ind]) == 0):
                    raise ValueError(
                        f"No time-cost meta-data for instruction '{inst.__repr__()}'. "
                    )
                # Pick longest duration by default, as default schedule policy is "eager"
                inst.duration = self.time_costs[type(inst)][stage_ind][0][0]
                # Set min/max duration for each instruction
                inst.max_duration = self.time_costs[type(inst)][stage_ind][0][0]
                inst.min_duration = self.time_costs[type(inst)][stage_ind][-1][0]

                # Set the output directory for each instruction
                inst.output_dir = self.output_dir
                # Do interpolation here
                if (type(inst) not in self.coeffs_dict):
                    self.coeffs_dict[type(inst)] = dict()
                if (inst.stage_id not in self.coeffs_dict[type(inst)]):
                    self.coeffs_dict[type(inst)][inst.stage_id] = inst.interpolate(self.fit_method)
                else:
                    inst.fit_coeffs = self.coeffs_dict[type(inst)][inst.stage_id]
                    inst.fit_method = self.fit_method
                # inst.unit_cost = abs(inst.k)

                self._insts.append(inst)

                # add a new node
                if inst.__repr__() not in self.inst_map:
                    self._dag.add_node(self.node_id, inst=inst)
                    self.inst_map[inst.__repr__()] = inst
                    self.inst_id_map[inst.__repr__()] = self.node_id
                    self.node_id += 1

                if prev_inst is not None:
                    self._dag.add_edge(self.inst_id_map[prev_inst.__repr__()], self.inst_id_map[inst.__repr__()])
                    prev_inst.then(inst)
                prev_inst = inst
            prev_inst = None

        # Define dependencies by the dependency rules passed in.
        for inst1, inst2 in itertools.product(self._insts, self._insts):
            if self._is_dependent(inst1, inst2):
                inst1.then(inst2)
                self._dag.add_edge(self.inst_id_map[repr(inst1)], self.inst_id_map[repr(inst2)])

        for node in self._insts:
            if not node.parents:
                self.entry_node.then(node)
                self._dag.add_edge(self.inst_id_map[self.entry_node.__repr__()], self.inst_id_map[node.__repr__()])
            if not node.children:
                node.then(self.exit_node)
                self._dag.add_edge(self.inst_id_map[node.__repr__()], self.inst_id_map[self.exit_node.__repr__()])

        # Don't do annotation at here
        # # Annotate earliest/latest start/finish/slack times in nodes.
        # self.annotate_nodes()

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
                for inst, type in zip([inst1, inst2], type_hints.values())
            ):
                result = rule(inst1, inst2)
                if not isinstance(result, bool):
                    raise RuntimeError("Dependency rule returned a non-boolean value.")
                if result:
                    return True
        return False

    def get_critical_path(self) -> list[Instruction]:
        """Return the critical path of the DAG.

        When there are multiple possible critical paths, choose the smoothest,
        i.e. one with minimum number of `stage_id` changes along the path.
        """
        # Length is the amount of total `stage_id` changes along the critical path.
        smallest_length, critical_path = float("inf"), []
        stack: deque[tuple[float, list[Instruction]]] = deque()
        stack.append((0.0, [self.entry_node]))

        while stack:
            length, path = stack.pop()
            node = path[-1]
            if node is self.exit_node and length < smallest_length:
                smallest_length, critical_path = length, path
            for child in node.children:
                # Only go through nodes on the critical path.
                # Cannot use the `==` operator due to floating point errors.
                if abs(child.earliest_start - child.latest_start) < 1e-5:
                    if isinstance(node, _Dummy) or isinstance(child, _Dummy):
                        stage_diff = 0.0
                    else:
                        stage_diff = abs(node.stage_id - child.stage_id)
                    stack.append((length + stage_diff, path + [child]))

        # Slice out entry and exit nodes
        return list(filter(lambda node: not isinstance(node, _Dummy), critical_path))

    @property
    def total_execution_time(self) -> float:
        """Return the finish time of the last instruction."""
        assert (
            self.exit_node.earliest_finish == self.exit_node.latest_finish
        ), "Dummy exit node is not on the critical path."
        return self.exit_node.earliest_finish

    @property
    def insts(self) -> Generator[Instruction, None, None]:
        """Yield non-dummy instructions."""
        yield from filter(lambda inst: not isinstance(inst, _Dummy), self._insts)

    @property
    def dag(self) -> nx.DiGraph:
        """Return a networkx graph representation of the dag"""
        return self._dag

    def schedule(self, algo: Literal["eager", "lazy", "pd"] = "eager") -> None:
        """Determine the actual start/finish times of all instructions.

        Available algorithms:
            eager: Whenever I can execute, I immediately execute.
            lazy: I postpone execution as much as possible.
            pd: Linearly optimal schedule for minimum energy cost
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
        # elif algo == "pd":
        #     self.run_pd_algo()
        else:
            raise NotImplementedError(
                f"Scheduling algorithm '{algo}' is not implemented"
            )

    def draw_aon_graph(self, path: str) -> None:
        pos = nx.spring_layout(self.dag)
        nx.draw(self.dag, pos, with_labels=True, font_weight='bold')
        labels = nx.get_node_attributes(self.dag, "repr")
        nx.draw_networkx_labels(self.dag, pos, labels=labels)
        plt.tight_layout()
        plt.savefig(path, format="PNG")
        plt.clf()
        plt.close()

class CriticalDAG(ReneDAG):
    """DAG of instructions on the critical path, represented in Activity-on-Node form (AON), a subgraph of ReneDAG. """
    def __init__(self,
        schedule_type: Callable[[int, int, int], Iterable[Instruction]],
        num_stages: int,
        num_micro_batches: int,
        time_costs: dict[Type[Instruction], dict[int, list[tuple]]],
        dependency_rules: Sequence[Callable[..., bool]] = [forward_dep, backward_dep], 
        output_dir: str = "",
        fit_method: str = "linear",
    ) -> None:
        super(CriticalDAG, self).__init__(schedule_type, num_stages, num_micro_batches, time_costs, dependency_rules, output_dir, fit_method)
        logging.info("Initializing CriticalDAG...")
        # store the original DAG as complete dag 
        self.complete_dag = self.dag
        # store the critical DAG as the new dag 
        self.update_critical_dag()

    def annotate_nodes(self) -> None:
        """Annotate earliest/latest start/finish/slack times in nodes.
        """
        logging.info("Annotating nodes with start/finish/slack times...")
        # Forward computation: Assign earliest start and finish times
        self.entry_node.earliest_start = 0.0
        self.entry_node.earliest_finish = 0.0
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[self.entry_node.__repr__()])

        visited: dict[int, bool] = dict()
        for node_id in self.complete_dag.nodes:
            visited[node_id] = False
        while not q.empty():
            node_id = q.get()
            if visited[node_id]:
                continue
            visited[node_id] = True
            node: Instruction = self.complete_dag.nodes[node_id]["inst"]
            for child_id in self.complete_dag.successors(node_id):
                child: Instruction = self.complete_dag.nodes[child_id]["inst"]
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
        q.put(self.inst_id_map[self.exit_node.__repr__()])

        for node_id in self.complete_dag.nodes:
            visited[node_id] = False
        while not q.empty():
            node_id = q.get()
            if visited[node_id]:
                continue
            visited[node_id] = True
            node: Instruction = self.complete_dag.nodes[node_id]["inst"]
            for parent_id in self.complete_dag.predecessors(node_id):
                parent: Instruction = self.complete_dag.nodes[parent_id]["inst"]
                # parent.latest_start = min(
                #     parent.latest_start, node.latest_start - parent.duration
                # )
                if parent.latest_start > node.latest_start - parent.duration:
                    visited[parent_id] = False
                    parent.latest_start = node.latest_start - parent.duration
                parent.latest_finish = parent.latest_start + parent.duration
                parent.slack = parent.latest_finish - parent.earliest_start - parent.duration
                    
                q.put(parent_id)

    def clear_annotations(self) -> None:
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[self.entry_node.__repr__()])

        visited: list[int] = []
        while not q.empty():
            cur_id: int = q.get()
            if cur_id in visited:
                continue
            visited.append(cur_id)
            cur_node: Instruction = self.complete_dag.nodes[cur_id]["inst"]
            cur_node.earliest_start = 0.0
            cur_node.latest_start = float("inf")
            cur_node.earliest_finish = 0.0
            cur_node.latest_finish = float("inf")
            cur_node.slack = 0.0
            for child_id in self.complete_dag.successors(cur_id):
                q.put(child_id)

    def update_critical_dag(self) -> None:
        """Update the critical DAG, which is a subgraph of self.complete_dag.
        """
        # TODO: add an if changed flag?
        self.annotate_nodes()
        critical_dag: nx.DiGraph = nx.DiGraph(self.complete_dag)
        # Start to construct critical path graph, in AON format
        # This is different than get_critical_path() in ReneDAG as it allows multiple critcal paths
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[self.entry_node.__repr__()])

        critical_ids: list[int] = []
        visited: list[int] = []
        logging.info("Updating critical dag...")
        while not q.empty():
            node_id = q.get()
            if node_id in visited:
                continue
            visited.append(node_id)
            node: Instruction = self.complete_dag.nodes[node_id]["inst"]
            if abs(node.latest_finish - node.earliest_start - node.duration) < 1e-10 and node_id not in critical_ids:
                critical_ids.append(node_id)
            for child_id in self.complete_dag.successors(node_id):
                q.put(child_id)
        # Remove all non-critical nodes
        for i in range(0, self.node_id):
            if i not in critical_ids:
                critical_dag.remove_node(i)

        self._dag = critical_dag

        # let's check if the critical dag has only a single path
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[self.entry_node.__repr__()])
        visited: list[int] = []
        while not q.empty():
            node_id = q.get()
            if node_id in visited:
                continue
            visited.append(node_id)
            if len(list(self._dag.successors(node_id))) > 1:
                raise Exception("Critical DAG has multiple paths!")
            for child_id in self._dag.successors(node_id):
                q.put(child_id)
    
