"""Defines the DAG of instructions and algorithms to define their execution times."""

from __future__ import annotations

import inspect
import itertools
import networkx as nx
import numpy as np
from typing import Iterable, Sequence, Type, Callable, Generator, Literal
from queue import SimpleQueue
from collections import deque
import typing

from rene.instruction import Instruction, InstructionType, Forward, Backward, _Dummy
from rene.pd import PD_Solver


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


class InstructionDAG:
    """DAG of instructions and analysis methods, represented in Activity-on-Node form (AON). """

    # pylint: disable=dangerous-default-value,too-many-branches
    def __init__(
        self,
        schedule_type: Callable[[int, int, int], Iterable[Instruction]],
        num_stages: int,
        num_micro_batches: int,
        time_costs: dict[Type[Instruction], dict[int, list[tuple]]],
        dependency_rules: Sequence[Callable[..., bool]] = [forward_dep, backward_dep],
    ) -> None:
        """Instantiate instructions and construct the DAG.

        Arguments:
            schedule_type: A callable that returns an iterable of pipeline instructions.
                Can also be a subclass of `rene.schedule.PipeSchedule` like `Synchronous1F1B`.
            num_stages: The number of pipeline stages.
            num_micro_batches: The number of micro batches in the pipeline.
            time_costs: A dict that maps instruction type to a dict of stage_id : list of (duration, cost, frequency) tuples.
            dependency_rules: A list of functions that define the dependency between instructions.

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
        Then, `InstructionDAG` will insepct the type annotations and only call `forward_dep` for
        `(inst1, inst2)` pairs where `isinstance(inst1, Forward)` and `isinstance(inst2, Forward)`,
        for example.
        """
        self.schedule_type = schedule_type
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.time_costs = time_costs
        self.dependency_rules = dependency_rules
        self.scheduled = False

        # For graph generation
        self.node_id: int = 0
        self.inst_map: dict[str, Instruction] = dict()
        self.inst_id_map: dict[str, int] = dict()
        self.dag: nx.DiGraph = nx.DiGraph()

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

        # Generate instructions from `PipelineSchedule` and pipeline configurations.
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
                # Do linear interpolation here
                inst.k, inst.b = self.linear_interpolate(inst)
                inst.unit_cost = abs(inst.k)

                self._insts.append(inst)

                # add a new node
                if inst.__repr__() not in self.inst_map:
                    self.dag.add_node(self.node_id, inst=inst)
                    self.inst_map[inst.__repr__()] = inst
                    self.inst_id_map[inst.__repr__()] = self.node_id
                    self.node_id += 1

                if prev_inst is not None:
                    self.dag.add_edge(self.inst_id_map[prev_inst.__repr__()], self.inst_id_map[inst.__repr__()])
                    prev_inst.then(inst)
                prev_inst = inst
            prev_inst = None

        # Define dependencies by the dependency rules passed in.
        for inst1, inst2 in itertools.product(self._insts, self._insts):
            if self._is_dependent(inst1, inst2):
                inst1.then(inst2)
                self.dag.add_edge(self.inst_id_map[repr(inst1)], self.inst_id_map[repr(inst2)])

        # Introduce dummy entry and exit nodes for analysis convenience.
        self.entry_node = _Dummy(-1, -1, duration=0.0, repr="Entry")
        self.dag.add_node(self.node_id, inst=self.entry_node)
        self.inst_map[self.entry_node.__repr__()] = self.entry_node
        self.inst_id_map[self.entry_node.__repr__()] = self.node_id
        self.node_id += 1
        self.exit_node = _Dummy(-1, -1, duration=0.0, repr="Exit")
        self.dag.add_node(self.node_id, inst=self.exit_node)
        self.inst_map[self.exit_node.__repr__()] = self.exit_node
        self.inst_id_map[self.exit_node.__repr__()] = self.node_id
        self.node_id += 1
        for node in self._insts:
            if not node.parents:
                self.entry_node.then(node)
                self.dag.add_edge(self.inst_id_map[self.entry_node.__repr__()], self.inst_id_map[node.__repr__()])
            if not node.children:
                node.then(self.exit_node)
                self.dag.add_edge(self.inst_id_map[node.__repr__()], self.inst_id_map[self.exit_node.__repr__()])

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
    
    def get_critical_dag(self) -> nx.DiGraph:
        """Return the critical DAG, which is a subgraph of self.dag.
        """
        # TODO: add an if changed flag?
        self.annotate_nodes()
        critical_dag: nx.DiGraph = nx.DiGraph(self.dag)
        # Start to construct critical path graph, in AON format
        # This is different than get_critical_path() in InstructionDAG as it allows multiple critcal paths
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[self.entry_node.__repr__()])

        critical_ids: list[int] = []
        while not q.empty():
            node_id = q.get()
            node: Instruction = self.dag.nodes[node_id]["inst"]
            if abs(node.latest_finish - node.earliest_start - node.duration) < 1e-10 and node_id not in critical_ids:
                critical_ids.append(node_id)
            for child_id in self.dag.successors(node_id):
                q.put(child_id)
        # Remove all non-critical nodes
        for node_id in critical_ids:
            critical_dag.remove_node(node_id)
        return critical_dag
    
    def linear_interpolate(self, inst: Instruction) -> (float, float):
        """Do linear interpolation on the given instruction and its time-costs meta-data, return the unit cost (slope)

        Assumes self.time_costs[inst] has already been sorted.
        """
        # Get the slope from two endpoints
        # right_end = self.time_costs[type(inst)][inst.stage_id][0]
        # left_end = self.time_costs[type(inst)][inst.stage_id][-1]
        # unit_cost = abs((right_end[1] - left_end[1]) / (right_end[0] - left_end[0]))
        time_cost_list = self.time_costs[type(inst)][inst.stage_id]
        time_list = []
        cost_list = []
        for t, e, f in time_cost_list:
            time_list.append(t)
            cost_list.append(e)
        k, b = np.polyfit(time_list, cost_list, 1)
        print(f"Linear fit {type(inst)} {inst.stage_id} as y={k}x+{b}")

        return (k, b)

    def annotate_nodes(self) -> None:
        """Annotate earliest/latest start/finish/slack times in nodes.
        """
        # Forward computation: Assign earliest start and finish times
        self.entry_node.earliest_start = 0.0
        q: SimpleQueue[int] = SimpleQueue()
        q.put(self.inst_id_map[self.entry_node.__repr__()])

        while not q.empty():
            node_id = q.get()
            node: Instruction = self.dag.nodes[node_id]["inst"]
            for child_id in self.dag.successors(node_id):
                child: Instruction = self.dag.nodes[child_id]["inst"]
                child.earliest_start = max(child.earliest_start, node.earliest_finish)
                child.earliest_finish = child.earliest_start + child.duration
                q.put(child_id)

        # Backward computation: Assign latest start and finish times
        # Exit node has duration 0, so latest finish and latest start should be the same.
        self.exit_node.latest_finish = (
            self.exit_node.latest_start
        ) = self.exit_node.earliest_start
        q.put(self.inst_id_map[self.exit_node.__repr__()])

        while not q.empty():
            node_id = q.get()
            node: Instruction = self.dag.nodes[node_id]["inst"]
            for parent_id in self.dag.predecessors(node_id):
                parent: Instruction = self.dag.nodes[parent_id]["inst"]
                parent.latest_start = min(
                    parent.latest_start, node.latest_start - parent.duration
                )
                parent.latest_finish = parent.latest_start + parent.duration
                parent.slack = parent.latest_finish - parent.earliest_start - parent.duration
                q.put(parent_id)

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
        elif algo == "pd":
            self.run_pd_algo()
        else:
            raise NotImplementedError(
                f"Scheduling algorithm '{algo}' is not implemented"
            )


    def run_pd_algo(self) -> None:
        # update duration to be the longest
        for inst in self._insts:
            inst.duration = self.time_costs[type(inst)][inst.stage_id][0][0]
        # update E/L start/finish/slack
        self.annotate_nodes()
        # get a new critical path
        critical_path = self.get_critical_path()
        print("critical path: ", critical_path)
        pd_solver: PD_Solver = PD_Solver(self.entry_node, self.exit_node, self._insts)
        # Placeholder: do eager schedule for now
        for inst in self.insts:
            inst.actual_start = inst.earliest_start
            inst.actual_finish = inst.earliest_finish
