"""Defines the DAG of instructions and algorithms to define their execution times."""

from __future__ import annotations

import inspect
import itertools
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


class InstructionDAG:
    """DAG of instructions and analysis methods."""

    # pylint: disable=dangerous-default-value,too-many-branches
    def __init__(
        self,
        schedule_type: Callable[[int, int, int], Iterable[Instruction]],
        num_stages: int,
        num_micro_batches: int,
        durations: dict[Type[Instruction], list[float]] | None,
        dependency_rules: Sequence[Callable[..., bool]] = [forward_dep, backward_dep],
    ) -> None:
        """Instantiate instructions, connect the DAG, and run critical path analysis.

        Arguments:
            schedule_type: A callable that returns an iterable of pipeline instructions.
                Can also be a subclass of `rene.schedule.PipeSchedule` like `Synchronous1F1B`.
            num_stages: The number of pipeline stages.
            num_micro_batches: The number of micro batches in the pipeline.
            durations: A dict that maps instruction classes to a list of durations for each stage.
                `None` is aceptable if `schedule_type` will produce `Instruction`s that already
                know their durations.
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
        self.durations = durations
        self.dependency_rules = dependency_rules

        self.scheduled = False

        # Sanity check.
        if self.durations is not None:
            for latencies in self.durations.values():
                if len(latencies) != num_stages:
                    raise ValueError(
                        "`len(durations[instruction_type])` should be the same as `num_stages`"
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
                if self.durations is not None:
                    inst.duration = self.durations[type(inst)][inst.stage_id]
                self._insts.append(inst)
                if prev_inst is not None:
                    prev_inst.then(inst)
                prev_inst = inst
            prev_inst = None

        # Define dependencies by the dependency rules passed in.
        for inst1, inst2 in itertools.product(self._insts, self._insts):
            if self._is_dependent(inst1, inst2):
                inst1.then(inst2)

        # Introduce dummy entry and exit nodes for analysis convenience.
        self.entry_node = _Dummy(-1, -1, duration=0.0)
        self.exit_node = _Dummy(-1, -1, duration=0.0)
        for node in self._insts:
            if not node.parents:
                self.entry_node.then(node)
            if not node.children:
                node.then(self.exit_node)

        # Annotate earliest/latest start/finish times in nodes.
        # Forward computation: Assign earliest start and finish times
        self.entry_node.earliest_start = 0.0
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
                q.put(child)

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
                if abs(child.earliest_start - child.latest_start) < 1e-10:
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
