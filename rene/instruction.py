from __future__ import annotations

from dataclasses import dataclass, field


class InstructionType(type):
    """Instruction metaclass.
    
    Metaclass for typing and subclass name collection.
    """

    # Names of Instruction subclasses.
    subclass_names: set[str] = set()

    def __new__(cls, name, bases, dct):
        if name in cls.subclass_names:
            raise ValueError(f"Instruction class '{name}' already exists")
        if name != "_Dummy":
            cls.subclass_names.add(name)
        return super().__new__(cls, name, bases, dct)


@dataclass(repr=False)
class Instruction(metaclass=InstructionType):
    """A chunk of operation in one pipeline stage.

    Attributes:
        stage_id: Zero-indexed pipeline stage
        micro_batch_id: Zero-indexed micro batch number
        duration: Duration of this instruction
        parents: Instructions that this instruction depends on
        children: Instructions that depend on this instruction
        earliest_start: The earliest time this instruction can start
        latest_start: The latest time this instruction can start
        earliest_finish: The earliest time this instruction can finish
        latest_finish: The latest time this instruction can finish
        actual_start: The actual start time determined by the scheduling algorithm
        actual_finish: The actual finish time determined by the scheduling algorithm
    """

    stage_id: int
    micro_batch_id: int
    duration: float = 0.0

    # DAG metadata
    parents: list[Instruction] = field(default_factory=list)
    children: list[Instruction] = field(default_factory=list)

    # Values set by critical path analysis (in `InstructionDAG.__init__`)
    earliest_start: float = 0.0
    latest_start: float = float("inf")
    earliest_finish: float = 0.0
    latest_finish: float = float("inf") 

    # Values set by `InstructionDAG.schedule`
    actual_start: float = 0.0
    actual_finish: float = 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(S{self.stage_id}B{self.micro_batch_id})"

    @property
    def actual_duration(self) -> float:
        return self.actual_finish - self.actual_start

    def then(self, other: Instruction) -> None:
        """Declare that `other` depends on the completion of this instruction."""
        self.children.append(other)
        other.parents.append(self)


class Forward(Instruction):
    """Forward computation for a pipeline stage."""
    pass


class Backward(Instruction):
    """Backward computation for a pipeline stage."""
    pass


class _Dummy(Instruction):
    """Dummy operation for entry and exit nodes in the DAG."""
    pass
