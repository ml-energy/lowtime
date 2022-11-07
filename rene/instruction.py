from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from matplotlib.axes import Axes

from matplotlib.patches import Rectangle


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

    def draw(
        self,
        ax: Axes,
        rectangle_args: dict[InstructionType, dict[str, Any]],
        annotation_args: dict[InstructionType, dict[str, Any]],
    ) -> None:
        """Draw the instruction on the Axes object.

        Override this method to change how instructions are drawn.
        """
        final_rectangle_args = dict(
            xy=(self.actual_start, self.stage_id),
            width=self.actual_duration,
            height=1.0,
        )
        final_rectangle_args.update(rectangle_args[type(self)])
        rectangle = Rectangle(**final_rectangle_args)
        ax.add_patch(rectangle)
        # Annotate the micro batch number inside the rectangle
        final_annotation_args = dict(
            text=str(self.micro_batch_id + 1),  # Draw with base index 1.
            xy=(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_y() + 0.5),  # type: ignore
        )
        final_annotation_args.update(annotation_args[type(self)])
        ax.annotate(**final_annotation_args)


class Forward(Instruction):
    """Forward computation for a pipeline stage."""
    pass


class Backward(Instruction):
    """Backward computation for a pipeline stage."""
    pass


class _Dummy(Instruction):
    """Dummy operation for entry and exit nodes in the DAG."""
    pass
