"""`PipelineVisualizer` draws a scheduled `InstructionDAG` with matplotlib."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes  # type: ignore
from matplotlib.ticker import FormatStrFormatter  # type: ignore

from rene.instruction import InstructionType, Forward, Backward
from rene.dag import InstructionDAG


DEFAULT_RECTANGLE_ARGS = {
    Forward: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    Backward: dict(facecolor="#9fc887", edgecolor="#000000", linewidth=1.0),
}

DEFAULT_ANNOTATION_ARGS = {
    Forward: dict(color="#ffffff", fontsize=20.0, ha="center", va="center"),
    Backward: dict(color="#000000", fontsize=20.0, ha="center", va="center"),
}

DEFAULT_LINE_ARGS = dict(color="#ff9900", linewidth=4.0)


class PipelineVisualizer:
    """Draws a scheduled `InstructionDAG` with matplotlib."""

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        dag: InstructionDAG,
        rectangle_args: dict[InstructionType, dict[str, Any]] = DEFAULT_RECTANGLE_ARGS,  # type: ignore
        annotation_args: dict[InstructionType, dict[str, Any]] = DEFAULT_ANNOTATION_ARGS,  # type: ignore
        line_args: dict[str, Any] = DEFAULT_LINE_ARGS,
    ) -> None:
        """Save the DAG and matplotilb arguments.

        Arguments:
            dag: The InstructionDAG. The instructions must be scheduled by calling `schedule`
            rectangle_args: Arguments passed to `matplotlib.patches.Rectangle` for instructions
            annotation_Args: Arguments passed to `matplotlib.axes.Axes.annotate` for the text
                inside instruction boxes
            line_args: Arguments passed to `matplitlib.axes.Axes.plot` for the critical path
        """
        if not dag.scheduled:
            raise ValueError("The DAG must be scheduled in order to be visualized.")

        self.dag = dag
        self.rectangle_args = rectangle_args
        self.annotation_args = annotation_args
        self.line_args = line_args

    def draw(self, ax: Axes, draw_time_axis: bool = False) -> None:
        """Draw the pipeline on the given Axes object."""
        for inst in self.dag.insts:
            # Draw rectangle for Instructions
            inst.draw(ax, self.rectangle_args, self.annotation_args)

        if draw_time_axis:
            ax.yaxis.set_visible(False)
            ax.grid(visible=False)

            total_time = self.dag.total_execution_time
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

    def draw_critical_path(self, ax: Axes) -> None:
        """Draw the critical path of the DAG on the given Axes object."""
        critical_path = self.dag.get_critical_path()
        for inst1, inst2 in zip(critical_path, critical_path[1:]):
            ax.plot(
                [
                    (inst1.actual_start + inst1.actual_finish) / 2,
                    (inst2.actual_start + inst2.actual_finish) / 2,
                ],
                [inst1.stage_id + 0.75, inst2.stage_id + 0.75],
                **self.line_args,
            )
