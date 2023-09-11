"""An instruction is an atomic an operation in pipeline training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Callable, Sequence, get_type_hints
from functools import cached_property
from dataclasses import dataclass, field

from attrs import define
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import Normalize  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from scipy.spatial import ConvexHull  # type: ignore

from rene.operation import Operation


@define(slots=False, kw_only=True)
class Instruction(Operation):
    """An operation on a pipeline training schedule."""

    stage_id: int
    micro_batch_id: int

    @cached_property
    def shorthand(self) -> str:
        """Return a shorthand representation of the instruction."""
        return f"{type(self).__name__}(S{self.stage_id}B{self.micro_batch_id})"


@define
class Forward(Instruction):
    """Forward computation for a pipeline stage."""


@define
class Backward(Instruction):
    """Backward computation for a pipeline stage."""


@define
class ForwardBackward(Instruction):
    """ForwardBackward computation for a pipeline stage."""


@define
class Recomputation(Instruction):
    """Activation recomputation (forward) for a pipeline stage."""


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


def forwardbackward_dep(inst1: ForwardBackward, inst2: ForwardBackward) -> bool:
    """Dependency rule between ForwardBackward instructions.

    ForwardBackward(stage i+1, microbatch j) -> ForwardBackward(stage i, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id == inst2.stage_id + 1
    )


def forwardbackward_backward_dep(inst1: ForwardBackward, inst2: Backward) -> bool:
    """Dependency rule between ForwardBackward and Backward.

    ForwardBackward(stage i+1, microbatch j) -> Backward(stage i, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id == inst2.stage_id + 1
    )


@dataclass(repr=False)
class InstructionOld:
    """A chunk of operation in one pipeline stage.

    Attributes:
        stage_id: Zero-indexed pipeline stage
        micro_batch_id: Zero-indexed micro batch number
        duration: Duration of this instruction
        unit_cost: Projected unit energy cost when changing a single unit of duration
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
    unit_cost: float = 0.0
    max_duration: float = 0.0
    min_duration: float = 0.0
    repr: str = ""
    alias: ClassVar[str] = ""

    # Values set by critical path analysis (in `ReneDAG.annotate_nodes`)
    earliest_start: float = 0.0
    latest_start: float = float("inf")
    earliest_finish: float = 0.0
    latest_finish: float = float("inf")

    # Values set by `ReneDAG.schedule`
    actual_start: float = 0.0
    actual_finish: float = 0.0

    # For time-cost Pareto frontier model fitting
    fit_method: str = "linear"
    fit_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    initial_guess: list[float] = field(default_factory=list)

    # For frequency assignment
    time_costs: list[tuple[float, float, int]] = field(default_factory=list)
    frequency: int = -1
    cost: float = -1.0  # This is decoupled energy cost, not computation energy cost.

    # For decoupled energy
    num_stages: int = 0
    p2p_power: float = 0.0

    # For time rescaling
    unit_time: float = 0.0

    output_dir: Path | None = None

    def __repr__(self) -> str:
        """Return a concise representation of the Instruction.

        This method is called many times, so we cache the string in `self.repr`
        when we construct it since it's a constant.
        """
        if self.repr:
            return self.repr
        name = self.alias or type(self).__name__
        self.repr = f"{name}(S{self.stage_id}B{self.micro_batch_id})"
        return self.repr

    @property
    def slack(self) -> float:
        """Return the slack of the Instruction."""
        return self.latest_finish - self.earliest_finish

    @property
    def actual_duration(self) -> float:
        """Return the execution duration of the Instruction."""
        return self.actual_finish - self.actual_start

    def draw(
        self,
        ax: Axes,
        rectangle_args: dict[InstructionType, dict[str, Any]],
        annotation_args: dict[InstructionType, dict[str, Any]],
        annotation_hook: Callable[[Instruction], str] | None = None,
        power_color: str | None = "Oranges",
        normalizer: Normalize = Normalize(vmin=0, vmax=400),
    ) -> None:
        """Draw the instruction on the Axes object.

        Override this method to change how instructions are drawn.

        Arguments:
            ax: Axes object to draw on
            rectangle_args: Arguments to pass to `Rectangle`
            annotation_args: Arguments to pass to `ax.annotate`
            annotation_hook: If not None, this is called with the instruction and the string
                returned will be annotated inside the instruction box.
            power_color: Color map to use for power coloring (default: "Oranges")
            normalizer: Normalizer to use for power coloring (default: Normalize(vmin=0, vmax=400))
        """
        annotation = (
            annotation_hook(self)
            if annotation_hook is not None
            else str(self.frequency)
        )

        final_rectangle_args = dict(
            xy=(self.actual_start * self.unit_time, self.stage_id),
            width=self.actual_duration * self.unit_time,
            height=1.0,
        )
        final_rectangle_args.update(rectangle_args[type(self)])
        if power_color is not None:
            # HACK: We want the original computation cost here.
            cost = (
                self.cost
                if self.cost != -1.0
                else (
                    self.get_cost(self.duration)
                    + self.p2p_power * self.duration * self.unit_time
                )
            )
            final_rectangle_args["facecolor"] = plt.get_cmap(power_color)(
                normalizer(cost / (self.duration * self.unit_time))
            )
        rectangle = Rectangle(**final_rectangle_args)
        ax.add_patch(rectangle)
        # Annotate the frequency inside the rectangle
        final_annotation_args = dict(
            text=annotation,
            # text=str(self.micro_batch_id),
            xy=(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_y() + 0.5),  # type: ignore
        )
        final_annotation_args.update(annotation_args[type(self)])
        ax.annotate(**final_annotation_args)

    # ruff: noqa: PLR0912
    def fit(self, fit_method: str) -> np.ndarray:
        """Do interpolation on the given instruction and its time-costs meta-data, return the coefficients.

        Arguments:
            fit_method: Fit method to use, currently supports "linear", "piecewise-linear" and "exponential"
        """
        # Get the slope from two endpoints
        self.fit_method = fit_method
        # Sort the time-costs meta-data by reverse time duration
        self.time_costs.sort(key=lambda x: x[0], reverse=True)
        time_list = []
        cost_list = []
        freq_list = []
        for t, e, _f in self.time_costs:
            time_list.append(t)
            cost_list.append(e - self.p2p_power * t * self.unit_time)
            freq_list.append(_f)

        if fit_method == "linear":
            # Linear interpolation
            self.fit_coeffs = np.polyfit(time_list, cost_list, 1)

        elif fit_method == "piecewise-linear":
            # Piecewise linear interpolation
            data = np.array(
                list(zip(time_list, cost_list)),  # noqa
                dtype=[("time", float), ("cost", float)],
            )
            data = data[data.argsort(order=["time", "cost"])]
            # Add a second axis to the array
            data = data.view((float, 2))
            # Flip the y-coordinates
            data[:, 1] = -data[:, 1]

            # Compute the convex hull
            hull = ConvexHull(data)

            # Restore the original y-coordinates
            # Points are guaranteed to be in counter-clockwise order.
            convex_points = data[hull.vertices]
            convex_points[:, 1] = -convex_points[:, 1]
            # Roll convex_points until the first point's x coordinate is the smallest on the convex hull
            convex_points = np.roll(
                convex_points, -np.argmin(convex_points[:, 0]), axis=0
            )
            # Scan points on the convex hull from the beginning, and when the x coordinate increases, remove everything
            # after that point. This is because the convex hull is not necessarily a piecewise linear function,
            # and we want to make it one.
            for i in range(len(convex_points) - 1):
                if convex_points[i, 0] > convex_points[i + 1, 0]:
                    convex_points = np.delete(convex_points, np.s_[1:i], axis=0)
                    break
            # Sort the convex_points by their x-coordinate in ascending order
            convex_points = convex_points[convex_points[:, 0].argsort()]
            self.fit_coeffs = convex_points
            # Now, convex_points contains the points that form a convex piecewise linear function
        elif fit_method == "exponential":
            # if initial guess is not an empty list
            if self.initial_guess:
                self.fit_coeffs, _ = curve_fit(
                    lambda t, a, b, c: a * np.exp(b * t) + c,
                    time_list,
                    cost_list,
                    p0=self.initial_guess,
                    maxfev=50000,
                )
                logging.info("%s Initial guess: %s", repr(self), self.initial_guess)
            else:
                self.fit_coeffs, _ = curve_fit(
                    lambda t, a, b, c: a * np.exp(b * t) + c,
                    time_list,
                    cost_list,
                    maxfev=50000,
                )
                logging.info("%s No initial guess", repr(self))

            logging.info("%s Fit coeffs: %s", repr(self), self.fit_coeffs)
        else:
            raise NotImplementedError(f"Unknown fit method: {fit_method}")

        if self.output_dir is not None:
            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            ax.plot(time_list, cost_list, "o")
            for i in range(len(time_list)):
                ax.annotate(
                    f"({time_list[i]:.6f}, {cost_list[i]:.6f}, {freq_list[i]})",
                    (time_list[i], cost_list[i]),
                )
            # generate a list with step size 0.1
            x = np.arange(min(time_list), max(time_list), 0.0001)
            y = []
            for i in x:
                y.append(self.get_cost(i))
            ax.plot(x, y, "r-")
            ax.set_xlabel("time")
            ax.set_ylabel("energy")
            fig.savefig(self.output_dir / f"{repr(self)}.png")
            plt.clf()
            plt.close()

        return self.fit_coeffs

    def get_cost(self, time: float) -> float:
        """Get the cost of the instruction at the given time.

        XXX: This returns the refined cost (whatever that was fit by the model), not the original cost.

        Arguments:
            time: Time to get the cost at
        """
        if len(self.fit_coeffs) == 0:
            raise ValueError("No fit coefficients have been computed yet")
        if self.fit_method == "linear":
            return np.polyval(self.fit_coeffs, time).item()
        elif self.fit_method == "piecewise-linear":
            return self.binary_search_piecewise_linear(time)
        elif self.fit_method == "exponential":
            a, b, c = self.fit_coeffs
            return a * np.exp(b * time) + c
        else:
            raise ValueError(f"Unknown fit method {self.fit_method}")

    def binary_search_piecewise_linear(self, time: float) -> float:
        """Perform a binary search on the piecewise linear function to find the cost at the given time.

        Arguments:
            time: Time to search for

        Returns:
            Cost of the instruction at the given time
        """
        if time < self.fit_coeffs[0, 0]:
            # TODO: return inf here?
            return self.fit_coeffs[0, 1]
        elif time > self.fit_coeffs[-1, 0]:
            # TODO: return 0 here?
            return self.fit_coeffs[-1, 1]
        # do a binary search for time in the correct interval of the first axis of self.fit_coeffs
        low = 0
        high = self.fit_coeffs.shape[0] - 1
        while low <= high:
            mid = (low + high) // 2
            # exact match found
            if abs(self.fit_coeffs[mid][0] - time) < 1e-6:  # noqa
                return self.fit_coeffs[mid][1]
            elif self.fit_coeffs[mid][0] < time:
                low = mid + 1
            else:
                high = mid - 1
        # if no exact match is found, return the closest x value
        if high < 0:
            return self.fit_coeffs[low][1]
        elif low >= self.fit_coeffs.shape[0]:
            return self.fit_coeffs[high][1]
        else:
            x1, y1 = self.fit_coeffs[high]
            x2, y2 = self.fit_coeffs[low]
            assert low == high + 1
            if x1 <= time <= x2:
                t = (time - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        raise ValueError(f"time = {time} is out of the range of the breakpoints")

    def get_derivative(self, time_left: float, time_right: float) -> float:
        """Get the derivative/slope between two time points time_left and time_right.

        Arguments:
            time_left: Start time points to get the derivative at
            time_right: End time points to get the derivative at

        Returns:
            The derivative between the two time points
        """
        return abs(
            (self.get_cost(time_left) - self.get_cost(time_right))
            / (time_left - time_right)
        )


# class Forward(Instruction):
#     """Forward computation for a pipeline stage."""

#     alias = "FW"


# class Backward(Instruction):
#     """Backward computation for a pipeline stage."""

#     alias = "BW"


# class ForwardBackward(Instruction):
#     """ForwardBackward computation for a pipeline stage."""

#     alias = "FB"


# class Recomputation(Instruction):
#     """Activation recomputation (forward) for a pipeline stage."""

#     alias = "RC"


# class _Dummy(Instruction):
#     """Dummy operation for entry and exit nodes in the DAG."""