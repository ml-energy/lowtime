"""An instruction is an atomic an operation in pipeline training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import logging
import os
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from scipy.spatial import ConvexHull  # type: ignore


class InstructionType(type):
    """Instruction metaclass.

    Metaclass for typing and subclass name collection.
    """

    # Names of Instruction subclasses.
    subclass_names: set[str] = set()

    def __new__(
        cls: InstructionType, name: str, bases: tuple[type, ...], dct: dict[str, Any]
    ) -> type:
        """Collect the names of all `Instruction` classes defined."""
        if name in cls.subclass_names:
            raise ValueError(f"Instruction class '{name}' already exists")
        if name != "_Dummy":
            cls.subclass_names.add(name)
        return super().__new__(cls, name, bases, dct)


@dataclass(repr=False)
class Instruction(metaclass=InstructionType):
    """A chunk of operation in one pipeline stage.

    Attributes:
        stage_id: {int} -- Zero-indexed pipeline stage
        micro_batch_id: {int} -- Zero-indexed micro batch number
        duration: {float} -- Duration of this instruction
        unit_cost: {float} -- Projected unit energy cost when changing a single unit of duration
        earliest_start: {float} -- The earliest time this instruction can start
        latest_start: {float} -- The latest time this instruction can start
        earliest_finish: {float} -- The earliest time this instruction can finish
        latest_finish: {float} -- The latest time this instruction can finish
        slack: {float} -- The max delay for this instruction without delaying latest_finish
        actual_start: {float} -- The actual start time determined by the scheduling algorithm
        actual_finish: {float} -- The actual finish time determined by the scheduling algorithm
    """

    stage_id: int
    micro_batch_id: int
    duration: float = 0.0
    unit_cost: float = 0.0
    max_duration: float = 0.0
    min_duration: float = 0.0
    repr: str = ""

    # Values set by critical path analysis (in `ReneDAG.__init__`)
    earliest_start: float = 0.0
    latest_start: float = float("inf")
    earliest_finish: float = 0.0
    latest_finish: float = float("inf")
    slack: float = 0.0

    # Values set by `ReneDAG.schedule`
    actual_start: float = 0.0
    actual_finish: float = 0.0

    # For time-cost Pareto frontier model fitting
    fit_method: str = "linear"
    fit_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    initial_guess: list[float] = field(default_factory=list)

    # For frequency assignment
    time_costs: list[tuple] = field(default_factory=list)
    frequency: int = -1
    cost: float = -1.0

    # For P2P blocking cost reduction
    num_stages: int = 0
    p2p_power: float = 0.0
    on_critical_path: bool = False

    output_dir: str = ""

    def __repr__(self) -> str:
        """Return a concise representation of the Instruction."""
        return self.repr or f"{type(self).__name__}(S{self.stage_id}B{self.micro_batch_id})"

    @property
    def actual_duration(self) -> float:
        """Return the execution duration of the Instruction."""
        return self.actual_finish - self.actual_start

    def draw(
        self,
        ax: Axes,
        rectangle_args: dict[InstructionType, dict[str, Any]],
        annotation_args: dict[InstructionType, dict[str, Any]],
        power_color: str | None = "Oranges",
    ) -> None:
        """Draw the instruction on the Axes object.

        Override this method to change how instructions are drawn.

        Arguments:
            ax: {Axes} -- Axes object to draw on
            rectangle_args: {dict[InstructionType, dict[str, Any]]} -- Arguments to pass to `Rectangle`
            annotation_args: {dict[InstructionType, dict[str, Any]]} -- Arguments to pass to `ax.annotate`
            power_color: {str | None} -- Color map to use for power coloring (default: {"Oranges"})
        """
        final_rectangle_args = dict(
            xy=(self.actual_start, self.stage_id),
            width=self.actual_duration,
            height=1.0,
        )
        final_rectangle_args.update(rectangle_args[type(self)])
        if power_color is not None:
            final_rectangle_args["facecolor"] = plt.get_cmap(power_color)(
                self.cost / self.duration / 400.0
            )
        rectangle = Rectangle(**final_rectangle_args)
        ax.add_patch(rectangle)
        # Annotate the micro batch number inside the rectangle
        final_annotation_args = dict(
            text=str(self.frequency),
            xy=(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_y() + 0.5),  # type: ignore
        )
        final_annotation_args.update(annotation_args[type(self)])
        ax.annotate(**final_annotation_args)

    def interpolate(self, fit_method: str) -> np.ndarray:
        """Do interpolation on the given instruction and its time-costs meta-data, return the coefficients.

        Assumes self.time_costs[inst] has already been sorted.

        Arguments:
            fit_method: {str} -- Fit method to use, currently supports "linear", "piecewise-linear" and "exponential"
        """
        # Get the slope from two endpoints
        self.fit_method = fit_method
        # Sort the time-costs meta-data by reverse time duration
        self.time_costs.sort(key=lambda x: x[0], reverse=True)
        time_list = []
        cost_list = []
        # cost_list_unrefined = []
        for t, e, _f in self.time_costs:
            time_list.append(t)
            # cost_list_unrefined.append(e)
            cost_list.append(e - self.p2p_power * t)

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
            # poly_coeffs = np.polyfit(time_list, np.log(cost_list), 1, w=np.sqrt(cost_list))
            # unrefined_poly_coeffs = np.polyfit(time_list, np.log(cost_list_unrefined), 1, w=np.sqrt(cost_list_unrefined))

            # if initial guess is not an empty list
            if self.initial_guess:
                self.fit_coeffs, _ = curve_fit(
                    lambda t, a, b, c: a * np.exp(b * t) + c,
                    time_list,
                    cost_list,
                    p0=self.initial_guess,
                    maxfev=10000,
                )
                logging.info(f"{repr(self)} Initial guess: {self.initial_guess}")
            else:
                self.fit_coeffs, _ = curve_fit(
                    lambda t, a, b, c: a * np.exp(b * t) + c,
                    time_list,
                    cost_list,
                    maxfev=10000,
                )
                logging.info(f"{repr(self)} No initial guess")
            # p0_unrefined = [cost_list_unrefined[-1] - cost_list_unrefined[0], -np.log(cost_list_unrefined[0] / cost_list_unrefined[-1]) / (time_list[0] - time_list[-1]), cost_list_unrefined[0]]
            # self.unrefined_fit_coeffs, _ = curve_fit(
            #     lambda t, a, b, c: a * np.exp(b * t) + c,
            #     time_list,
            #     cost_list_unrefined,
            #     p0=p0,
            #     maxfev=10000,
            # )
            logging.info(f"{repr(self)} Fit coeffs: {self.fit_coeffs}")
            # logging.info(f"{repr(self)} Unrefined fit coeffs: {self.unrefined_fit_coeffs}")
            # self.fit_coeffs = np.array([np.exp(poly_coeffs[1]), poly_coeffs[0]])
            # self.unrefined_fit_coeffs = np.array([np.exp(unrefined_poly_coeffs[1]), unrefined_poly_coeffs[0]])
        else:
            raise NotImplementedError(f"Unknown fit method: {fit_method}")

        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
        ax.plot(time_list, cost_list, "o")
        # ax.plot(time_list, cost_list_unrefined, "x")
        # generate a list with step size 0.1
        x = np.arange(min(time_list), max(time_list), 0.0001)
        # ax.plot(x, np.polyval(self.fit_coeffs, x), 'r-')
        y = []
        # unrefined_y = []
        for i in x:
            y.append(self.get_cost(i))
            # unrefined_y.append(self.unrefined_fit_coeffs[0] * np.exp(self.unrefined_fit_coeffs[1] * i) + self.unrefined_fit_coeffs[2])
        # a, b, c = unrefined_fit_coeffs
        # unrefined_y = a * np.exp(b * x) + c

        # refined_y = []
        # for i in x:
        #     refined_y.append(self.get_p2p_refined_cost(i))
        ax.plot(x, y, "r-")
        # ax.plot(x, unrefined_y, "b-")
        if fit_method == "piecewise-linear":
            for x, y in convex_points:
                ax.annotate(f"({x:.6f}, {y:.6f})", (x, y))
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        fig.savefig(
            os.path.join(self.output_dir, f"{repr(self)}_HULL.png"), format="PNG"
        )
        plt.clf()
        plt.close()

        return self.fit_coeffs

    def get_cost(self, time: float) -> float:
        """Get the cost of the instruction at the given time.

        Arguments:
            time: {float} -- Time to get the cost at
        """
        if len(self.fit_coeffs) == 0:
            raise ValueError("No fit coefficients have been computed yet")
        if self.fit_method == "linear":
            return np.polyval(self.fit_coeffs, time)
        elif self.fit_method == "piecewise-linear":
            return self.binary_search_piecewise_linear(time)
        elif self.fit_method == "exponential":
            a, b, c = self.fit_coeffs
            # a, b = self.fit_coeffs
            return a * np.exp(b * time) + c
        else:
            raise ValueError(f"Unknown fit method {self.fit_method}")

    def binary_search_piecewise_linear(self, time: float) -> int:
        """Perform a binary search on the piecewise linear function to find the cost at the given time.

        Arguments:
            time: {float} -- Time to search for

        Returns:
            {int} -- Cost of the instruction at the given time
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

    def get_p2p_refined_cost(self, time: float) -> float:
        """Get the cost of the instruction at the given time using p2p refinement.

        Arguments:
            time: {float} -- Time to get the cost at
        """
        cost = self.get_cost(time)
        refined_cost = cost - self.p2p_power * time
        assert refined_cost >= 0
        return refined_cost

    def get_derivative(self, time_left: float, time_right: float) -> float:
        """Get the derivative/slope between two time points time_left and time_right.

        Arguments:
            time_left: {float} -- Start time points to get the derivative at
            time_right: {float} -- End time points to get the derivative at

        Returns:
            {float} -- The derivative between the two time points
        """
        return abs(
            (
                self.get_cost(time_left)
                - self.get_cost(time_right)
            )
            / (time_left - time_right)
        )


class Forward(Instruction):
    """Forward computation for a pipeline stage."""

    def __repr__(self) -> str:
        """Return a concise representation of the Instruction."""
        if not self.repr:
            return f"FW(S{self.stage_id}B{self.micro_batch_id})"
        else:
            return self.repr


class Backward(Instruction):
    """Backward computation for a pipeline stage."""

    def __repr__(self) -> str:
        """Return a concise representation of the Instruction."""
        if not self.repr:
            return f"BW(S{self.stage_id}B{self.micro_batch_id})"
        else:
            return self.repr


class Recomputation(Instruction):
    """Activation recomputation (forward) for a pipeline stage."""

    def __repr__(self) -> str:
        """Return a concise representation of the Instruction."""
        if not self.repr:
            return f"RC(S{self.stage_id}B{self.micro_batch_id})"
        else:
            return self.repr


class _Dummy(Instruction):
    """Dummy operation for entry and exit nodes in the DAG."""
