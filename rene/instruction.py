"""An instruction is an atomic an operation in pipeline training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import os
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


class InstructionType(type):
    """Instruction metaclass.

    Metaclass for typing and subclass name collection.
    """

    # Names of Instruction subclasses.
    subclass_names: set[str] = set()

    def __new__(cls, name, bases, dct):
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
        stage_id: Zero-indexed pipeline stage
        micro_batch_id: Zero-indexed micro batch number
        duration: Duration of this instruction
        unit_cost: Projected unit energy cost when changing a single unit of duration
        parents: Instructions that this instruction depends on
        children: Instructions that depend on this instruction
        earliest_start: The earliest time this instruction can start
        latest_start: The latest time this instruction can start
        earliest_finish: The earliest time this instruction can finish
        latest_finish: The latest time this instruction can finish
        slack: The max delay for this instruction without delaying latest_finish
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

    # DAG metadata
    parents: list[Instruction] = field(default_factory=list)
    children: list[Instruction] = field(default_factory=list)

    # Values set by critical path analysis (in `ReneDAG.__init__`)
    earliest_start: float = 0.0
    latest_start: float = float("inf")
    earliest_finish: float = 0.0
    latest_finish: float = float("inf")
    slack: float = 0.0

    # Values set by `ReneDAG.schedule`
    actual_start: float = 0.0
    actual_finish: float = 0.0

    # For poly fit
    fit_degree: int = 0
    fit_coeffs: list[float] = field(default_factory=list)

    # For frequency assignment
    time_costs: list[tuple] = field(default_factory=list) 
    frequency: int = -1
    cost: float = -1.0

    output_dir: str = ""

    def __repr__(self) -> str:
        """Return a concise representation of the Instruction."""
        if self.repr == "":
            return f"{type(self).__name__}(S{self.stage_id}B{self.micro_batch_id})"
        else:
            return self.repr

    @property
    def actual_duration(self) -> float:
        """Return the execution duration of the Instruction."""
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
        power_color: str | None = "Oranges",
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
        if power_color is not None:
            final_rectangle_args["facecolor"] = plt.get_cmap(power_color)(self.cost / self.duration / 400.0)
        rectangle = Rectangle(**final_rectangle_args)
        ax.add_patch(rectangle)
        # Annotate the micro batch number inside the rectangle
        final_annotation_args = dict(
            text=str(self.micro_batch_id),
            xy=(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_y() + 0.5),  # type: ignore
        )
        final_annotation_args.update(annotation_args[type(self)])
        ax.annotate(**final_annotation_args)

    def interpolate(self, fit_degree: int) -> (float, float):
        """Do interpolation on the given instruction and its time-costs meta-data, return the coefficients
        Assumes self.time_costs[inst] has already been sorted.

        Arguments:
            fit_degree: Degree of the polynomial to fit
        """
        # Get the slope from two endpoints
        # right_end = self.time_costs[type(inst)][inst.stage_id][0]
        # left_end = self.time_costs[type(inst)][inst.stage_id][-1]
        # unit_cost = abs((right_end[1] - left_end[1]) / (right_end[0] - left_end[0]))
        self.fit_degree = fit_degree
        time_list = []
        cost_list = []
        for t, e, f in self.time_costs:
            time_list.append(t)
            cost_list.append(e)

        # Your dataset of (x, y) pairs
        time_data = time_list
        cost_data = cost_list

        # Combine the time and cost data into an array of (x, y) pairs
        data = np.column_stack((time_data, cost_data))

        # Sort the points by their x-coordinate in ascending order
        data = data[data[:, 0].argsort()]

        # Flip the y-coordinates
        data[:, 1] = -data[:, 1]

        # Compute the convex hull
        hull = ConvexHull(data)

        # Restore the original y-coordinates
        convex_points = data[hull.vertices]
        convex_points[:, 1] = -convex_points[:, 1]
        # Roll convex_points until the first point's x coordinate is the smallest on the convex hull
        convex_points = np.roll(convex_points, -np.argmin(convex_points[:, 0]), axis=0)
        print(repr(self), convex_points)
        # Scan points on the convex hull from the beginning, and when the x coordinate increases, remove everything after that
        # point. This is because the convex hull is not necessarily a piecewise linear function, and we want to make it one.
        for i in range(len(convex_points) - 1):
            if convex_points[i, 0] >= convex_points[i + 1, 0]:
                convex_points = np.delete(convex_points, np.s_[1:i], axis=0)
                break
        # Sort the convex_points by their x-coordinate in ascending order
        convex_points = convex_points[convex_points[:, 0].argsort()]
        # Now, convex_points contains the points that form a convex piecewise linear function
        def piecewise_linear(x, convex_points):
            if x < convex_points[0, 0]:
                return convex_points[0, 1]
            elif x > convex_points[-1, 0]:
                return convex_points[-1, 1]

            for i in range(len(convex_points) - 1):
                x1, y1 = convex_points[i]
                x2, y2 = convex_points[i + 1]

                if x1 <= x <= x2:
                    t = (x - x1) / (x2 - x1)
                    return y1 + t * (y2 - y1)

            raise ValueError(f"x = {x} is out of the range of the convex_points")

        # Example usage:
        x_value = 5
        y_value = piecewise_linear(x_value, convex_points)
        # print(f"y_value: {y_value}")

        # # Degree of the polynomial
        # degree = 2

        # # Define the objective function to minimize (sum of squared errors)
        # def objective_function(coeffs):
        #     poly = np.poly1d(coeffs)
        #     y_pred = poly(x_data)
        #     return np.sum((y_data - y_pred)**2)

        # # Define the constraint (P'(x) ≤ 0)
        # def constraint(coeffs):
        #     poly = np.poly1d(coeffs)
        #     poly_derivative = np.polyder(poly)
        #     slopes = poly_derivative(x_data)
        #     return np.min(-slopes)

        # # Set the constraint in a dictionary
        # con = {"type": "ineq", "fun": constraint}

        # # Initial guess for the polynomial coefficients
        # initial_guess = np.ones(degree + 1)

        # # Solve the constrained optimization problem
        # result = minimize(objective_function, initial_guess, constraints=[con])

        # # Get the coefficients of the best-fit polynomial
        # self.fit_coeffs = result.x
        # best_fit_poly = np.poly1d(best_fit_coeffs)

        # self.fit_coeffs = np.polyfit(time_list, cost_list, self.fit_degree)
        # plot the fit polynomial (self.fit_coeffs) as continuous curve
        # plt.plot(time_list, cost_list, 'o')
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
        ax.plot(time_list, cost_list, 'o')
        # generate a list with step size 0.1
        x = np.arange(min(time_list), max(time_list), 0.0001)
        # ax.plot(x, np.polyval(self.fit_coeffs, x), 'r-')
        y = []
        for i in x:
            y.append(piecewise_linear(i, convex_points))
        ax.plot(x, y, 'r-')
        for x, y in convex_points:
            ax.annotate(f"({x:.6f}, {y:.6f})", (x, y))
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        fig.savefig(os.path.join(self.output_dir, f"{self.__repr__()}_HULL.png"), format="PNG")
        plt.clf()
        plt.close()

        return self.fit_coeffs
    
    def get_cost(self, time: float) -> float:
        """Get the cost of the instruction at the given time.

        Arguments:
            time: Time to get the cost at
        """
        return np.polyval(self.fit_coeffs, time)
    
    def get_derivative(self, time: float) -> float:
        """Get the derivative of the instruction at the given time.

        Arguments:
            time: Time to get the derivative at
        """
        return np.polyval(np.polyder(self.fit_coeffs), time)

class Forward(Instruction):
    """Forward computation for a pipeline stage."""


class Backward(Instruction):
    """Backward computation for a pipeline stage."""


class _Dummy(Instruction):
    """Dummy operation for entry and exit nodes in the DAG."""
