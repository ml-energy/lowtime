from __future__ import annotations

import bisect
import itertools
import logging
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field
from attrs.setters import frozen
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

KnobT = TypeVar("KnobT")


@define(repr=False, slots=False)
class ExecutionOption(Generic[KnobT]):
    """One option for executing an operation.

    An operation can be executed by with one among multiple possible knob,
    and the execution expenditure of choosing this particular `knob` value is
    represented by this class.

    Attributes:
        real_time: The wall clock time it took to execute the operation.
        unit_time: The time unit used to quantize `real_time`.
        quant_time: Integer-quantized time (`int(real_time // unit_time)`).
        cost: The cost of the operation.
        knob: The knob value associated with this option.
    """

    real_time: float
    unit_time: float
    cost: float
    knob: KnobT

    @cached_property
    def quant_time(self) -> int:
        return int(self.real_time // self.unit_time)

    def __repr__(self) -> str:
        return (
            f"ExecutionOption(knob={self.knob}, quant_time={self.quant_time}, "
            f"cost={self.cost}, real_time={self.real_time}, unit_time={self.unit_time})"
        )


@define
class CandidateExecutionOptions(Generic[KnobT]):
    """A list of selected candidate execution options for an operation.

    Candidate execution options are filtered from execution options given to __init__:
    1. Filter Pareto-optimal options based on `real_time` and `cost`.
    2. Deduplicate `quant_time` by keeping only the option with the largest `cost`.
        This is because time quantization is inherently rounding down, so the closest
        `quant_time` is the one with the largest `cost`.
    3. Sort by `quant_time` in ascending order.

    Args:
        options: All candidate execution options of the operation.
    """

    options: list[ExecutionOption[KnobT]]

    def __attrs_post_init__(self) -> None:
        """Return a new `ExecutionOptions` object with only Pareto-optimal options."""
        # Find and only leave Pareto-optimal options.
        orig_options = sorted(self.options, key=lambda x: x.real_time, reverse=True)
        filtered_options: list[ExecutionOption[KnobT]] = []
        for option in orig_options:
            if any(
                other.real_time < option.real_time and other.cost < option.cost
                for other in orig_options
            ):
                continue
            filtered_options.append(option)

        # There may be multiple options with the same `quant_time`.
        # Only leave the option with the largest `cost` because that's the one whose
        # `quant_time` is closest to `real_time`.
        filtered_options.sort(key=lambda x: x.cost, reverse=True)
        orig_options, filtered_options = filtered_options, []
        quant_times = set()
        for option in orig_options:
            if option.quant_time in quant_times:
                continue
            filtered_options.append(option)
            quant_times.add(option.quant_time)

        # Sort by `quant_time` in descending order.
        self.options = filtered_options
        self.options.sort(key=lambda x: x.quant_time)


class CostModel(ABC):
    """A continuous cost model fit from Pareto-optimal execution options."""

    @abstractmethod
    def __call__(self, quant_time: int) -> float:
        """Predict execution cost given quantized time."""
        ...

    def draw(
        self,
        ax: plt.Axes,
        options: CandidateExecutionOptions,
    ) -> None:
        """Plot a cost model's predictions with its target costs and save to path.

        Args:
            ax: Matplotlib axes to draw on.
            options: `quant_time` is taken as the x axis and `cost` will be drawn
                separately as the target (ground truth) cost plot.
        """
        quant_times = [option.quant_time for option in options.options]
        target_costs = [option.cost for option in options.options]

        # Plot the ground truth costs.
        ax.plot(quant_times, target_costs, "o")
        for option in options.options:
            ax.annotate(
                f"({option.quant_time}, {option.cost}, {option.knob})",
                (option.quant_time, option.cost),
            )

        # Plot the cost model's predictions.
        xs = np.arange(min(quant_times), max(quant_times), 0.01)
        ys = [self(x) for x in xs]
        ax.plot(xs, ys, "r-")

        # Plot metadata.
        ax.set_xlabel("quant_time")
        ax.set_ylabel("cost")


class ExponentialModel(CostModel):
    """An exponential cost model.

    cost = a * exp(b * quant_time) + c

    XXX(JW): For Perseus, first filter candidate execution options on measured cost.
    Then translate them into effective cost (cost - p2p_power * quant_time * unit_time)
    and fit the cost model on effective cost.
    """

    def __init__(
        self,
        options: CandidateExecutionOptions,
        initial_guess: tuple[float, float, float] | None = None,
        search_strategy: Literal["best", "first"] = "first",
    ) -> None:
        """Fit the cost model from Pareto-optimal execution options.

        Args:
            options: Candidate execution options to fit the cost model with.
            initial_guess: Initial guess for the parameters of the exponential function.
                If None, do a grid search on the initial guesses.
            search_strategy: Strategy to use when doing a grid search on the initial guesses.
                'first' means to take the first set of parameters that fit.
                'best' means to take the set of parameters that fit with the lowest error.
        """
        self.fn = lambda t, a, b, c: a * np.exp(b * t) + c

        quant_times = np.array([option.quant_time for option in options.options])
        target_costs = np.array([option.cost for option in options.options])

        def l2_error(coefficients: tuple[float, float, float]) -> float:
            preds = np.array([self.fn(t, *coefficients) for t in quant_times])
            return np.mean(np.square(target_costs - preds))

        # When an initial parameter guess is provided, just use it.
        if initial_guess is not None:
            self.coefficients, pcov = curve_fit(
                self.fn,
                quant_times,
                target_costs,
                p0=initial_guess,
                maxfev=50000,
            )
            if np.inf in pcov:
                raise ValueError("Initial guess failed to fit.")
            logger.info(
                "Exponential cost model with initial guesses: %s, coefficients: %s, L2 error: %f",
                initial_guess,
                self.coefficients,
                l2_error(self.coefficients),
            )
            return

        # Otherwise, do a grid search on the initial guesses.
        if search_strategy not in ["best", "first"]:
            raise ValueError("search_strategy must be either 'best' or 'first'.")

        unit_time = options.options[0].unit_time
        a_candidates = [1e1, 1e2, 1e3]
        b_candidates = [
            -c * unit_time / 0.001 for c in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
        ]
        c_candidates = [1e1, 1e2, 1e3]
        coefficients = self.run_grid_search(
            options, a_candidates, b_candidates, c_candidates, search_strategy
        )
        if coefficients is None:
            raise ValueError(
                "Grid search failed to fit. "
                "Manually fit the model with `Exponential.run_grid_search` "
                "and provide an initial guess to `Exponential.__init__`."
            )
        self.coefficients = coefficients

    def run_grid_search(
        self,
        options: CandidateExecutionOptions,
        a_candidates: list[float],
        b_candidates: list[float],
        c_candidates: list[float],
        search_strategy: Literal["best", "first"],
    ) -> tuple[float, float, float] | None:
        """Run a grid search on the initial guesses."""
        quant_times = np.array([option.quant_time for option in options.options])
        target_costs = np.array([option.cost for option in options.options])

        def l2_error(coefficients: tuple[float, float, float]) -> float:
            preds = np.array([self.fn(t, *coefficients) for t in quant_times])
            return np.mean(np.square(target_costs - preds))

        best_error = np.inf
        best_coefficients = (np.inf, np.inf, np.inf)
        initial_guess = next(
            itertools.product(a_candidates, b_candidates, c_candidates)
        )

        logger.info(
            "Running grid search for exponential model initial parameter guess."
        )
        for a, b, c in itertools.product(a_candidates, b_candidates, c_candidates):
            initial_guess = [a, b, c]
            (opt_a, opt_b, opt_c), pcov = curve_fit(
                self.fn,
                quant_times,
                target_costs,
                p0=initial_guess,
                maxfev=50000,
            )
            coefficients = (opt_a, opt_b, opt_c)

            # Skip if the fit failed.
            if np.inf in pcov:
                continue
            error = l2_error(coefficients)
            if error == np.inf:
                continue

            # We have coefficients that somewhat fit the data.
            logger.info(
                "Initial guess %s fit with coefficients %s and error %f.",
                initial_guess,
                coefficients,
                error,
            )
            if search_strategy == "first":
                logger.info("Strategy is 'first'. Search finished.")
                best_coefficients = coefficients
                best_error = error
                break
            elif search_strategy == "best":
                if error < best_error:
                    logger.info("Strategy is 'best' and error is better. Taking it.")
                    best_coefficients = coefficients
                    best_error = error
                else:
                    logger.info("Strategy is 'best' but error is worse.")

        if best_error == np.inf:
            raise ValueError("Nothing in the grid search space was able to fit.")

        logger.info("Final coefficients: %s", best_coefficients)
        return best_coefficients

    def __call__(self, quant_time: int) -> float:
        """Predict execution cost given quantized time."""
        return self.fn(quant_time, *self.coefficients)


@define
class OperationSpec(Generic[KnobT]):
    """An operation spec with multiple Pareto-optimal (time, cost) execution options.

    In the computation DAG, there may be multiple operations with the same type
    that share the set of execution options and cost model. This class specifies the
    type of operation (i.e., spec), and actual operations on the DAG hold a reference
    to its operation spec.

    Attributes:
        options: Candidate execution options of this operation.
        cost_model: A continuous cost model fit from candidate execution options.
    """

    options: CandidateExecutionOptions[KnobT]
    cost_model: CostModel


@define(slots=False)
class Operation(Generic[KnobT]):
    """Base class for operations on the computation DAG.

    Attributes:
        spec: The operation spec of this operation.
        is_dummy: Whether this operation is a dummy operation. See `DummyOperation`.
        duration: The current planned duration, in quanitzed time.
        max_duration: The maximum duration of this operation, in quantized time.
        min_duration: The minimum duration of this operation, in quantized time.
        assigned_knob: The knob value chosen for the value of `duration`.
        earliest_start: The earliest time this operation can start, in quantized time.
        latest_start: The latest time this operation can start, in quantized time.
        earliest_finish: The earliest time this operation can finish, in quantized time.
        latest_finish: The latest time this operation can finish, in quantized time.
    """

    spec: OperationSpec[KnobT] = field(on_setattr=frozen)
    is_dummy: bool = field(default=False, init=False, on_setattr=frozen)

    duration: int = field(init=False)
    max_duration: int = field(init=False)
    min_duration: int = field(init=False)
    assigned_knob: KnobT = field(init=False)

    earliest_start: int = field(default=0, init=False)
    latest_start: int = field(default=sys.maxsize, init=False)
    earliest_finish: int = field(default=0, init=False)
    latest_finish: int = field(default=sys.maxsize, init=False)

    def __attrs_post_init__(self) -> None:
        """Compute derived attributes."""
        quant_times = [o.quant_time for o in self.spec.options.options]
        self.max_duration = max(quant_times)
        self.min_duration = min(quant_times)
        self.duration = self.min_duration

    def assign_knob(self) -> None:
        """Find and assign the slowest `knob` value that still meets `duration`."""
        if self.duration < self.min_duration or self.duration > self.max_duration:
            raise ValueError(
                f"Duration {self.duration} is not in range "
                f"[{self.min_duration}, {self.max_duration}]."
            )

        # Run binary search on options.
        sorted_options = sorted(self.spec.options.options, key=lambda x: x.quant_time)
        i = bisect.bisect_right([o.quant_time for o in sorted_options], self.duration)
        self.assigned_knob = sorted_options[i - 1].knob


@define(slots=False, repr=False)
class DummyOperation(Operation):
    """A dummy operation that does nothing.

    An `AttributeError` is raised when you try to access `spec`.
    """

    spec: OperationSpec = field(init=False, on_setattr=frozen)
    is_dummy: bool = field(default=True, init=False, on_setattr=frozen)

    def __attrs_post_init__(self) -> None:
        """Compute derived attributes."""
        self.max_duration = sys.maxsize
        self.min_duration = 0
        self.duration = 0

    def __repr__(self) -> str:
        return "DummyOperation()"