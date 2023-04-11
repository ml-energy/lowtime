"""A schedule describes the ordering of pipeline instructions like forward and backward."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator

from rene.instruction import Instruction, Forward, Backward, Recomputation


class PipelineSchedule(ABC):
    """Abstract class that defines a pipeline schedule.

    Designed to look like DeepSpeed's PipeSchedule class.
    """

    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
        stage_id: int,
    ) -> None:
        """Instantiate the pipeline schedule.

        Arguments:
            num_stages: The number of pipeline stages
            num_micro_batches: The number of micro batches in the pipeline
            stage_id: Zero-indexed pipeline stage for `step` to yield instructions for
        """
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def __iter__(self) -> Generator[Instruction, None, None]:
        """Return a generator that yields `Instruction`s for one stage.

        `Instruction`s just need their stage ID and microbatch ID.

        This method also corresponds to DeepSpeed's PipeSchedule.steps method.
        However, in Rene, one step doesn't have much meaning. We just exhaust the
        generator immediately to get a list of all instructions.
        """


class Synchronous1F1B(PipelineSchedule):
    """Describes the synchronous 1F1B schedule.

    Adapted from DeepSpeed's TrainSchedule class.
    """

    def __iter__(self) -> Generator[Instruction, None, None]:
        """Return a generator that yields `Instruction`s for one stage."""
        total_steps = 2 * (self.num_micro_batches + self.num_stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    yield Forward(self.stage_id, micro_batch_id)
                else:
                    yield Backward(self.stage_id, micro_batch_id)

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.num_micro_batches

    def _step_to_micro_batch(self, step_id):
        def _is_even(x: int) -> bool:
            return x % 2 == 0

        def _is_odd(x: int) -> bool:
            return x % 2 != 0

        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            raise AssertionError()

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.num_stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.num_stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class EarlyRecomputation1F1B(Synchronous1F1B):
    """Early recomputation 1F1B schedule from Merak."""

    def __iter__(self) -> Generator[Instruction, None, None]:
        """Return a generator that yields `Instruction`s for one stage."""
        total_steps = 2 * (self.num_micro_batches + self.num_stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    yield Forward(self.stage_id, micro_batch_id)
                else:
                    # Recomputateion right before backward.
                    yield Recomputation(self.stage_id, micro_batch_id)
                    yield Backward(self.stage_id, micro_batch_id)
