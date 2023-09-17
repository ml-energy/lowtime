# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An instruction is an atomic an operation in pipeline training."""

from __future__ import annotations

from attrs import define

from poise.operation import Operation


@define(slots=False, kw_only=True)
class Instruction(Operation):
    """An operation on a pipeline training schedule."""

    stage_id: int
    micro_batch_id: int

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"{type(self).__name__}(S{self.stage_id}B{self.micro_batch_id}, {self.duration}@{self.assigned_knob})"


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
