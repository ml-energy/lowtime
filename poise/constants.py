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

"""Constants used throughout the package."""

from poise.perseus.instruction import Forward, Backward, Recomputation, ForwardBackward

# The default error tolerance for floating point comparisons.
FP_ERROR = 1e-6

# The default arguments for matplotlib.patches.Rectangle.
DEFAULT_RECTANGLE_ARGS = {
    Forward: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    Backward: dict(facecolor="#9fc887", edgecolor="#000000", linewidth=1.0),
    Recomputation: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    ForwardBackward: dict(facecolor="#f542a4", edgecolor="#000000", linewidth=1.0),
}

# The default arguments for matplotlib.axes.Axes.annotate.
DEFAULT_ANNOTATION_ARGS = {
    Forward: dict(color="#2a5889", fontsize=10.0, ha="center", va="center"),
    Backward: dict(color="#000000", fontsize=10.0, ha="center", va="center"),
    Recomputation: dict(color="#5b2a89", fontsize=10.0, ha="center", va="center"),
    ForwardBackward: dict(color="#f542a4", fontsize=10.0, ha="center", va="center"),
}

# The default arguments for matplotlib.axes.Axes.plot.
DEFAULT_LINE_ARGS = dict(color="#00a6ff", linewidth=4.0)
