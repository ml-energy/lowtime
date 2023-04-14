"""Constants used throughout the package."""

from rene.instruction import Forward, Backward

# The default error tolerance for floating point comparisons.
FP_ERROR = 1e-6

# The default arguments for matplotlib.patches.Rectangle.
DEFAULT_RECTANGLE_ARGS = {
    Forward: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    Backward: dict(facecolor="#9fc887", edgecolor="#000000", linewidth=1.0),
}

# The default arguments for matplotlib.axes.Axes.annotate.
DEFAULT_ANNOTATION_ARGS = {
    Forward: dict(color="#000000", fontsize=10.0, ha="center", va="center"),
    Backward: dict(color="#000000", fontsize=10.0, ha="center", va="center"),
}

# The default arguments for matplotlib.axes.Axes.plot.
DEFAULT_LINE_ARGS = dict(color="#00a6ff", linewidth=4.0)
