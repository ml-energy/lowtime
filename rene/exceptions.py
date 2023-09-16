class ReneBaseError(Exception):
    """Base class for all exceptions in rene."""

    def __init__(self, message):
        self.message = message


class ReneFlowError(ReneBaseError):
    """Error while running flow operations on graphs."""


class ReneGraphError(ReneBaseError):
    """Error while manipulating graphs."""
