"""
Error helper for use with non-ADMM optimization.
"""


class ADMMDisabledError(Exception):
    """
    Exception used when a method is called for an ADMMDisabledProjection that
    should never be called.
    """

    def __init__(self, method_name: str):
        message = "Called method {} on instance of ADMMDisabled".format(method_name)
        super(ADMMDisabledError, self).__init__(message)
