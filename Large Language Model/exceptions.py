__all__ = [
    "NormalizationError",
    "AttentionError",
    ]


class NormalizationError(Exception):
    """
    An error with normalization layer initialization.
    """


class AttentionError(Exception):
    """
    An error with attention layer initialization.
    """
