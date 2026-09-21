"""Exceptions raised by aex.

Every exception carries ``error_class``, the server's classification, so code
can branch on it without parsing messages.

Each one also inherits the standard exception that fits it, so ``except
ConnectionError`` and the usual retry wrappers catch aex errors too.
"""


class AexError(Exception):
    """Base of every aex error."""

    def __init__(self, message: str, error_class: str = "PERMANENT") -> None:
        super().__init__(message)
        self.error_class = error_class

    @property
    def is_retryable(self) -> bool:
        """Whether retrying the same request could succeed."""
        return self.error_class == "TRANSIENT"


class AexProtocolError(AexError):
    """The two sides disagree about the protocol. A bug on one of them."""


class AexConnectionError(AexError, ConnectionError):
    """The server could not be reached, or refused the session."""


class AexTransferError(AexError, OSError):
    """A transfer failed on the server's side."""


class AexValueError(AexError, ValueError):
    """The request was invalid: out of range, over a limit, unsupported."""


class AexNotFoundError(AexError, KeyError):
    """No file or item at that path."""

    # KeyError would quote the message.
    __str__ = Exception.__str__


class AexFallbackError(AexError):
    """A numpy function would have downloaded the array, and the policy forbids it."""


class AexFallbackWarning(UserWarning):
    """A numpy function is downloading the array to compute locally."""


class AexQualityWarning(UserWarning):
    """The server sent the data at a different quality than was asked for."""
