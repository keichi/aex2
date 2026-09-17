"""AEX2: partial, on-demand transfer of array data over wide-area networks."""

from .array_proxy import ArrayProxy, QualityView, set_fallback_policy, set_fallback_threshold
from .client import Client, FileProxy, GroupProxy
from .errors import (
    AexConnectionError,
    AexError,
    AexFallbackError,
    AexFallbackWarning,
    AexNotFoundError,
    AexProtocolError,
    AexQualityWarning,
    AexTransferError,
    AexValueError,
)

__all__ = [
    "AexConnectionError",
    "AexError",
    "AexFallbackError",
    "AexFallbackWarning",
    "AexNotFoundError",
    "AexProtocolError",
    "AexQualityWarning",
    "AexTransferError",
    "AexValueError",
    "ArrayProxy",
    "Client",
    "FileProxy",
    "GroupProxy",
    "QualityView",
    "set_fallback_policy",
    "set_fallback_threshold",
]
