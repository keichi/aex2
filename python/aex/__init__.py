"""AEX2: partial, on-demand transfer of array data over wide-area networks.

Arrays stay on a server. Indexing a proxy transfers only the selection, and
reductions such as `np.mean` run on the server, so only the result comes back.

Serve a directory of `.npy` and HDF5 (netCDF-4 included) files, read-only:

```console
$ aex-server --control-addr 127.0.0.1:50051 --data-addr 127.0.0.1:50052 --root DIR
```

```python
import aex
import numpy as np

with aex.Client("127.0.0.1:50051") as client:
    arr = client.open("ocean.npy")["array"]  # an ArrayProxy
    arr[10:20, ::2]                          # a read-only ndarray
    arr[arr[:, 0] > 0]                       # boolean mask

    np.mean(arr, axis=0)                     # computed on the server
    np.max(arr.view[10:90], axis=1)          # the selection is not transferred

    arr.read_into(buf, np.s_[0:1000])        # into a buffer you already have
    a, b = arr.gather([np.s_[0:10], np.s_[90:100]])   # one round trip, not two
    future = arr.get_async(np.s_[:])         # overlap transfer with compute
    client.stats()                           # bytes moved, round trip, and so on
```

Transfer settings come from the environment, so a benchmark can sweep them
without touching the code: `AEX_STREAMS` (data connections, default 8),
`AEX_CREDIT` (`FETCH` frames in flight per connection, default 16),
`AEX_CHUNK_BYTES`, and `AEX_DATA_ENDPOINT` when the address the server
advertises for the data plane is not the one to dial. On a link with latency,
throughput is set by `streams * credit * chunk_bytes` alone.
"""

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
    "Client",
    "FileProxy",
    "GroupProxy",
    "ArrayProxy",
    "QualityView",
    "set_fallback_policy",
    "set_fallback_threshold",
    "AexError",
    "AexConnectionError",
    "AexProtocolError",
    "AexTransferError",
    "AexValueError",
    "AexNotFoundError",
    "AexFallbackError",
    "AexFallbackWarning",
    "AexQualityWarning",
]
