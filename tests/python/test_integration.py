"""End-to-end workflows on .npy files."""

import numpy as np

from aex import ArrayProxy, GroupProxy


def test_workflow_open_explore_read_close(client, ds_paths):
    f = client.open(ds_paths["ds1"])

    assert len(list(f)) == 1
    assert "array" in f

    dataset = f["/array"]
    assert dataset.shape == (100, 200)
    assert dataset.dtype == np.float32

    data = dataset[0:10]
    assert data.shape == (10, 200)
    assert isinstance(data, np.ndarray)

    f.close()


def test_workflow_navigate(client, ds_paths):
    f = client.open(ds_paths["ds3"])
    assert isinstance(f["/"], GroupProxy)
    ds = f["/array"]
    assert isinstance(ds, ArrayProxy)
    assert ds.dtype == np.int8
    f.close()


def test_workflow_multiple_selections_same_file(client, ds_paths):
    f = client.open(ds_paths["ds1"])
    dataset = f["/array"]
    assert dataset[0:10].shape == (10, 200)
    assert dataset[50:60].shape == (10, 200)

    other = client.open(ds_paths["ds2"])["/array"]
    assert other[0].shape == (200,)
    f.close()


def test_two_clients_at_once(server, ds_paths):
    from aex import Client

    with Client(server) as a, Client(f"http://{server}") as b:
        x = a.open(ds_paths["ds1"])["array"][3]
        y = b.open(ds_paths["ds1"])["array"][3]
        assert np.array_equal(x, y)
