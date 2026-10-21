"""Optional, transactional Icechunk I/O for xarray datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import xarray as xr

if TYPE_CHECKING:
    from icechunk import Repository

__all__ = ["open_icechunk", "write_icechunk"]


def open_icechunk(repository: Repository, *, branch: str = "main") -> xr.Dataset:
    """Open the committed branch snapshot as a lazy Dataset.

    Args:
        repository: An existing Icechunk repository. Install ``climate_indices[icechunk]``
            to create or open repositories with Icechunk (Python 3.11 or later).
        branch: The branch to read. The read-only session fixes the snapshot at open time.

    Returns:
        A Dataset with Dask-backed variables. Close it after use.
    """
    session = repository.readonly_session(branch=branch)
    return cast(xr.Dataset, xr.open_zarr(session.store, consolidated=False, chunks={}))


def write_icechunk(
    dataset: xr.Dataset,
    repository: Repository,
    *,
    message: str,
    branch: str = "main",
    mode: Literal["w-", "w", "a", "r+"] = "w-",
) -> str:
    """Write a Dataset in one session, then commit and return its snapshot ID.

    Use ``to_icechunk``, not ``Dataset.to_zarr``, to track Dask worker writes.
    The writer disables consolidated metadata and completes all writes before commit.
    A write or commit failure propagates without replacing the committed snapshot.

    Args:
        dataset: The eager or Dask-backed Dataset to write.
        repository: An existing Icechunk repository. This helper never deletes or
            recreates it. Install ``climate_indices[icechunk]`` (Python 3.11 or later).
        message: The commit message.
        branch: The branch to update. Use one writer per branch.
        mode: ``w-`` rejects an existing group. Use ``w`` to replace its contents
            in the new snapshot, ``a`` to update variables, or ``r+`` to update values.

    Returns:
        The committed snapshot ID.
    """
    from icechunk.xarray import to_icechunk

    session = repository.writable_session(branch=branch)
    to_icechunk(dataset, session, mode=mode)
    return session.commit(message)
