"""
HDF5 I/O for Sample objects.

File layout
-----------
/samples/{uid}/          ← storage-optimised (chunked, compressed)
    images/{marker}
    denoised_images/{marker}
    segmentations/{marker}
    features/{column}    ← one dataset per feature column
    logs                 ← JSON string
    attrs: uid, written_at, root_dir, markers, compartments

/view/{uid}/             ← view-optimised (soft-links to /samples/{uid})
    images               → /samples/{uid}/images
    denoised_images      → /samples/{uid}/denoised_images
    segmentations        → /samples/{uid}/segmentations
    features             → /samples/{uid}/features
    logs                 → /samples/{uid}/logs

Concurrency
-----------
Multiple processes may call ``write_sample`` simultaneously on the same file.
Writes are serialised with an advisory POSIX file lock on a companion
``<filepath>.lock`` file (``fcntl.flock``).
"""

import fcntl
import json
import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from gamgee.analysis import Sample


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_chunks(shape: tuple) -> tuple:
    """One chunk = one full image for small arrays; 512-tile otherwise."""
    if all(s <= 4096 for s in shape):
        return shape
    return tuple(min(s, 512) for s in shape)


def _write_image_group(grp: h5py.Group, images: dict) -> None:
    """Write a {marker: ndarray} dict into *grp* with chunked gzip storage."""
    for marker, img in images.items():
        if img is None:
            continue
        arr = np.asarray(img)
        grp.create_dataset(
            marker,
            data=arr,
            chunks=_image_chunks(arr.shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        grp.attrs["CLASS"] = np.bytes_("IMAGE")
    
        # Recommended: spec version
        grp.attrs["IMAGE_VERSION"] = np.bytes_("1.2")


def _write_features(sample_grp: h5py.Group, features: list) -> None:
    """
    Store a list of feature dicts as a columnar group.

    String columns use variable-length HDF5 strings; numeric columns use
    float64.  Column order and names are stored in the group's ``columns``
    attribute.
    """
    if not features:
        return
    feat_grp = sample_grp.create_group("features")
    columns = list(features[0].keys())

    for col in columns:
        values = [row.get(col) for row in features]
        if isinstance(values[0], str):
            arr = np.array([v if v is not None else "" for v in values], dtype=object)
            feat_grp.create_dataset(col, data=arr, dtype=h5py.string_dtype())
        else:
            arr = np.array(
                [v if v is not None else np.nan for v in values], dtype=np.float64
            )
            feat_grp.create_dataset(col, data=arr)

    feat_grp.attrs["n_rows"] = len(features)
    feat_grp.attrs["columns"] = json.dumps(columns)


def _write_sample_to_file(f: h5py.File, sample: "Sample") -> None:
    uid = str(sample.uid)

    # ------------------------------------------------------------------
    # /samples/{uid}  –  storage-optimised
    # ------------------------------------------------------------------
    storage_root = f.require_group("samples")
    if uid in storage_root:
        del storage_root[uid]  # replace stale data for this uid

    sg = storage_root.create_group(uid)
    sg.attrs["uid"] = uid
    sg.attrs["written_at"] = datetime.datetime.now().isoformat()
    sg.attrs["root_dir"] = str(sample.root_dir)
    sg.attrs["markers"] = json.dumps(sample.markers)
    sg.attrs["compartments"] = json.dumps(sample.compartments)

    _write_image_group(sg.create_group("images"), sample.images)
    _write_image_group(sg.create_group("denoised_images"), sample.denoised_images)
    _write_image_group(sg.create_group("segmentations"), sample.segmentations)

    sg.create_dataset("logs", data=json.dumps(sample.logs))

    _write_features(sg, sample.features)

    # ------------------------------------------------------------------
    # /view/{uid}  –  view-optimised soft-link mirror
    # ------------------------------------------------------------------
    view_root = f.require_group("view")
    if uid in view_root:
        del view_root[uid]

    vg = view_root.create_group(uid)
    for subpath in ("images", "denoised_images", "segmentations", "features", "logs"):
        vg[subpath] = h5py.SoftLink(f"/samples/{uid}/{subpath}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_sample(filepath: "str | Path", sample: "Sample") -> None:
    """
    Append (or overwrite) *sample* in the HDF5 file at *filepath*.

    The file is created automatically when it does not yet exist.
    Concurrent calls from ``multiprocessing.Pool`` workers are safe: each
    call acquires an exclusive POSIX advisory lock before touching the file.

    Parameters
    ----------
    filepath:
        Path to the ``.h5`` / ``.hdf5`` output file.
    sample:
        A fully processed :class:`~gamgee.analysis.Sample` instance.
    """
    filepath = Path(filepath)
    lock_path = filepath.with_suffix(filepath.suffix + ".lock")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            with h5py.File(filepath, "a") as f:
                _write_sample_to_file(f, sample)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_sample(filepath: "str | Path", uid: str) -> dict:
    """
    Read a single sample back from *filepath* by its *uid*.

    Returns a dict with keys ``uid``, ``markers``, ``compartments``,
    ``images``, ``denoised_images``, ``segmentations``, ``features``,
    ``logs``.  Images and segmentations are returned as ``np.ndarray``.
    Features are returned as a list of dicts (same shape as
    ``Sample.features``).
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        sg = f[f"samples/{uid}"]
        result = {
            "uid": sg.attrs["uid"],
            "markers": json.loads(sg.attrs["markers"]),
            "compartments": json.loads(sg.attrs["compartments"]),
            "root_dir": sg.attrs["root_dir"],
            "images": {k: sg["images"][k][()] for k in sg["images"]},
            "denoised_images": {k: sg["denoised_images"][k][()] for k in sg["denoised_images"]},
            "segmentations": {k: sg["segmentations"][k][()] for k in sg["segmentations"]},
            "logs": json.loads(sg["logs"][()]),
        }

        features = []
        if "features" in sg:
            feat_grp = sg["features"]
            columns = json.loads(feat_grp.attrs["columns"])
            n = feat_grp.attrs["n_rows"]
            col_arrays = {col: feat_grp[col][()] for col in columns}
            for i in range(n):
                row = {}
                for col in columns:
                    v = col_arrays[col][i]
                    row[col] = v.decode() if isinstance(v, bytes) else v
                features.append(row)
        result["features"] = features
    return result


def list_samples(filepath: "str | Path") -> list[str]:
    """Return the list of sample UIDs stored in *filepath*."""
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        return list(f.get("samples", {}).keys())
