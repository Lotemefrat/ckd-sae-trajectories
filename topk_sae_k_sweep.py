#!/usr/bin/env python3
"""Train a Top-K sparse autoencoder sweep on Qwen visit embeddings.

Expected project layout
-----------------------

    DATA_DIR/
        meta_v3_all.csv                 # ``$`` separated
        PATIENT_ID_1/PATIENT_ID_1.npy   # (n_embedding_rows, 4096)
        PATIENT_ID_2/PATIENT_ID_2.npy
        ...

The visit embeddings were generated after sorting encounters by ``PatientID``
and ``date`` and assigning the zero-based, patient-local ``emb_id``.  The
metadata was subsequently joined to CKD rows, which can duplicate a
``PatientID + emb_id``.  Depending on which embedding-generator version made a
file, its rows can therefore correspond either to:

1. every stably sorted metadata row (including repeated ``emb_id`` values), or
2. one row per unique ``emb_id``.

This script detects both layouts.  During a k sweep it reads only ``PatientID``
and ``emb_id`` and builds the minimal row map needed to retain one copy of each
true visit for model fitting.  The full note/encounter audit index is deferred
until activation export is explicitly requested.  At that point repeated
metadata and saved embeddings are verified, and every activation can be traced
back to ``PatientID``, ``emb_id``, ``enc_id``, ``date``, and ``enc_summary``.

The default model is the current project architecture, 4096 -> 16384 -> 4096.
For every requested k, a fresh model is trained with the same initialization,
patient split, and data order.  Sparsity is imposed only by Top-K; there is no
L1 penalty.  The optimized reconstruction loss is

    mean_i ||x_i - x_hat_i||_2^2

(the sum over 4096 dimensions, then the mean over visits), matching the loss
convention used in the previous SAE runs.

Important outputs
-----------------

``k_vs_reconstruction_error.csv``
    One row per k.  ``validation_reconstruction_error`` is the primary value
    for the k-versus-error plot.
``k_vs_reconstruction_error.png``
    Plot produced automatically when matplotlib is available.
``minimal_index_qc.json``
    Lightweight ``PatientID + emb_id`` mapping checks used for every sweep.
``patients_missing_metadata.csv``
    Explicit record of embedding directories absent from the metadata.  These
    are skipped only when ``--skip-patients-missing-metadata`` is supplied.
``embedding_row_index.csv``
    Created only when activation export is requested; one row per saved
    embedding row, including the original note and IDs.
``unique_visit_index.csv``
    Created only when activation export is requested; one row per true
    ``PatientID + emb_id`` visit after full duplicate checks.
``data_manifest.json``
    Source files, stable patient split, file statistics, and mapping QC.
``k_XXXX/best.pt``
    Best validation checkpoint for that k.
``k_XXXX/activations/<split>/<PatientID>/sae_topk.npz``
    Optional storage-efficient Top-K indices/values.  Array row r maps to
    ``activation_row == r`` in the corresponding activation index CSV.

Example
-------

    python topk_sae_k_sweep.py \
        --data-dir /path/to/embeddings \
        --output-dir topk_sae_k_sweep \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


DEFAULT_DATA_DIR = ("/path/to/embeddings"
)
DEFAULT_K_VALUES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000)
INDEX_SEPARATOR = "$"
LOGGER = logging.getLogger("topk_sae_k_sweep")


# -----------------------------------------------------------------------------
# Configuration, files, and reproducibility
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientFile:
    patient_id: str
    path: str
    n_embedding_rows: int
    size_bytes: int
    mtime_ns: int


def parse_k_values(text: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(piece.strip()) for piece in text.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("every k must be a positive integer")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--ks contains duplicate values")
    return values


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_dataframe_csv(df: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    df.to_csv(temporary, index=False, sep=sep)
    os.replace(temporary, path)


def atomic_csv_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_embedding_file(
    patient_dir: Path,
    embedding_filename: str,
) -> Path:
    if embedding_filename.lower() == "auto":
        candidates = sorted(patient_dir.glob("*.npy"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one .npy file in {patient_dir}; found "
                f"{len(candidates)}: {[path.name for path in candidates]}"
            )
        return candidates[0]
    filename = embedding_filename.format(patient_id=patient_dir.name)
    path = patient_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing embedding file: {path}")
    return path


def discover_patient_files(
    data_dir: Path,
    embedding_filename: str,
    d_model: int,
    require_unit_norm: bool,
    unit_norm_tolerance: float,
    full_value_validation: bool,
    skip_invalid_files: bool,
) -> Tuple[List[PatientFile], List[Dict[str, str]], Dict[str, float]]:
    """Find one embedding matrix per patient and validate shape/content."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    patient_dirs = [path for path in sorted(data_dir.iterdir()) if path.is_dir()]
    if not patient_dirs:
        raise RuntimeError(f"no patient directories found under {data_dir}")

    entries: List[PatientFile] = []
    skipped: List[Dict[str, str]] = []
    norm_count = 0
    norm_sum = 0.0
    norm_sq_sum = 0.0
    norm_min = float("inf")
    norm_max = float("-inf")

    for number, patient_dir in enumerate(patient_dirs, start=1):
        candidate = patient_dir
        try:
            candidate = resolve_embedding_file(patient_dir, embedding_filename).resolve()
            array = np.load(candidate, mmap_mode="r", allow_pickle=False)
            if array.ndim == 1 and array.shape == (d_model,):
                array_2d = array.reshape(1, d_model)
            elif array.ndim == 2 and array.shape[1] == d_model:
                array_2d = array
            else:
                raise ValueError(
                    f"shape {array.shape}; expected ({d_model},) or (n, {d_model})"
                )
            n_rows = int(array_2d.shape[0])
            if n_rows == 0:
                raise ValueError("embedding file contains zero rows")
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"embedding dtype is nonnumeric: {array.dtype}")

            if full_value_validation:
                starts = range(0, n_rows, 2048)
            else:
                starts = (0,)

            file_norms: List[np.ndarray] = []
            for start in starts:
                if full_value_validation:
                    block = np.asarray(array_2d[start : start + 2048], dtype=np.float32)
                else:
                    sample_n = min(256, n_rows)
                    sample_rows = np.linspace(0, n_rows - 1, sample_n, dtype=np.int64)
                    block = np.asarray(array_2d[sample_rows], dtype=np.float32)
                if not np.isfinite(block).all():
                    raise ValueError("embedding file contains NaN or infinity")
                norms = np.linalg.norm(block, axis=1)
                if np.any(norms <= 1e-8):
                    raise ValueError("embedding file contains a zero vector")
                if require_unit_norm:
                    outside = np.abs(norms - 1.0) > unit_norm_tolerance
                    if np.any(outside):
                        bad = float(norms[np.flatnonzero(outside)[0]])
                        raise ValueError(
                            f"embedding norm {bad:.6f} is outside 1 +/- "
                            f"{unit_norm_tolerance}"
                        )
                file_norms.append(norms.astype(np.float64, copy=False))
                if not full_value_validation:
                    break

            norms_all = np.concatenate(file_norms)
            norm_count += int(norms_all.size)
            norm_sum += float(norms_all.sum())
            norm_sq_sum += float(np.square(norms_all).sum())
            norm_min = min(norm_min, float(norms_all.min()))
            norm_max = max(norm_max, float(norms_all.max()))
            stat = candidate.stat()
            entries.append(
                PatientFile(
                    patient_id=patient_dir.name,
                    path=str(candidate),
                    n_embedding_rows=n_rows,
                    size_bytes=int(stat.st_size),
                    mtime_ns=int(stat.st_mtime_ns),
                )
            )
        except Exception as exc:
            record = {
                "patient_id": patient_dir.name,
                "path": str(candidate),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            skipped.append(record)
            LOGGER.warning("Skipping invalid patient file: %s", record)
            if not skip_invalid_files:
                raise

        if number % 100 == 0 or number == len(patient_dirs):
            LOGGER.info(
                "Validated %d/%d patient directories | valid=%d | skipped=%d",
                number,
                len(patient_dirs),
                len(entries),
                len(skipped),
            )

    if not entries:
        raise RuntimeError("no valid patient embedding files remain")
    mean = norm_sum / norm_count
    variance = max(norm_sq_sum / norm_count - mean * mean, 0.0)
    qc = {
        "n_patient_directories": len(patient_dirs),
        "n_valid_patients": len(entries),
        "n_skipped_patients": len(skipped),
        "n_embedding_rows": sum(item.n_embedding_rows for item in entries),
        "n_norms_checked": norm_count,
        "embedding_norm_mean": mean,
        "embedding_norm_std": math.sqrt(variance),
        "embedding_norm_min": norm_min,
        "embedding_norm_max": norm_max,
    }
    return entries, skipped, qc


def stable_patient_split(
    entries: Sequence[PatientFile],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[List[PatientFile], List[PatientFile], List[PatientFile]]:
    if not math.isclose(
        train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-10
    ):
        raise ValueError("train/validation/test fractions must sum to 1")
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("all split fractions must be positive")

    train: List[PatientFile] = []
    validation: List[PatientFile] = []
    test: List[PatientFile] = []
    train_cut = train_fraction
    validation_cut = train_fraction + val_fraction
    for entry in entries:
        token = f"{seed}:{entry.patient_id}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
        if value < train_cut:
            train.append(entry)
        elif value < validation_cut:
            validation.append(entry)
        else:
            test.append(entry)
    if not train or not validation or not test:
        raise RuntimeError(
            "stable patient split produced an empty split; use more patients or "
            "different split fractions/seed"
        )
    return train, validation, test


def manifest_signature(entries: Sequence[PatientFile], metadata_path: Path) -> str:
    digest = hashlib.sha256()
    metadata_stat = metadata_path.stat()
    digest.update(
        f"{metadata_path.resolve()}\t{metadata_stat.st_size}\t"
        f"{metadata_stat.st_mtime_ns}\n".encode("utf-8")
    )
    for entry in sorted(entries, key=lambda item: item.patient_id):
        digest.update(
            (
                f"{entry.patient_id}\t{entry.path}\t{entry.n_embedding_rows}\t"
                f"{entry.size_bytes}\t{entry.mtime_ns}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Exact metadata-to-embedding row mapping
# -----------------------------------------------------------------------------


def build_minimal_model_row_map(
    metadata_path: Path,
    metadata_separator: str,
    entries: Sequence[PatientFile],
    output_dir: Path,
    deduplicate_repeated_emb_id: bool,
    skip_patients_missing_metadata: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object], List[str]]:
    """Build the fitting row map without loading notes or encounter columns.

    This is the only metadata pass needed for a k sweep.  The metadata is
    filtered and sorted once, then patient groups are retrieved in constant
    time.  It preserves the same two historical embedding layouts recognized
    by the full activation index.
    """
    LOGGER.info(
        "Building minimal model-row map from PatientID and emb_id only: %s",
        metadata_path,
    )
    try:
        meta = pd.read_csv(
            metadata_path,
            sep=metadata_separator,
            usecols=["PatientID", "emb_id"],
            dtype="string",
            keep_default_na=True,
            low_memory=False,
        )
    except ValueError as exc:
        raise KeyError(
            "metadata must contain PatientID and emb_id columns"
        ) from exc

    metadata_rows_total = len(meta)
    meta["_source_meta_row"] = np.arange(metadata_rows_total, dtype=np.int64)
    meta["PatientID"] = meta["PatientID"].astype("string").str.strip()
    if meta["PatientID"].isna().any() or (meta["PatientID"] == "").any():
        raise ValueError("metadata contains a missing/blank PatientID")

    numeric_emb_id = pd.to_numeric(meta["emb_id"], errors="coerce")
    invalid = numeric_emb_id.isna() | (numeric_emb_id < 0) | (
        numeric_emb_id != np.floor(numeric_emb_id)
    )
    if bool(invalid.any()):
        bad = meta.loc[invalid, ["PatientID", "emb_id"]].head(10).to_dict("records")
        raise ValueError(f"metadata contains invalid emb_id values: {bad}")
    meta["emb_id"] = numeric_emb_id.astype(np.int64)

    entries_by_id = {entry.patient_id: entry for entry in entries}
    relevant = meta.loc[meta["PatientID"].isin(entries_by_id)].copy()
    if relevant.empty:
        raise RuntimeError("metadata contains none of the patient directory IDs")
    relevant = relevant.sort_values(
        ["PatientID", "emb_id", "_source_meta_row"], kind="mergesort"
    ).reset_index(drop=True)
    group_indices = {
        str(patient_id): np.asarray(positions, dtype=np.int64)
        for patient_id, positions in relevant.groupby(
            "PatientID", sort=False, observed=True
        ).indices.items()
    }

    missing_patient_ids = sorted(set(entries_by_id).difference(group_indices))
    missing_records = [
        {
            "PatientID": patient_id,
            "embedding_path": entries_by_id[patient_id].path,
            "reason": f"no rows in {metadata_path.name}",
        }
        for patient_id in missing_patient_ids
    ]
    atomic_dataframe_csv(
        pd.DataFrame.from_records(
            missing_records,
            columns=["PatientID", "embedding_path", "reason"],
        ),
        output_dir / "patients_missing_metadata.csv",
    )
    if missing_patient_ids and not skip_patients_missing_metadata:
        raise ValueError(
            f"{len(missing_patient_ids)} patient embedding directories have no rows "
            f"in {metadata_path.name}; first IDs={missing_patient_ids[:10]}. "
            "Inspect patients_missing_metadata.csv, then rerun with "
            "--skip-patients-missing-metadata only if these directories are stale."
        )
    if missing_patient_ids:
        LOGGER.warning(
            "Skipping %d patient embedding directories missing from metadata; "
            "details -> %s",
            len(missing_patient_ids),
            output_dir / "patients_missing_metadata.csv",
        )

    matched_patient_ids = sorted(set(entries_by_id).intersection(group_indices))
    if not matched_patient_ids:
        raise RuntimeError("no embedding patients have matching metadata rows")

    model_rows_by_patient: Dict[str, np.ndarray] = {}
    mapping_mode_counts: Dict[str, int] = {}
    unique_true_visits = 0
    rows_used_for_sae = 0
    for number, patient_id in enumerate(matched_patient_ids, start=1):
        entry = entries_by_id[patient_id]
        positions = group_indices.get(patient_id)
        if positions is None:  # Guard for static type checkers; filtered above.
            raise RuntimeError(f"internal metadata grouping error for {patient_id}")
        emb_ids = relevant.iloc[positions]["emb_id"].to_numpy(dtype=np.int64)
        first_occurrence = np.empty(len(emb_ids), dtype=bool)
        first_occurrence[0] = True
        first_occurrence[1:] = emb_ids[1:] != emb_ids[:-1]
        unique_ids = emb_ids[first_occurrence]
        expected_ids = np.arange(len(unique_ids), dtype=np.int64)
        if not np.array_equal(unique_ids, expected_ids):
            raise ValueError(
                f"patient {patient_id} emb_id values are not contiguous zero-based: "
                f"first values={unique_ids[:10].tolist()}"
            )

        if len(emb_ids) == entry.n_embedding_rows:
            mode = "all_sorted_metadata_rows"
            if deduplicate_repeated_emb_id:
                model_rows = np.flatnonzero(first_occurrence).astype(
                    np.int64, copy=False
                )
            else:
                model_rows = np.arange(len(emb_ids), dtype=np.int64)
        elif len(unique_ids) == entry.n_embedding_rows:
            mode = "one_row_per_unique_emb_id"
            model_rows = np.arange(len(unique_ids), dtype=np.int64)
        else:
            raise ValueError(
                f"patient {patient_id}: embedding array has {entry.n_embedding_rows} "
                f"rows, metadata has {len(emb_ids)} rows and "
                f"{len(unique_ids)} unique emb_id values; neither layout matches"
            )

        if model_rows.size == 0:
            raise RuntimeError(f"patient {patient_id} has no rows selected for the SAE")
        model_rows_by_patient[patient_id] = model_rows
        mapping_mode_counts[mode] = mapping_mode_counts.get(mode, 0) + 1
        unique_true_visits += len(unique_ids)
        rows_used_for_sae += len(model_rows)

        if number % 10_000 == 0 or number == len(matched_patient_ids):
            LOGGER.info(
                "Prepared minimal row map for %d/%d patients",
                number,
                len(matched_patient_ids),
            )

    qc: Dict[str, object] = {
        "index_level": "minimal_training_row_map",
        "metadata_path": str(metadata_path.resolve()),
        "metadata_separator": metadata_separator,
        "metadata_columns_loaded": ["PatientID", "emb_id"],
        "metadata_rows_total": int(metadata_rows_total),
        "metadata_rows_for_valid_embedding_patients": int(len(relevant)),
        "embedding_patients_input": int(len(entries_by_id)),
        "matched_embedding_patients": int(len(model_rows_by_patient)),
        "embedding_patients_missing_metadata": int(len(missing_patient_ids)),
        "missing_metadata_report": "patients_missing_metadata.csv",
        "skipped_patients_missing_metadata": bool(
            missing_patient_ids and skip_patients_missing_metadata
        ),
        "unique_true_visits": int(unique_true_visits),
        "rows_used_for_sae": int(rows_used_for_sae),
        "repeated_source_rows_not_used_for_sae": int(
            len(relevant) - rows_used_for_sae
        ),
        "deduplicate_repeated_emb_id_for_model": deduplicate_repeated_emb_id,
        "mapping_mode_patient_counts": mapping_mode_counts,
        "full_note_encounter_index_deferred": True,
    }
    atomic_json_dump(qc, output_dir / "minimal_index_qc.json")
    return model_rows_by_patient, qc, missing_patient_ids


def _assert_duplicate_metadata_consistency(meta: pd.DataFrame) -> None:
    key = ["PatientID", "emb_id"]
    duplicated = meta.duplicated(key, keep=False)
    if not bool(duplicated.any()):
        return
    repeated = meta.loc[duplicated]
    for column in ("enc_id", "date", "enc_summary"):
        counts = repeated.groupby(key, sort=False, dropna=False)[column].nunique(
            dropna=False
        )
        bad = counts[counts > 1]
        if not bad.empty:
            examples = [tuple(index) for index in bad.index[:5]]
            raise ValueError(
                f"Repeated PatientID + emb_id rows disagree in {column}; "
                f"example keys: {examples}"
            )


def _verify_duplicate_embedding_rows(
    entry: PatientFile,
    mapped: pd.DataFrame,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> int:
    """Check duplicate emb_id rows only after their raw row mapping is fixed."""
    duplicated = mapped.duplicated("emb_id", keep=False)
    if not bool(duplicated.any()):
        return 0
    array = np.load(entry.path, mmap_mode="r", allow_pickle=False)
    array_2d = array.reshape(1, -1) if array.ndim == 1 else array
    checked = 0
    for _, group in mapped.loc[duplicated].groupby("emb_id", sort=False):
        positions = group["embedding_row"].to_numpy(dtype=np.int64)
        reference = np.asarray(array_2d[int(positions[0])], dtype=np.float32)
        for position in positions[1:]:
            comparison = np.asarray(array_2d[int(position)], dtype=np.float32)
            if not np.allclose(reference, comparison, atol=atol, rtol=rtol):
                maximum = float(np.max(np.abs(reference - comparison)))
                raise ValueError(
                    f"patient {entry.patient_id}, emb_id {int(group['emb_id'].iloc[0])} "
                    f"maps to nonidentical embedding rows; max abs difference={maximum}"
                )
            checked += 1
    return checked


def build_embedding_row_index(
    metadata_path: Path,
    metadata_separator: str,
    entries: Sequence[PatientFile],
    split_lookup: Mapping[str, str],
    output_dir: Path,
    deduplicate_repeated_emb_id: bool,
    verify_duplicate_embeddings: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    """Create a one-to-one row index for the saved embedding matrices.

    The full mapping is established before ``used_for_sae`` is chosen.  Thus,
    deduplication can change model weighting without ever changing raw row IDs.
    """
    LOGGER.info("Reading metadata from %s", metadata_path)
    meta = pd.read_csv(
        metadata_path,
        sep=metadata_separator,
        dtype="string",
        keep_default_na=True,
        low_memory=False,
    )
    required = {"PatientID", "emb_id", "enc_id", "date", "enc_summary"}
    missing = sorted(required.difference(meta.columns))
    if missing:
        raise KeyError(f"metadata is missing required columns: {missing}")
    meta["_source_meta_row"] = np.arange(len(meta), dtype=np.int64)
    meta["PatientID"] = meta["PatientID"].astype("string").str.strip()
    if meta["PatientID"].isna().any() or (meta["PatientID"] == "").any():
        raise ValueError("metadata contains a missing/blank PatientID")

    numeric_emb_id = pd.to_numeric(meta["emb_id"], errors="coerce")
    invalid = numeric_emb_id.isna() | (numeric_emb_id < 0) | (
        numeric_emb_id != np.floor(numeric_emb_id)
    )
    if bool(invalid.any()):
        bad = meta.loc[invalid, ["PatientID", "emb_id"]].head(10).to_dict("records")
        raise ValueError(f"metadata contains invalid emb_id values: {bad}")
    meta["emb_id"] = numeric_emb_id.astype(np.int64)

    valid_ids = {entry.patient_id for entry in entries}
    relevant = meta.loc[meta["PatientID"].isin(valid_ids)].copy()
    if relevant.empty:
        raise RuntimeError("metadata contains none of the patient directory IDs")
    _assert_duplicate_metadata_consistency(relevant)
    relevant = relevant.sort_values(
        ["PatientID", "emb_id", "_source_meta_row"], kind="mergesort"
    ).reset_index(drop=True)
    group_indices = {
        str(patient_id): np.asarray(positions, dtype=np.int64)
        for patient_id, positions in relevant.groupby(
            "PatientID", sort=False, observed=True
        ).indices.items()
    }

    mapped_parts: List[pd.DataFrame] = []
    mapping_modes: Dict[str, str] = {}
    duplicate_embedding_rows_checked = 0
    entries_by_id = {entry.patient_id: entry for entry in entries}

    for number, patient_id in enumerate(sorted(entries_by_id), start=1):
        entry = entries_by_id[patient_id]
        positions = group_indices.get(patient_id)
        if positions is None:
            raise ValueError(
                f"patient directory {patient_id!r} has no rows in {metadata_path.name}"
            )
        patient_meta = relevant.iloc[positions].copy().reset_index(drop=True)
        unique_meta = patient_meta.drop_duplicates("emb_id", keep="first").copy()
        unique_ids = unique_meta["emb_id"].to_numpy(dtype=np.int64)
        expected_ids = np.arange(len(unique_ids), dtype=np.int64)
        if not np.array_equal(unique_ids, expected_ids):
            raise ValueError(
                f"patient {patient_id} emb_id values are not contiguous zero-based: "
                f"first values={unique_ids[:10].tolist()}"
            )

        if len(patient_meta) == entry.n_embedding_rows:
            # Original generator sorted the post-join metadata and embedded every row.
            mapped = patient_meta
            mode = "all_sorted_metadata_rows"
        elif len(unique_meta) == entry.n_embedding_rows:
            # Corrected generator embedded one row per true visit.
            mapped = unique_meta.reset_index(drop=True)
            mode = "one_row_per_unique_emb_id"
        else:
            raise ValueError(
                f"patient {patient_id}: embedding array has {entry.n_embedding_rows} "
                f"rows, metadata has {len(patient_meta)} rows and "
                f"{len(unique_meta)} unique emb_id values; neither layout matches"
            )

        mapped = mapped.copy().reset_index(drop=True)
        mapped["embedding_row"] = np.arange(len(mapped), dtype=np.int64)
        mapped["activation_row"] = mapped["embedding_row"]
        source_counts = patient_meta.groupby("emb_id", sort=False).size()
        mapped["source_meta_row_count"] = (
            mapped["emb_id"].map(source_counts).astype(np.int64)
        )
        mapped["embedding_row_count_for_emb_id"] = (
            mapped.groupby("emb_id", sort=False)["emb_id"].transform("size").astype(np.int64)
        )
        mapped["metadata_mapping_mode"] = mode
        mapped["split"] = split_lookup[patient_id]
        mapped["embedding_path"] = entry.path
        if deduplicate_repeated_emb_id:
            mapped["used_for_sae"] = ~mapped.duplicated("emb_id", keep="first")
        else:
            mapped["used_for_sae"] = True

        if verify_duplicate_embeddings and mode == "all_sorted_metadata_rows":
            duplicate_embedding_rows_checked += _verify_duplicate_embedding_rows(
                entry, mapped
            )
        mapped_parts.append(mapped)
        mapping_modes[patient_id] = mode

        if number % 10_000 == 0 or number == len(entries_by_id):
            LOGGER.info(
                "Indexed %d/%d patients against metadata",
                number,
                len(entries_by_id),
            )

    row_index = pd.concat(mapped_parts, ignore_index=True)
    row_index = row_index.sort_values(
        ["PatientID", "embedding_row"], kind="mergesort"
    ).reset_index(drop=True)
    row_index = row_index.rename(columns={"_source_meta_row": "source_meta_row"})

    first_columns = [
        "PatientID",
        "embedding_row",
        "activation_row",
        "emb_id",
        "enc_id",
        "date",
        "enc_summary",
        "split",
        "used_for_sae",
        "source_meta_row",
        "source_meta_row_count",
        "embedding_row_count_for_emb_id",
        "metadata_mapping_mode",
        "embedding_path",
    ]
    remaining = [column for column in row_index.columns if column not in first_columns]
    row_index = row_index[first_columns + remaining]

    unique_visit_index = row_index.drop_duplicates(
        ["PatientID", "emb_id"], keep="first"
    ).copy()
    unique_visit_index = unique_visit_index.rename(
        columns={"embedding_row": "first_embedding_row"}
    )

    model_rows_by_patient: Dict[str, np.ndarray] = {}
    for patient_id, group in row_index.groupby("PatientID", sort=False):
        rows = group.loc[group["used_for_sae"], "embedding_row"].to_numpy(
            dtype=np.int64
        )
        if rows.size == 0:
            raise RuntimeError(f"patient {patient_id} has no rows selected for the SAE")
        model_rows_by_patient[str(patient_id)] = rows

    atomic_dataframe_csv(
        row_index, output_dir / "embedding_row_index.csv", sep=INDEX_SEPARATOR
    )
    atomic_dataframe_csv(
        unique_visit_index,
        output_dir / "unique_visit_index.csv",
        sep=INDEX_SEPARATOR,
    )

    mode_counts = pd.Series(mapping_modes).value_counts().to_dict()
    qc: Dict[str, object] = {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_separator": metadata_separator,
        "metadata_rows_total": int(len(meta)),
        "metadata_rows_for_valid_embedding_patients": int(len(relevant)),
        "indexed_embedding_rows": int(len(row_index)),
        "unique_true_visits": int(len(unique_visit_index)),
        "rows_used_for_sae": int(row_index["used_for_sae"].sum()),
        "repeated_embedding_rows_not_used_for_sae": int(
            len(row_index) - row_index["used_for_sae"].sum()
        ),
        "duplicate_embedding_rows_checked": duplicate_embedding_rows_checked,
        "deduplicate_repeated_emb_id_for_model": deduplicate_repeated_emb_id,
        "mapping_mode_patient_counts": {
            str(key): int(value) for key, value in mode_counts.items()
        },
    }
    atomic_json_dump(qc, output_dir / "metadata_index_qc.json")
    return row_index, unique_visit_index, model_rows_by_patient, qc


# -----------------------------------------------------------------------------
# Streaming visit dataset
# -----------------------------------------------------------------------------


class PatientEmbeddingDataset(IterableDataset):
    """Stream embeddings without losing their immutable source-row mapping."""

    def __init__(
        self,
        entries: Sequence[PatientFile],
        model_rows_by_patient: Mapping[str, np.ndarray],
        d_model: int,
        seed: int,
        shuffle: bool,
        shuffle_buffer_rows: int,
        max_visits_per_patient: Optional[int],
    ) -> None:
        super().__init__()
        self.entries = list(entries)
        self.model_rows_by_patient = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in model_rows_by_patient.items()
        }
        self.d_model = d_model
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer_rows = max(int(shuffle_buffer_rows), 1)
        self.max_visits_per_patient = max_visits_per_patient
        self._epoch = mp.Value("q", 0)

    @property
    def num_examples(self) -> int:
        total = 0
        for entry in self.entries:
            count = len(self.model_rows_by_patient[entry.patient_id])
            if self.max_visits_per_patient is not None:
                count = min(count, self.max_visits_per_patient)
            total += count
        return total

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __iter__(self) -> Iterable[torch.Tensor]:
        with self._epoch.get_lock():
            epoch = int(self._epoch.value)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        n_workers = 1 if worker is None else worker.num_workers
        entries = self.entries.copy()
        if self.shuffle:
            random.Random(self.seed + 1_000_003 * epoch).shuffle(entries)
        entries = entries[worker_id::n_workers]
        rng = np.random.default_rng(
            self.seed + 10_000_019 * epoch + 97_409 * worker_id
        )

        if not self.shuffle:
            for entry in entries:
                array = np.load(entry.path, mmap_mode="r", allow_pickle=False)
                array_2d = array.reshape(1, self.d_model) if array.ndim == 1 else array
                rows = self.model_rows_by_patient[entry.patient_id]
                for start in range(0, len(rows), 1024):
                    block = np.asarray(array_2d[rows[start : start + 1024]], dtype=np.float32)
                    for row in block:
                        yield torch.from_numpy(row.copy())
            return

        buffer = np.empty(
            (self.shuffle_buffer_rows, self.d_model), dtype=np.float32
        )
        filled = 0

        def flush(n_rows: int) -> Iterable[torch.Tensor]:
            for index in rng.permutation(n_rows):
                yield torch.from_numpy(buffer[int(index)].copy())

        for entry in entries:
            array = np.load(entry.path, mmap_mode="r", allow_pickle=False)
            array_2d = array.reshape(1, self.d_model) if array.ndim == 1 else array
            rows = self.model_rows_by_patient[entry.patient_id].copy()
            if (
                self.max_visits_per_patient is not None
                and len(rows) > self.max_visits_per_patient
            ):
                rows = rng.choice(
                    rows, size=self.max_visits_per_patient, replace=False
                )
            else:
                rng.shuffle(rows)
            cursor = 0
            while cursor < len(rows):
                take = min(self.shuffle_buffer_rows - filled, len(rows) - cursor)
                buffer[filled : filled + take] = np.asarray(
                    array_2d[rows[cursor : cursor + take]], dtype=np.float32
                )
                cursor += take
                filled += take
                if filled == self.shuffle_buffer_rows:
                    yield from flush(filled)
                    filled = 0
        if filled:
            yield from flush(filled)


def make_loader(
    dataset: PatientEmbeddingDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs: Dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def compute_training_center(
    entries: Sequence[PatientFile],
    model_rows_by_patient: Mapping[str, np.ndarray],
    d_model: int,
) -> torch.Tensor:
    """Compute the exact training mean with bounded memory."""
    total = np.zeros(d_model, dtype=np.float64)
    count = 0
    for number, entry in enumerate(entries, start=1):
        array = np.load(entry.path, mmap_mode="r", allow_pickle=False)
        array_2d = array.reshape(1, d_model) if array.ndim == 1 else array
        rows = model_rows_by_patient[entry.patient_id]
        for start in range(0, len(rows), 2048):
            block = np.asarray(array_2d[rows[start : start + 2048]], dtype=np.float32)
            total += block.sum(axis=0, dtype=np.float64)
            count += int(block.shape[0])
        if number % 100 == 0 or number == len(entries):
            LOGGER.info("Computed center from %d/%d training patients", number, len(entries))
    if count == 0:
        raise RuntimeError("cannot initialize SAE from zero training examples")
    return torch.from_numpy((total / count).astype(np.float32))


# -----------------------------------------------------------------------------
# Top-K SAE and metrics
# -----------------------------------------------------------------------------


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, k: int) -> None:
        super().__init__()
        if not 0 < k <= d_hidden:
            raise ValueError(f"k must be in [1, {d_hidden}], found {k}")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    @torch.no_grad()
    def initialize_(self, center: torch.Tensor) -> None:
        if tuple(center.shape) != (self.d_model,):
            raise ValueError("training center has the wrong shape")
        self.b_dec.copy_(center.to(self.b_dec))
        nn.init.xavier_normal_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.copy_(self.decoder.weight.T)
        nn.init.zeros_(self.encoder.bias)

    def encode_topk(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dense = F.relu(self.encoder(x - self.b_dec))
        values, indices = torch.topk(dense, k=self.k, dim=-1, sorted=True)
        return values, indices

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        values, indices = self.encode_topk(x)
        sparse = torch.zeros(
            (x.shape[0], self.d_hidden), dtype=values.dtype, device=x.device
        )
        return sparse.scatter(1, indices, values)

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        return self.decoder(sparse) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sparse = self.encode(x)
        return sparse, self.decode(sparse)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def project_decoder_gradient_(self) -> None:
        gradient = self.decoder.weight.grad
        if gradient is None:
            return
        directions = F.normalize(self.decoder.weight, dim=0)
        parallel = (gradient * directions).sum(dim=0, keepdim=True)
        gradient.sub_(parallel * directions)


def reconstruction_loss_per_visit(
    original: torch.Tensor, reconstruction: torch.Tensor
) -> torch.Tensor:
    """mean_i sum_dimension (x - x_hat)^2, computed in float32."""
    return (
        (original.float() - reconstruction.float()).square().sum(dim=1).mean()
    )


def amp_configuration(device: torch.device) -> Tuple[bool, torch.dtype, bool]:
    if device.type != "cuda":
        return False, torch.float32, False
    use_bfloat16 = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    return True, dtype, dtype == torch.float16


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def evaluate_model(
    model: TopKSparseAutoencoder,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    squared_error_sum = 0.0
    input_square_sum = 0.0
    input_sum = torch.zeros(model.d_model, dtype=torch.float64)
    cosine_sum = 0.0
    active_sum = 0.0
    n_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with autocast_context(device, use_amp, amp_dtype):
                sparse, reconstruction = model(batch)
            batch_f32 = batch.float()
            reconstruction_f32 = reconstruction.float()
            residual = batch_f32 - reconstruction_f32
            squared_error_sum += float(residual.square().sum().item())
            input_square_sum += float(batch_f32.square().sum().item())
            input_sum += batch_f32.sum(dim=0).double().cpu()
            cosine_sum += float(
                F.cosine_similarity(reconstruction_f32, batch_f32, dim=1).sum().item()
            )
            active_sum += float((sparse > 0).sum().item())
            n_examples += int(batch.shape[0])
    if n_examples == 0:
        raise RuntimeError("evaluation loader yielded zero examples")
    total_variance = input_square_sum - float(input_sum.square().sum().item()) / n_examples
    return {
        "n_examples": float(n_examples),
        "reconstruction_error": squared_error_sum / n_examples,
        "mse_per_dimension": squared_error_sum / (n_examples * model.d_model),
        "fvu": squared_error_sum / max(total_variance, 1e-12),
        "relative_squared_error": squared_error_sum / max(input_square_sum, 1e-12),
        "cosine_similarity": cosine_sum / n_examples,
        "mean_active_features": active_sum / n_examples,
    }


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    final_lr_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_fraction + (1.0 - final_lr_fraction) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_k(
    k: int,
    d_model: int,
    d_hidden: int,
    center: torch.Tensor,
    train_dataset: PatientEmbeddingDataset,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    manifest_hash: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_fraction: float,
    final_lr_fraction: float,
    gradient_clip: float,
    patience: int,
    min_delta: float,
) -> Tuple[TopKSparseAutoencoder, Dict[str, object], List[Dict[str, object]]]:
    """Train one independent k and restore its best validation checkpoint."""
    k_dir = output_dir / f"k_{k:04d}"
    k_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)  # Identical initialization for every k.
    model = TopKSparseAutoencoder(d_model=d_model, d_hidden=d_hidden, k=k).to(device)
    model.initialize_(center.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps_per_epoch = max(math.ceil(train_dataset.num_examples / batch_size), 1)
    total_steps = steps_per_epoch * epochs
    scheduler = make_scheduler(
        optimizer, total_steps, warmup_fraction, final_lr_fraction
    )
    use_amp, amp_dtype, use_scaler = amp_configuration(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_error = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        model.train()
        train_error_sum = 0.0
        train_active_sum = 0.0
        train_examples = 0
        epoch_started = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp, amp_dtype):
                sparse, reconstruction = model(batch)
                loss = reconstruction_loss_per_visit(batch, reconstruction)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                model.project_decoder_gradient_()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                model.project_decoder_gradient_()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            model.normalize_decoder_()
            scheduler.step()

            n_batch = int(batch.shape[0])
            train_error_sum += float(loss.detach().item()) * n_batch
            train_active_sum += float((sparse.detach() > 0).sum().item())
            train_examples += n_batch
            if step % 100 == 0:
                LOGGER.info(
                    "k=%d | epoch=%d/%d | step=%d/%d | train recon=%.6f",
                    k,
                    epoch,
                    epochs,
                    step,
                    steps_per_epoch,
                    train_error_sum / train_examples,
                )

        validation = evaluate_model(
            model, validation_loader, device, use_amp, amp_dtype
        )
        train_error = train_error_sum / max(train_examples, 1)
        record: Dict[str, object] = {
            "k": k,
            "epoch": epoch,
            "train_reconstruction_error": train_error,
            "train_mean_active_features": train_active_sum / max(train_examples, 1),
            "validation_reconstruction_error": validation["reconstruction_error"],
            "validation_mse_per_dimension": validation["mse_per_dimension"],
            "validation_fvu": validation["fvu"],
            "validation_relative_squared_error": validation[
                "relative_squared_error"
            ],
            "validation_cosine_similarity": validation["cosine_similarity"],
            "validation_mean_active_features": validation["mean_active_features"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(record)
        atomic_csv_rows(history, k_dir / "epoch_metrics.csv")
        LOGGER.info(
            "k=%d | epoch=%d/%d | train recon=%.6f | val recon=%.6f | "
            "val active=%.2f | %.1fs",
            k,
            epoch,
            epochs,
            train_error,
            validation["reconstruction_error"],
            validation["mean_active_features"],
            record["epoch_seconds"],
        )

        current_error = float(validation["reconstruction_error"])
        if current_error < best_error - min_delta:
            best_error = current_error
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "k": k,
                    "d_model": d_model,
                    "d_hidden": d_hidden,
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "manifest_signature": manifest_hash,
                    "seed": seed,
                },
                k_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info(
                    "k=%d early stopping after epoch %d; best epoch=%d",
                    k,
                    epoch,
                    best_epoch,
                )
                break

    best_path = k_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"best checkpoint was not created for k={k}")
    checkpoint = load_checkpoint(best_path, device)
    if checkpoint.get("manifest_signature") != manifest_hash:
        raise RuntimeError("best checkpoint manifest does not match current data")
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation = evaluate_model(
        model, validation_loader, device, use_amp, amp_dtype
    )
    best_history = next(item for item in history if int(item["epoch"]) == best_epoch)
    result: Dict[str, object] = {
        "k": k,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train_reconstruction_error_at_best_epoch": best_history[
            "train_reconstruction_error"
        ],
        "validation_reconstruction_error": final_validation[
            "reconstruction_error"
        ],
        "validation_mse_per_dimension": final_validation["mse_per_dimension"],
        "validation_fvu": final_validation["fvu"],
        "validation_relative_squared_error": final_validation[
            "relative_squared_error"
        ],
        "validation_cosine_similarity": final_validation["cosine_similarity"],
        "validation_mean_active_features": final_validation[
            "mean_active_features"
        ],
        "validation_n_visits": int(final_validation["n_examples"]),
        "train_n_visits_per_epoch": train_dataset.num_examples,
        "training_seconds": time.time() - started,
        "checkpoint": str(best_path.resolve()),
    }
    return model, result, history


# -----------------------------------------------------------------------------
# Storage-efficient activation export with exact visit indices
# -----------------------------------------------------------------------------


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def export_sparse_activations(
    model: TopKSparseAutoencoder,
    entries: Sequence[PatientFile],
    row_index: pd.DataFrame,
    split_name: str,
    k_dir: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    """Export only Top-K indices/values; do not create huge dense matrices."""
    model.eval()
    use_amp, amp_dtype, _ = amp_configuration(device)
    activation_root = k_dir / "activations" / split_name
    index_parts: List[pd.DataFrame] = []
    manifest_files: List[Dict[str, object]] = []

    for number, entry in enumerate(sorted(entries, key=lambda item: item.patient_id), start=1):
        patient_index = row_index.loc[
            row_index["PatientID"] == entry.patient_id
        ].sort_values("embedding_row", kind="mergesort")
        expected_rows = np.arange(entry.n_embedding_rows, dtype=np.int64)
        actual_rows = patient_index["embedding_row"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_rows, expected_rows):
            raise RuntimeError(
                f"activation index is not complete/ordered for patient {entry.patient_id}"
            )

        source = np.load(entry.path, mmap_mode="r", allow_pickle=False)
        source_2d = source.reshape(1, model.d_model) if source.ndim == 1 else source
        values = np.empty((entry.n_embedding_rows, model.k), dtype=np.float32)
        indices = np.empty((entry.n_embedding_rows, model.k), dtype=np.int32)
        with torch.no_grad():
            for start in range(0, entry.n_embedding_rows, batch_size):
                stop = min(start + batch_size, entry.n_embedding_rows)
                batch_np = np.asarray(source_2d[start:stop], dtype=np.float32)
                batch = torch.from_numpy(batch_np.copy()).to(device, non_blocking=True)
                with autocast_context(device, use_amp, amp_dtype):
                    top_values, top_indices = model.encode_topk(batch)
                values[start:stop] = top_values.float().cpu().numpy()
                indices[start:stop] = top_indices.int().cpu().numpy()

        relative_file = (
            Path("activations") / split_name / entry.patient_id / "sae_topk.npz"
        )
        output_file = k_dir / relative_file
        _atomic_save_npz(
            output_file,
            topk_indices=indices,
            topk_values=values,
            embedding_row=actual_rows,
            emb_id=patient_index["emb_id"].to_numpy(dtype=np.int64),
        )
        patient_index = patient_index.copy()
        patient_index["activation_file"] = str(relative_file)
        patient_index["activation_storage"] = "topk_indices_and_values"
        patient_index["k"] = model.k
        index_parts.append(patient_index)
        manifest_files.append(
            {
                "patient_id": entry.patient_id,
                "path": str(relative_file),
                "shape": [entry.n_embedding_rows, model.k],
                "row_mapping": "array row equals embedding_row equals activation_row",
            }
        )
        if number % 100 == 0 or number == len(entries):
            LOGGER.info(
                "k=%d | exported %s activations for %d/%d patients",
                model.k,
                split_name,
                number,
                len(entries),
            )

    activation_index = pd.concat(index_parts, ignore_index=True)
    atomic_dataframe_csv(
        activation_index,
        k_dir / f"{split_name}_activation_index.csv",
        sep=INDEX_SEPARATOR,
    )
    atomic_json_dump(
        {
            "k": model.k,
            "d_hidden": model.d_hidden,
            "split": split_name,
            "storage": {
                "topk_indices": "int32, shape (n_embedding_rows, k)",
                "topk_values": "float32, shape (n_embedding_rows, k)",
                "embedding_row": "int64; exact row in the source embedding array",
                "emb_id": "int64; original patient-local visit ID",
            },
            "files": manifest_files,
        },
        k_dir / f"{split_name}_activation_manifest.json",
    )


def plot_k_curve(results: Sequence[Mapping[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; CSV saved but plot was not created")
        return
    ordered = sorted(results, key=lambda row: int(row["k"]))
    x = [int(row["k"]) for row in ordered]
    y = [float(row["validation_reconstruction_error"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.plot(x, y, marker="o", linewidth=1.8)
    axis.set_xscale("log")
    axis.set_xlabel("Top-K active features (k; log scale)")
    axis.set_ylabel("Validation reconstruction error\nmean squared L2 error per visit")
    axis.set_title("Top-K SAE reconstruction curve")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    temporary = path.with_name(path.name + ".tmp.png")
    figure.savefig(temporary, dpi=200)
    plt.close(figure)
    os.replace(temporary, path)


# -----------------------------------------------------------------------------
# Command line and orchestration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Defaults to DATA_DIR/meta_v3_all.csv.",
    )
    parser.add_argument("--metadata-separator", default="$" )
    parser.add_argument(
        "--embedding-filename",
        default="{patient_id}.npy",
        help="Filename template inside each patient folder, or 'auto'.",
    )
    parser.add_argument("--output-dir", default="topk_sae_k_sweep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--d-hidden", type=int, default=16384)
    parser.add_argument(
        "--ks",
        type=parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--final-lr-fraction", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--num-workers-train", type=int, default=4)
    parser.add_argument("--num-workers-eval", type=int, default=2)
    parser.add_argument("--shuffle-buffer-rows", type=int, default=4096)
    parser.add_argument(
        "--max-train-visits-per-patient",
        type=int,
        default=128,
        help="Fresh per-epoch cap; use 0 for every selected visit.",
    )
    parser.add_argument(
        "--deduplicate-repeated-emb-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use one embedding per true PatientID+emb_id visit for fitting and "
            "evaluation. The sweep determines these rows from a minimal index."
        ),
    )
    parser.add_argument(
        "--verify-duplicate-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-unit-norm-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Qwen visit embeddings were L2-normalized during generation.",
    )
    parser.add_argument("--unit-norm-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--full-value-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-invalid-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default is fail-fast because silent loss can break metadata indexing.",
    )
    parser.add_argument(
        "--skip-patients-missing-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip embedding directories with no PatientID rows in metadata and "
            "record them in patients_missing_metadata.csv. Default is fail-fast."
        ),
    )
    parser.add_argument(
        "--export-split",
        choices=("none", "validation", "test", "train-validation", "all"),
        default="none",
        help=(
            "Export compact per-patient Top-K arrays from each best model and "
            "build the full note/encounter index. The default 'none' defers both; "
            "'train-validation' exports both development splits without test."
        ),
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Retrain k values already present in the results CSV.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Write QC and split manifests without training. The full note index "
            "is written only if --export-split is not 'none'."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.d_model <= 0 or args.d_hidden <= 0:
        raise ValueError("d-model and d-hidden must be positive")
    if any(k > args.d_hidden for k in args.ks):
        raise ValueError("every k must be <= d-hidden")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning-rate and gradient-clip must be positive")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup-fraction must be in [0, 1)")
    if not 0 <= args.final_lr_fraction <= 1:
        raise ValueError("final-lr-fraction must be in [0, 1]")
    if args.patience <= 0 or args.min_delta < 0:
        raise ValueError("patience must be positive and min-delta nonnegative")
    if args.num_workers_train < 0 or args.num_workers_eval < 0:
        raise ValueError("worker counts cannot be negative")
    if args.shuffle_buffer_rows <= 0:
        raise ValueError("shuffle-buffer-rows must be positive")
    if args.unit_norm_tolerance <= 0:
        raise ValueError("unit-norm-tolerance must be positive")


def read_existing_results(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict("records")


def main() -> None:
    args = parse_args()
    validate_args(args)
    data_dir = Path(args.data_dir).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_file).expanduser().resolve()
        if args.metadata_file is not None
        else data_dir / "meta_v3_all.csv"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_logging(output_dir)
    set_global_seed(args.seed)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    LOGGER.info("Using device %s", device)
    LOGGER.info("K sweep: %s", list(args.ks))

    entries, skipped, embedding_qc = discover_patient_files(
        data_dir=data_dir,
        embedding_filename=args.embedding_filename,
        d_model=args.d_model,
        require_unit_norm=args.require_unit_norm_inputs,
        unit_norm_tolerance=args.unit_norm_tolerance,
        full_value_validation=args.full_value_validation,
        skip_invalid_files=args.skip_invalid_files,
    )
    if skipped:
        atomic_csv_rows(skipped, output_dir / "skipped_files.csv")

    discovered_embedding_patients = len(entries)
    (
        model_rows_by_patient,
        minimal_metadata_qc,
        missing_metadata_patient_ids,
    ) = build_minimal_model_row_map(
        metadata_path=metadata_path,
        metadata_separator=args.metadata_separator,
        entries=entries,
        output_dir=output_dir,
        deduplicate_repeated_emb_id=args.deduplicate_repeated_emb_id,
        skip_patients_missing_metadata=args.skip_patients_missing_metadata,
    )
    if missing_metadata_patient_ids:
        missing_set = set(missing_metadata_patient_ids)
        entries = [entry for entry in entries if entry.patient_id not in missing_set]
    embedding_qc["n_valid_patients_after_metadata_match"] = len(entries)
    embedding_qc["n_embedding_rows_after_metadata_match"] = sum(
        entry.n_embedding_rows for entry in entries
    )
    embedding_qc["n_patients_skipped_missing_metadata"] = (
        discovered_embedding_patients - len(entries)
    )

    # Split only the retained patients so every split count and manifest entry
    # corresponds to an embedding that can be rematched to metadata.
    train_entries, validation_entries, test_entries = stable_patient_split(
        entries,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        args.split_seed,
    )
    split_lookup = {
        **{entry.patient_id: "train" for entry in train_entries},
        **{entry.patient_id: "validation" for entry in validation_entries},
        **{entry.patient_id: "test" for entry in test_entries},
    }
    row_index: Optional[pd.DataFrame] = None
    full_metadata_qc: Optional[Dict[str, object]] = None
    if args.export_split != "none":
        LOGGER.info(
            "Activation export requested; building the full note/encounter index"
        )
        (
            row_index,
            _,
            indexed_model_rows,
            full_metadata_qc,
        ) = build_embedding_row_index(
            metadata_path=metadata_path,
            metadata_separator=args.metadata_separator,
            entries=entries,
            split_lookup=split_lookup,
            output_dir=output_dir,
            deduplicate_repeated_emb_id=args.deduplicate_repeated_emb_id,
            verify_duplicate_embeddings=args.verify_duplicate_embeddings,
        )
        for patient_id, minimal_rows in model_rows_by_patient.items():
            indexed_rows = indexed_model_rows.get(patient_id)
            if indexed_rows is None or not np.array_equal(
                minimal_rows, indexed_rows
            ):
                raise RuntimeError(
                    f"minimal and full row mappings disagree for patient {patient_id}"
                )
    signature = manifest_signature(entries, metadata_path)
    data_manifest = {
        "manifest_signature": signature,
        "embedding_qc": embedding_qc,
        "metadata_qc": (
            full_metadata_qc
            if full_metadata_qc is not None
            else minimal_metadata_qc
        ),
        "metadata_qc_level": (
            "full_activation_index"
            if full_metadata_qc is not None
            else "minimal_training_row_map"
        ),
        "minimal_metadata_qc": minimal_metadata_qc,
        "full_metadata_index_qc": full_metadata_qc,
        "split_seed": args.split_seed,
        "split_fractions": {
            "train": args.train_fraction,
            "validation": args.val_fraction,
            "test": args.test_fraction,
        },
        "split_patient_counts": {
            "train": len(train_entries),
            "validation": len(validation_entries),
            "test": len(test_entries),
        },
        "files": [
            {**asdict(entry), "split": split_lookup[entry.patient_id]}
            for entry in entries
        ],
    }
    atomic_json_dump(data_manifest, output_dir / "data_manifest.json")
    run_config = vars(args).copy()
    run_config["ks"] = list(args.ks)
    run_config["resolved_data_dir"] = str(data_dir)
    run_config["resolved_metadata_file"] = str(metadata_path)
    run_config["resolved_output_dir"] = str(output_dir)
    run_config["device"] = str(device)
    atomic_json_dump(run_config, output_dir / "run_config.json")
    LOGGER.info(
        "Split: train=%d patients, validation=%d, test=%d",
        len(train_entries),
        len(validation_entries),
        len(test_entries),
    )
    LOGGER.info(
        "Prepared %d model rows with the minimal metadata map",
        sum(len(model_rows_by_patient[item.patient_id]) for item in entries),
    )
    if row_index is not None:
        LOGGER.info("Full activation index contains %d raw embedding rows", len(row_index))
    if args.validate_only:
        LOGGER.info("Validation-only run complete")
        return

    train_dataset = PatientEmbeddingDataset(
        train_entries,
        model_rows_by_patient,
        d_model=args.d_model,
        seed=args.seed,
        shuffle=True,
        shuffle_buffer_rows=args.shuffle_buffer_rows,
        max_visits_per_patient=(
            None
            if args.max_train_visits_per_patient <= 0
            else args.max_train_visits_per_patient
        ),
    )
    validation_dataset = PatientEmbeddingDataset(
        validation_entries,
        model_rows_by_patient,
        d_model=args.d_model,
        seed=args.seed + 1,
        shuffle=False,
        shuffle_buffer_rows=args.shuffle_buffer_rows,
        max_visits_per_patient=None,
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers_train, pin_memory
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.num_workers_eval, pin_memory
    )
    center = compute_training_center(
        train_entries, model_rows_by_patient, args.d_model
    )

    results_path = output_dir / "k_vs_reconstruction_error.csv"
    history_path = output_dir / "k_sweep_epoch_history.csv"
    results = read_existing_results(results_path)
    completed = {int(row["k"]) for row in results}
    global_history: List[Dict[str, object]] = []
    if history_path.is_file():
        global_history = pd.read_csv(history_path).to_dict("records")

    split_entries = {
        "train": train_entries,
        "validation": validation_entries,
        "test": test_entries,
    }
    for k in args.ks:
        k_dir = output_dir / f"k_{k:04d}"
        best_path = k_dir / "best.pt"
        already_complete = k in completed and best_path.is_file()
        if already_complete and not args.rerun_completed:
            LOGGER.info("Skipping completed k=%d", k)
            if args.export_split != "none":
                if args.export_split == "all":
                    requested_splits = tuple(split_entries)
                elif args.export_split == "train-validation":
                    requested_splits = ("train", "validation")
                else:
                    requested_splits = (args.export_split,)
                missing_export = any(
                    not (k_dir / f"{name}_activation_manifest.json").is_file()
                    for name in requested_splits
                )
                if missing_export:
                    if row_index is None:
                        raise RuntimeError(
                            "activation export requires the full metadata row index"
                        )
                    checkpoint = load_checkpoint(best_path, device)
                    if checkpoint.get("manifest_signature") != signature:
                        raise RuntimeError(
                            f"completed k={k} checkpoint belongs to different data"
                        )
                    model = TopKSparseAutoencoder(
                        args.d_model, args.d_hidden, k
                    ).to(device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    for split_name in requested_splits:
                        export_sparse_activations(
                            model,
                            split_entries[split_name],
                            row_index,
                            split_name,
                            k_dir,
                            device,
                            args.batch_size,
                        )
                    del model, checkpoint
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            continue

        # Remove stale rows for a deliberately rerun k before appending replacements.
        results = [row for row in results if int(row["k"]) != k]
        global_history = [row for row in global_history if int(row["k"]) != k]
        LOGGER.info("Starting independent Top-K SAE for k=%d", k)
        model, result, k_history = train_one_k(
            k=k,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            center=center,
            train_dataset=train_dataset,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            output_dir=output_dir,
            manifest_hash=signature,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_fraction=args.warmup_fraction,
            final_lr_fraction=args.final_lr_fraction,
            gradient_clip=args.gradient_clip,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        results.append(result)
        results = sorted(results, key=lambda row: int(row["k"]))
        global_history.extend(k_history)
        global_history = sorted(
            global_history, key=lambda row: (int(row["k"]), int(row["epoch"]))
        )
        atomic_csv_rows(results, results_path)
        atomic_csv_rows(global_history, history_path)
        plot_k_curve(results, output_dir / "k_vs_reconstruction_error.png")

        if args.export_split != "none":
            if row_index is None:
                raise RuntimeError(
                    "activation export requires the full metadata row index"
                )
            if args.export_split == "all":
                requested_splits = tuple(split_entries)
            elif args.export_split == "train-validation":
                requested_splits = ("train", "validation")
            else:
                requested_splits = (args.export_split,)
            for split_name in requested_splits:
                export_sparse_activations(
                    model,
                    split_entries[split_name],
                    row_index,
                    split_name,
                    k_dir,
                    device,
                    args.batch_size,
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    LOGGER.info("K sweep complete; results -> %s", results_path)


if __name__ == "__main__":
    main()
