"""Max-pool exported Top-K SAE activations over the CKD predictor window.

This module is designed for the sparse activation export produced by
``topk_sae_k_sweep.py``::

    <selected_k_dir>/activations/<split>/<PatientID>/sae_topk.npz

Each NPZ must contain ``topk_indices``, ``topk_values``, ``embedding_row``,
and ``emb_id``. Each split's exported activation-index CSV supplies the exact
``activation_row`` for every ``PatientID + emb_id``. The selected NPZ row is
then checked against its stored ``emb_id`` before pooling.

Notebook example
----------------
from maxpool_sae_activations import maxpool_sae_activations

X_sae, sae_index_df, sae_visit_index_df = maxpool_sae_activations(
    cohort_df=cohort_df,
    pooling_index_df=pooling_index_df,
    selected_k_dir="/path/to/topk_sae_x2_k_sweep/k_0064",
)

The predictor window is inclusive:
``lookback_start_date <= date <= first_stage3_date``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]

DEFAULT_SPLIT_DIRECTORY_MAP = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
}

REQUIRED_NPZ_KEYS = {
    "topk_indices",
    "topk_values",
    "embedding_row",
    "emb_id",
}

# ``topk_sae_k_sweep.py`` writes all row-index files with this separator,
# despite retaining the conventional ``.csv`` filename suffix.  In
# particular, ``enc_summary`` can contain commas, so using pandas' default
# comma separator corrupts the apparent field count.
ACTIVATION_INDEX_SEPARATOR = "$"


def _clean_patient_ids(frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    frame = frame.copy()
    if frame["PatientID"].isna().any():
        raise ValueError(f"{frame_name} contains missing PatientID values.")

    frame["PatientID"] = frame["PatientID"].astype(str).str.strip()
    if frame["PatientID"].eq("").any():
        raise ValueError(f"{frame_name} contains empty PatientID values.")
    return frame


def _prepare_pooling_rows(
    cohort_df: pd.DataFrame,
    pooling_index_df: pd.DataFrame,
    split_column: str,
    split_directory_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    required_cohort_columns = {
        "PatientID",
        "lookback_start_date",
        "first_stage3_date",
        split_column,
    }
    required_pooling_columns = {
        "PatientID",
        "date",
        "emb_id",
        "enc_id",
        "CKD_stage_numeric",
    }

    missing_cohort = sorted(required_cohort_columns - set(cohort_df.columns))
    missing_pooling = sorted(required_pooling_columns - set(pooling_index_df.columns))
    if missing_cohort:
        raise KeyError(f"cohort_df is missing columns: {missing_cohort}")
    if missing_pooling:
        raise KeyError(f"pooling_index_df is missing columns: {missing_pooling}")

    cohort = _clean_patient_ids(
        cohort_df.reset_index(drop=True),
        "cohort_df",
    )
    pooling = _clean_patient_ids(
        pooling_index_df.reset_index(drop=True),
        "pooling_index_df",
    )

    duplicate_patients = cohort.loc[
        cohort["PatientID"].duplicated(keep=False), "PatientID"
    ].unique()
    if len(duplicate_patients) > 0:
        raise ValueError(
            "cohort_df must contain exactly one row per patient. "
            f"Duplicated examples: {duplicate_patients[:10].tolist()}"
        )

    cohort["_cohort_row"] = np.arange(len(cohort), dtype=np.int64)
    cohort["lookback_start_date"] = pd.to_datetime(
        cohort["lookback_start_date"], errors="coerce"
    ).dt.normalize()
    cohort["first_stage3_date"] = pd.to_datetime(
        cohort["first_stage3_date"], errors="coerce"
    ).dt.normalize()
    pooling["date"] = pd.to_datetime(pooling["date"], errors="coerce").dt.normalize()

    if cohort[["lookback_start_date", "first_stage3_date"]].isna().any().any():
        raise ValueError(
            "cohort_df contains an unusable lookback_start_date or "
            "first_stage3_date."
        )
    if pooling["date"].isna().any():
        raise ValueError("pooling_index_df contains unusable date values.")
    if (cohort["lookback_start_date"] > cohort["first_stage3_date"]).any():
        raise ValueError(
            "At least one lookback_start_date occurs after first_stage3_date."
        )

    if cohort[split_column].isna().any():
        raise ValueError(f"cohort_df contains missing {split_column} values.")
    normalized_split = cohort[split_column].astype(str).str.strip().str.lower()
    unknown_splits = sorted(set(normalized_split) - set(split_directory_map))
    if unknown_splits:
        raise ValueError(
            f"Unrecognized {split_column} values: {unknown_splits}. "
            f"Accepted values: {sorted(split_directory_map)}"
        )
    cohort["_activation_split"] = normalized_split.map(split_directory_map)

    cohort_patients = set(cohort["PatientID"])
    pooling_patients = set(pooling["PatientID"])
    missing_from_pooling = sorted(cohort_patients - pooling_patients)
    extra_in_pooling = sorted(pooling_patients - cohort_patients)
    if missing_from_pooling or extra_in_pooling:
        raise ValueError(
            "Patient sets do not agree between cohort_df and pooling_index_df. "
            f"Missing from pooling_index_df: {missing_from_pooling[:10]}; "
            f"extra in pooling_index_df: {extra_in_pooling[:10]}."
        )

    pooling = pooling.drop(
        columns=[
            "lookback_start_date",
            "first_stage3_date",
            "_cohort_row",
            "_activation_split",
        ],
        errors="ignore",
    )
    pooling = pooling.merge(
        cohort[
            [
                "PatientID",
                "lookback_start_date",
                "first_stage3_date",
                "_cohort_row",
                "_activation_split",
            ]
        ],
        on="PatientID",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    numeric_emb_id = pd.to_numeric(pooling["emb_id"], errors="coerce")
    if numeric_emb_id.isna().any():
        examples = pooling.loc[
            numeric_emb_id.isna(), ["PatientID", "emb_id"]
        ].head(10)
        raise ValueError(
            "Some emb_id values are missing or nonnumeric:\n"
            f"{examples.to_string(index=False)}"
        )

    numeric_values = numeric_emb_id.to_numpy()
    if not np.all(numeric_values == np.floor(numeric_values)):
        raise ValueError("Every emb_id must be an integer.")
    if (numeric_values < 0).any():
        raise ValueError("Every emb_id must be zero or greater.")
    pooling["emb_id"] = numeric_emb_id.astype(np.int64)

    inside_window = (
        pooling["date"].ge(pooling["lookback_start_date"])
        & pooling["date"].le(pooling["first_stage3_date"])
    )
    n_outside_window = int((~inside_window).sum())
    visits = pooling.loc[inside_window].copy()

    duplicate_mask = visits.duplicated(["PatientID", "emb_id"], keep=False)
    duplicate_rows = visits.loc[duplicate_mask]
    if not duplicate_rows.empty:
        consistency = (
            duplicate_rows.groupby(["PatientID", "emb_id"], as_index=False)
            .agg(
                n_dates=("date", lambda x: x.nunique(dropna=False)),
                n_enc_ids=(
                    "enc_id",
                    lambda x: (
                        x.astype("string")
                        .fillna("<MISSING>")
                        .nunique(dropna=False)
                    ),
                ),
            )
        )
        conflicts = consistency.loc[
            (consistency["n_dates"] > 1) | (consistency["n_enc_ids"] > 1)
        ]
        if not conflicts.empty:
            raise ValueError(
                "The same (PatientID, emb_id) maps to conflicting dates or "
                "enc_id values:\n"
                f"{conflicts.head(10).to_string(index=False)}"
            )

    n_before_deduplication = len(visits)
    visits = visits.drop_duplicates(["PatientID", "emb_id"], keep="first")
    n_duplicate_rows_removed = n_before_deduplication - len(visits)

    visits = (
        visits.sort_values(
            ["_cohort_row", "date", "emb_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    visits["sae_pool_position"] = (
        visits.groupby("PatientID", sort=False).cumcount().astype(np.int64)
    )

    summary = (
        visits.groupby("PatientID", as_index=False, sort=False)
        .agg(
            n_embedding_rows_pooled=("emb_id", "size"),
            n_unique_visit_dates_pooled=("date", "nunique"),
            first_pooled_date=("date", "min"),
            last_pooled_date=("date", "max"),
        )
    )

    patients_without_rows = sorted(cohort_patients - set(summary["PatientID"]))
    if patients_without_rows:
        raise ValueError(
            "Some cohort patients have no embeddings in the inclusive window: "
            f"{patients_without_rows[:10]}"
        )

    stage_numeric = pd.to_numeric(visits["CKD_stage_numeric"], errors="coerce")
    is_stage3_on_index = (
        visits["date"].eq(visits["first_stage3_date"])
        & stage_numeric.eq(3)
    )
    has_index_stage3 = is_stage3_on_index.groupby(visits["PatientID"]).any()
    missing_index_stage3 = [
        patient_id
        for patient_id in cohort["PatientID"]
        if not has_index_stage3.get(patient_id, False)
    ]
    if missing_index_stage3:
        raise ValueError(
            "The pooling index lacks a Stage 3 embedding on "
            "first_stage3_date for some patients. Examples: "
            f"{missing_index_stage3[:10]}"
        )

    return cohort, visits, summary, n_outside_window, n_duplicate_rows_removed


def _resolve_activations_root(selected_k_dir: PathLike) -> Path:
    supplied = Path(selected_k_dir).expanduser().resolve()
    if not supplied.is_dir():
        raise FileNotFoundError(f"Selected K directory does not exist: {supplied}")

    if supplied.name == "activations":
        root = supplied
    elif (supplied / "activations").is_dir():
        root = supplied / "activations"
    else:
        raise FileNotFoundError(
            "Could not find the activations directory. Pass either the selected "
            "K directory (for example, k_0064) or its activations subdirectory."
        )
    return root


def _find_manifest(activations_root: Path, split_name: str) -> Path:
    filename = f"{split_name}_activation_manifest.json"
    candidates = [
        activations_root / filename,
        activations_root / split_name / filename,
        activations_root.parent / filename,
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"Could not find {filename}. Checked: "
            + ", ".join(str(path) for path in candidates)
        )
    return existing[0]


def _find_activation_index(activations_root: Path, split_name: str) -> Path:
    filename = f"{split_name}_activation_index.csv"
    candidates = [
        activations_root.parent / filename,
        activations_root / filename,
        activations_root / split_name / filename,
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"Could not find {filename}. Checked: "
            + ", ".join(str(path) for path in candidates)
        )
    return existing[0]


def _load_export_contract(
    activations_root: Path,
    required_splits: list[str],
) -> tuple[int, int, dict[str, Path]]:
    d_hidden = None
    k = None
    manifest_paths: dict[str, Path] = {}

    for split_name in required_splits:
        split_dir = activations_root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing exported activation split directory: {split_dir}"
            )

        manifest_path = _find_manifest(activations_root, split_name)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        for field in ["k", "d_hidden", "split", "files"]:
            if field not in manifest:
                raise KeyError(f"{manifest_path} is missing manifest field {field!r}.")

        manifest_split = str(manifest["split"]).strip().lower()
        if manifest_split != split_name:
            raise ValueError(
                f"Manifest {manifest_path} says split={manifest_split!r}; "
                f"expected {split_name!r}."
            )

        current_k = int(manifest["k"])
        current_d_hidden = int(manifest["d_hidden"])
        if current_k <= 0 or current_d_hidden <= 0 or current_k > current_d_hidden:
            raise ValueError(
                f"Invalid k/d_hidden in {manifest_path}: "
                f"k={current_k}, d_hidden={current_d_hidden}."
            )

        if k is None:
            k = current_k
            d_hidden = current_d_hidden
        elif current_k != k or current_d_hidden != d_hidden:
            raise ValueError(
                "Export manifests disagree across splits: "
                f"expected k={k}, d_hidden={d_hidden}; "
                f"{manifest_path} has k={current_k}, "
                f"d_hidden={current_d_hidden}."
            )
        manifest_paths[split_name] = manifest_path

    if k is None or d_hidden is None:
        raise ValueError("No activation splits were found in cohort_df.")
    return d_hidden, k, manifest_paths


def _integer_series(
    values: pd.Series,
    column_name: str,
    source_path: Path,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"{source_path}: {column_name} contains missing or nonnumeric values."
        )
    array = numeric.to_numpy()
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{source_path}: {column_name} must contain integers.")
    if (array < 0).any():
        raise ValueError(
            f"{source_path}: {column_name} must contain nonnegative values."
        )
    return numeric.astype(np.int64)


def _load_activation_indexes(
    activations_root: Path,
    required_splits: list[str],
) -> tuple[pd.DataFrame, dict[str, Path]]:
    required_columns = {
        "PatientID",
        "emb_id",
        "activation_row",
        "embedding_row",
        "date",
        "enc_id",
    }
    index_frames = []
    index_paths: dict[str, Path] = {}

    for split_name in required_splits:
        index_path = _find_activation_index(activations_root, split_name)
        frame = pd.read_csv(
            index_path,
            sep=ACTIVATION_INDEX_SEPARATOR,
            dtype={"PatientID": "string", "enc_id": "string"},
            low_memory=False,
        )
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise KeyError(
                f"{index_path} is missing required columns: {missing_columns}"
            )

        frame = _clean_patient_ids(frame, str(index_path))
        frame["emb_id"] = _integer_series(frame["emb_id"], "emb_id", index_path)
        frame["activation_row"] = _integer_series(
            frame["activation_row"], "activation_row", index_path
        )
        frame["embedding_row"] = _integer_series(
            frame["embedding_row"], "embedding_row", index_path
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        if frame["date"].isna().any():
            raise ValueError(f"{index_path} contains unusable date values.")

        if not frame["activation_row"].equals(frame["embedding_row"]):
            conflicts = frame.loc[
                frame["activation_row"].ne(frame["embedding_row"]),
                ["PatientID", "emb_id", "activation_row", "embedding_row"],
            ].head(10)
            raise ValueError(
                f"{index_path}: activation_row and embedding_row disagree:\n"
                f"{conflicts.to_string(index=False)}"
            )

        if "split" in frame.columns:
            observed_splits = set(
                frame["split"].astype(str).str.strip().str.lower().unique()
            )
            if observed_splits != {split_name}:
                raise ValueError(
                    f"{index_path}: expected only split {split_name!r}, found "
                    f"{sorted(observed_splits)}."
                )

        duplicate_row = frame.duplicated(
            ["PatientID", "activation_row"], keep=False
        )
        if duplicate_row.any():
            examples = frame.loc[
                duplicate_row,
                ["PatientID", "emb_id", "activation_row"],
            ].head(10)
            raise ValueError(
                f"{index_path}: PatientID + activation_row is not unique:\n"
                f"{examples.to_string(index=False)}"
            )

        # In the exporter's ``all_sorted_metadata_rows`` layout, several
        # embedding rows can intentionally refer to the same true visit.  The
        # first row (lowest activation_row) is the same canonical choice used
        # when ``unique_visit_index.csv`` is created.  Metadata must agree here;
        # the corresponding NPZ vectors are independently checked for exact
        # equality in ``_load_patient_sparse_file`` before pooling.
        duplicate_visit = frame.duplicated(["PatientID", "emb_id"], keep=False)
        if duplicate_visit.any():
            repeated = frame.loc[duplicate_visit]
            consistency = (
                repeated.groupby(["PatientID", "emb_id"], as_index=False)
                .agg(
                    n_dates=("date", lambda values: values.nunique(dropna=False)),
                    n_enc_ids=(
                        "enc_id",
                        lambda values: (
                            values.astype("string")
                            .str.strip()
                            .fillna("<MISSING>")
                            .nunique(dropna=False)
                        ),
                    ),
                )
            )
            conflicts = consistency.loc[
                consistency["n_dates"].gt(1)
                | consistency["n_enc_ids"].gt(1)
            ]
            if not conflicts.empty:
                raise ValueError(
                    f"{index_path}: repeated PatientID + emb_id rows disagree "
                    "in date or enc_id:\n"
                    f"{conflicts.head(10).to_string(index=False)}"
                )

        frame = (
            frame.sort_values(
                ["PatientID", "activation_row"], kind="mergesort"
            )
            .drop_duplicates(["PatientID", "emb_id"], keep="first")
            .reset_index(drop=True)
        )

        frame["_activation_split"] = split_name
        index_frames.append(
            frame[
                [
                    "_activation_split",
                    "PatientID",
                    "emb_id",
                    "activation_row",
                    "embedding_row",
                    "date",
                    "enc_id",
                ]
            ].copy()
        )
        index_paths[split_name] = index_path

    return pd.concat(index_frames, ignore_index=True), index_paths


def _attach_activation_rows(
    visits: pd.DataFrame,
    activation_index: pd.DataFrame,
) -> pd.DataFrame:
    visits = visits.copy()
    visits["_visit_order"] = np.arange(len(visits), dtype=np.int64)
    mapping = activation_index.rename(
        columns={
            "date": "activation_index_date",
            "enc_id": "activation_index_enc_id",
            "embedding_row": "activation_embedding_row",
        }
    )

    joined = visits.merge(
        mapping,
        on=["_activation_split", "PatientID", "emb_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    missing_mapping = joined["activation_row"].isna()
    if missing_mapping.any():
        examples = joined.loc[
            missing_mapping,
            ["_activation_split", "PatientID", "emb_id", "date", "enc_id"],
        ].head(10)
        raise KeyError(
            "Some pooling visits are absent from the exported activation index:\n"
            f"{examples.to_string(index=False)}"
        )

    date_conflict = joined["date"].ne(joined["activation_index_date"])
    if date_conflict.any():
        examples = joined.loc[
            date_conflict,
            [
                "PatientID",
                "emb_id",
                "date",
                "activation_index_date",
            ],
        ].head(10)
        raise ValueError(
            "Pooling and activation-index dates disagree:\n"
            f"{examples.to_string(index=False)}"
        )

    pooling_enc = (
        joined["enc_id"].astype("string").str.strip().fillna("<MISSING>")
    )
    activation_enc = (
        joined["activation_index_enc_id"]
        .astype("string")
        .str.strip()
        .fillna("<MISSING>")
    )
    enc_conflict = pooling_enc.ne(activation_enc)
    if enc_conflict.any():
        examples = joined.loc[
            enc_conflict,
            ["PatientID", "emb_id", "enc_id", "activation_index_enc_id"],
        ].head(10)
        raise ValueError(
            "Pooling and activation-index enc_id values disagree:\n"
            f"{examples.to_string(index=False)}"
        )

    joined["activation_row"] = joined["activation_row"].astype(np.int64)
    joined["activation_embedding_row"] = joined[
        "activation_embedding_row"
    ].astype(np.int64)
    joined = (
        joined.sort_values("_visit_order", kind="mergesort")
        .drop(
            columns=[
                "_visit_order",
                "activation_index_date",
                "activation_index_enc_id",
            ]
        )
        .reset_index(drop=True)
    )
    return joined


def _load_patient_sparse_file(
    activation_path: Path,
    expected_k: int,
    d_hidden: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(activation_path, allow_pickle=False) as saved:
        missing_keys = sorted(REQUIRED_NPZ_KEYS - set(saved.files))
        if missing_keys:
            raise KeyError(f"{activation_path} is missing NPZ keys: {missing_keys}")

        indices = np.asarray(saved["topk_indices"])
        values = np.asarray(saved["topk_values"])
        embedding_row = np.asarray(saved["embedding_row"])
        saved_emb_ids = np.asarray(saved["emb_id"])

    if indices.ndim != 2 or values.ndim != 2:
        raise ValueError(
            f"{activation_path}: topk arrays must be two-dimensional; "
            f"got {indices.shape} and {values.shape}."
        )
    if indices.shape != values.shape:
        raise ValueError(
            f"{activation_path}: topk_indices shape {indices.shape} does not "
            f"match topk_values shape {values.shape}."
        )
    if indices.shape[1] != expected_k:
        raise ValueError(
            f"{activation_path}: stored K is {indices.shape[1]}, "
            f"but the manifest says K={expected_k}."
        )

    n_rows = indices.shape[0]
    if embedding_row.shape != (n_rows,) or saved_emb_ids.shape != (n_rows,):
        raise ValueError(
            f"{activation_path}: embedding_row and emb_id must each have "
            f"shape ({n_rows},); got {embedding_row.shape} and "
            f"{saved_emb_ids.shape}."
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{activation_path}: topk_indices must be integer-valued.")
    if not np.issubdtype(embedding_row.dtype, np.integer):
        raise TypeError(f"{activation_path}: embedding_row must be integer-valued.")
    if not np.issubdtype(saved_emb_ids.dtype, np.integer):
        raise TypeError(f"{activation_path}: emb_id must be integer-valued.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{activation_path}: topk_values must be numeric.")

    indices = indices.astype(np.int64, copy=False)
    values = values.astype(np.float32, copy=False)
    embedding_row = embedding_row.astype(np.int64, copy=False)
    saved_emb_ids = saved_emb_ids.astype(np.int64, copy=False)

    if not np.array_equal(embedding_row, np.arange(n_rows, dtype=np.int64)):
        raise ValueError(
            f"{activation_path}: embedding_row is not contiguous zero-based "
            "array-row order."
        )
    if indices.size and (indices.min() < 0 or indices.max() >= d_hidden):
        raise IndexError(
            f"{activation_path}: a Top-K neuron index is outside "
            f"[0, {d_hidden})."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{activation_path}: topk_values contains NaN or Inf.")

    # A Top-K row should never contain the same neuron twice.
    if indices.shape[1] > 1:
        sorted_indices = np.sort(indices, axis=1)
        if (np.diff(sorted_indices, axis=1) == 0).any():
            raise ValueError(
                f"{activation_path}: at least one row repeats a neuron index."
            )

    # The deferred metadata join may create several activation rows that point
    # to the same true visit/emb_id. They are safe only if their sparse vectors
    # are exactly identical. The split activation index chooses the canonical
    # row used below.
    unique_emb_ids, emb_id_counts = np.unique(
        saved_emb_ids, return_counts=True
    )
    repeated_emb_ids = unique_emb_ids[emb_id_counts > 1]
    for repeated_emb_id in repeated_emb_ids:
        repeated_rows = np.flatnonzero(saved_emb_ids == repeated_emb_id)
        repeated_indices = indices[repeated_rows]
        repeated_values = values[repeated_rows]

        sort_order = np.argsort(repeated_indices, axis=1)
        canonical_indices = np.take_along_axis(
            repeated_indices, sort_order, axis=1
        )
        canonical_values = np.take_along_axis(
            repeated_values, sort_order, axis=1
        )
        if not (
            np.all(canonical_indices == canonical_indices[0])
            and np.all(canonical_values == canonical_values[0])
        ):
            raise ValueError(
                f"{activation_path}: repeated emb_id={int(repeated_emb_id)} "
                "maps to non-identical Top-K activation rows. This is a real "
                "indexing conflict and should not be silently deduplicated."
            )

    return indices, values, saved_emb_ids


def _select_and_pool_patient(
    activation_path: Path,
    requested_emb_ids: np.ndarray,
    requested_activation_rows: np.ndarray,
    expected_k: int,
    d_hidden: int,
) -> np.ndarray:
    indices, values, saved_emb_ids = _load_patient_sparse_file(
        activation_path=activation_path,
        expected_k=expected_k,
        d_hidden=d_hidden,
    )

    requested_emb_ids = np.asarray(requested_emb_ids, dtype=np.int64)
    requested_activation_rows = np.asarray(
        requested_activation_rows, dtype=np.int64
    )
    if requested_emb_ids.shape != requested_activation_rows.shape:
        raise ValueError(
            f"{activation_path}: requested emb_id and activation_row arrays "
            "have different shapes."
        )
    if requested_activation_rows.size == 0:
        raise ValueError(f"{activation_path}: no activation rows were requested.")
    if (
        requested_activation_rows.min() < 0
        or requested_activation_rows.max() >= len(saved_emb_ids)
    ):
        raise IndexError(
            f"{activation_path}: a requested activation_row is outside "
            f"[0, {len(saved_emb_ids)})."
        )

    observed_emb_ids = saved_emb_ids[requested_activation_rows]
    if not np.array_equal(observed_emb_ids, requested_emb_ids):
        mismatch = np.flatnonzero(observed_emb_ids != requested_emb_ids)[:10]
        examples = [
            {
                "activation_row": int(requested_activation_rows[position]),
                "expected_emb_id": int(requested_emb_ids[position]),
                "npz_emb_id": int(observed_emb_ids[position]),
            }
            for position in mismatch
        ]
        raise ValueError(
            f"{activation_path}: activation-index rows do not agree with the "
            f"NPZ emb_id mapping. Examples: {examples}"
        )

    selected_indices = indices[requested_activation_rows]
    selected_values = values[requested_activation_rows]
    n_visits = len(requested_activation_rows)

    # Compute the exact maximum of sparse vectors without materializing a
    # visits-by-d_hidden dense matrix. Inactive neurons equal zero.
    pooled = np.full(d_hidden, -np.inf, dtype=np.float32)
    selected_count = np.zeros(d_hidden, dtype=np.int64)
    flat_indices = selected_indices.reshape(-1)
    flat_values = selected_values.reshape(-1)
    np.maximum.at(pooled, flat_indices, flat_values)
    np.add.at(selected_count, flat_indices, 1)

    # If a neuron was inactive in at least one visit, zero participates in
    # the maximum. This also makes never-active neurons exactly zero.
    has_inactive_visit = selected_count < n_visits
    pooled[has_inactive_visit] = np.maximum(pooled[has_inactive_visit], 0.0)

    if not np.isfinite(pooled).all():
        raise RuntimeError(
            f"Sparse max pooling produced a nonfinite value for {activation_path}."
        )
    return pooled


def _verify_saved_rows(
    pooled_path: Path,
    patient_index_path: Path,
    patient_order: list[str],
    visits: pd.DataFrame,
    activation_paths: dict[str, Path],
    k: int,
    d_hidden: int,
    spot_check_count: int,
    random_seed: int,
) -> None:
    saved_matrix = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
    saved_index = pd.read_csv(patient_index_path, dtype={"PatientID": str})

    if saved_matrix.shape != (len(patient_order), d_hidden):
        raise RuntimeError(
            f"Saved matrix has shape {saved_matrix.shape}; expected "
            f"({len(patient_order)}, {d_hidden})."
        )
    if not np.array_equal(
        saved_index["pooled_row"].to_numpy(),
        np.arange(len(saved_index)),
    ):
        raise RuntimeError("pooled_row does not match the saved matrix row order.")
    if saved_index["PatientID"].tolist() != patient_order:
        raise RuntimeError("Saved patient order does not match cohort_df order.")

    n_checks = min(max(int(spot_check_count), 0), len(patient_order))
    if n_checks == 0:
        return

    rng = np.random.default_rng(random_seed)
    rows_to_check = rng.choice(len(patient_order), size=n_checks, replace=False)
    for pooled_row in rows_to_check:
        patient_id = patient_order[pooled_row]
        patient_visits = visits.loc[visits["PatientID"].eq(patient_id)]
        requested_emb_ids = patient_visits["emb_id"].to_numpy(dtype=np.int64)
        requested_activation_rows = patient_visits["activation_row"].to_numpy(
            dtype=np.int64
        )
        recomputed = _select_and_pool_patient(
            activation_path=activation_paths[patient_id],
            requested_emb_ids=requested_emb_ids,
            requested_activation_rows=requested_activation_rows,
            expected_k=k,
            d_hidden=d_hidden,
        )
        np.testing.assert_array_equal(
            saved_matrix[pooled_row],
            recomputed,
            err_msg=f"Saved pooled row failed verification for {patient_id}.",
        )


def maxpool_sae_activations(
    cohort_df: pd.DataFrame,
    pooling_index_df: pd.DataFrame,
    selected_k_dir: PathLike,
    output_dir: PathLike | None = None,
    *,
    split_column: str = "split",
    split_directory_map: Mapping[str, str] | None = None,
    verify_saved: bool = True,
    spot_check_count: int = 5,
    random_seed: int = 2026,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Max-pool sparse Top-K SAE visit activations for each cohort patient.

    ``selected_k_dir`` may point either to a chosen K directory such as
    ``k_0064`` or directly to its ``activations`` subdirectory. All cohort
    columns, including split and survival outcomes, are preserved in the
    patient index. Outputs default to
    ``Path.cwd() / 'sae_maxpooled_2y_prestage3'``.
    """
    if split_directory_map is None:
        split_directory_map = DEFAULT_SPLIT_DIRECTORY_MAP
    else:
        split_directory_map = {
            str(key).strip().lower(): str(value).strip()
            for key, value in split_directory_map.items()
        }

    if output_dir is None:
        output_dir =  Path("/path/to/sae_maxpooled_2y_prestage3")
    else:
        output_dir = Path(output_dir).expanduser()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_root = _resolve_activations_root(selected_k_dir)
    (
        cohort,
        visits,
        summary,
        n_outside_window,
        n_duplicate_rows_removed,
    ) = _prepare_pooling_rows(
        cohort_df=cohort_df,
        pooling_index_df=pooling_index_df,
        split_column=split_column,
        split_directory_map=split_directory_map,
    )

    required_splits = sorted(cohort["_activation_split"].unique().tolist())
    d_hidden, k, manifest_paths = _load_export_contract(
        activations_root=activations_root,
        required_splits=required_splits,
    )
    activation_index, activation_index_paths = _load_activation_indexes(
        activations_root=activations_root,
        required_splits=required_splits,
    )
    visits = _attach_activation_rows(
        visits=visits,
        activation_index=activation_index,
    )

    patient_order = cohort.sort_values("_cohort_row")["PatientID"].tolist()
    patient_split = cohort.set_index("PatientID")["_activation_split"].to_dict()
    activation_paths = {
        patient_id: (
            activations_root
            / patient_split[patient_id]
            / patient_id
            / "sae_topk.npz"
        )
        for patient_id in patient_order
    }
    missing_files = [
        str(path) for path in activation_paths.values() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} patient activation files were not found. "
            "First examples:\n"
            + "\n".join(missing_files[:10])
        )

    visit_groups = {
        patient_id: patient_visits
        for patient_id, patient_visits in visits.groupby(
            "PatientID", sort=False
        )
    }
    visit_audit = visits.drop(columns="_cohort_row", errors="ignore").copy()
    visit_audit["activation_file"] = ""
    visit_audit["topk_k"] = k
    visit_audit["sae_hidden_dimension"] = d_hidden

    final_pooled_path = output_dir / "sae_maxpooled_activations.npy"
    temporary_pooled_path = output_dir / ".sae_maxpooled_activations.tmp.npy"

    try:
        pooled_matrix = np.lib.format.open_memmap(
            temporary_pooled_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(patient_order), d_hidden),
        )

        for pooled_row, patient_id in enumerate(patient_order):
            patient_visits = visit_groups[patient_id]
            requested_emb_ids = patient_visits["emb_id"].to_numpy(dtype=np.int64)
            requested_activation_rows = patient_visits[
                "activation_row"
            ].to_numpy(dtype=np.int64)
            patient_pooled = _select_and_pool_patient(
                activation_path=activation_paths[patient_id],
                requested_emb_ids=requested_emb_ids,
                requested_activation_rows=requested_activation_rows,
                expected_k=k,
                d_hidden=d_hidden,
            )
            pooled_matrix[pooled_row] = patient_pooled

            visit_audit.loc[patient_visits.index, "activation_file"] = str(
                activation_paths[patient_id]
            )

            patient_number = pooled_row + 1
            if verbose and (
                patient_number % 100 == 0
                or patient_number == len(patient_order)
            ):
                print(
                    f"Processed {patient_number:,}/{len(patient_order):,} patients"
                )

        pooled_matrix.flush()
        del pooled_matrix
        os.replace(temporary_pooled_path, final_pooled_path)
    except Exception:
        try:
            del pooled_matrix
        except UnboundLocalError:
            pass
        if temporary_pooled_path.exists():
            temporary_pooled_path.unlink()
        raise

    patient_index = (
        cohort.sort_values("_cohort_row")
        .drop(columns=["_cohort_row", "_activation_split"])
        .merge(
            summary,
            on="PatientID",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )
    if patient_index["PatientID"].tolist() != patient_order:
        raise RuntimeError("Patient order changed while creating the index.")

    patient_index.insert(
        0,
        "pooled_row",
        np.arange(len(patient_index), dtype=np.int64),
    )
    patient_index["topk_k"] = k
    patient_index["sae_hidden_dimension"] = d_hidden
    patient_index["selected_k_dir"] = str(activations_root.parent)

    patient_to_pooled_row = {
        patient_id: pooled_row
        for pooled_row, patient_id in enumerate(patient_order)
    }
    visit_audit.insert(
        0,
        "pooled_row",
        visit_audit["PatientID"]
        .map(patient_to_pooled_row)
        .astype(np.int64),
    )
    visit_audit = visit_audit.drop(
        columns="_activation_split", errors="ignore"
    ).sort_values(
        ["pooled_row", "sae_pool_position"], kind="mergesort"
    ).reset_index(drop=True)

    patient_index_path = output_dir / "sae_maxpooled_index.csv"
    visit_audit_path = output_dir / "sae_maxpool_visit_index.csv"
    patient_index.to_csv(patient_index_path, index=False)
    visit_audit.to_csv(visit_audit_path, index=False)

    if verify_saved:
        _verify_saved_rows(
            pooled_path=final_pooled_path,
            patient_index_path=patient_index_path,
            patient_order=patient_order,
            visits=visits,
            activation_paths=activation_paths,
            k=k,
            d_hidden=d_hidden,
            spot_check_count=spot_check_count,
            random_seed=random_seed,
        )

    saved_matrix = np.load(
        final_pooled_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if verbose:
        print("\nTop-K SAE max pooling complete.")
        print("Inclusive window: lookback_start_date <= date <= first_stage3_date")
        print(f"Selected K: {k}")
        print(f"SAE hidden dimension: {d_hidden}")
        print(f"Patients: {len(patient_index):,}")
        print(f"Pooled matrix shape: {saved_matrix.shape}")
        print(f"Contributing embedding rows: {len(visit_audit):,}")
        print(f"Rows outside window excluded: {n_outside_window:,}")
        print(f"Repeated metadata emb_id rows removed: {n_duplicate_rows_removed:,}")
        print(
            "Unique visit dates per patient: "
            f"min={summary['n_unique_visit_dates_pooled'].min():,}, "
            f"median={summary['n_unique_visit_dates_pooled'].median():.1f}, "
            f"max={summary['n_unique_visit_dates_pooled'].max():,}"
        )
        print("Activation manifests:")
        for split_name, manifest_path in manifest_paths.items():
            print(f"  {split_name}: {manifest_path}")
        print("Activation indexes:")
        for split_name, index_path in activation_index_paths.items():
            print(f"  {split_name}: {index_path}")
        print(f"Saved pooled matrix: {final_pooled_path}")
        print(f"Saved patient index: {patient_index_path}")
        print(f"Saved visit audit: {visit_audit_path}")
        if verify_saved:
            print("Saved-output alignment checks passed.")

    return saved_matrix, patient_index, visit_audit


__all__ = ["maxpool_sae_activations"]
