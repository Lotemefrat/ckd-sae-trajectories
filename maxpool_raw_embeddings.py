"""Max-pool raw visit embeddings over the pre-Stage-3 cohort window.

Notebook usage
--------------
from maxpool_raw_embeddings import maxpool_raw_embeddings

X_raw, raw_index_df, raw_visit_index_df = maxpool_raw_embeddings(
    cohort_df=cohort_df,
    pooling_index_df=pooling_index_df,
)

The function uses the inclusive calendar-date window
``lookback_start_date <= date <= first_stage3_date``. Each ``emb_id`` is
treated as a zero-based row index into ``<PatientID>/<PatientID>.npy``.
``enc_id`` is retained for auditing and is never used as an array index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]

DEFAULT_RAW_EMBEDDING_ROOT = Path(
    "/path/to/embeddings/outputs/"
    "ckd_embedding_full_v3_icd_stage_filter"
)


def _clean_patient_ids(frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    """Return a copy with validated string PatientID values."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """Validate inputs and return cohort, contributing rows, and summary."""
    required_cohort_columns = {
        "PatientID",
        "lookback_start_date",
        "first_stage3_date",
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

    duplicated_patients = cohort.loc[
        cohort["PatientID"].duplicated(keep=False),
        "PatientID",
    ].unique()
    if len(duplicated_patients) > 0:
        raise ValueError(
            "cohort_df must contain one row per patient. Duplicated examples: "
            f"{duplicated_patients[:10].tolist()}"
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

    # Always use the authoritative window dates from cohort_df.
    pooling = pooling.drop(
        columns=["lookback_start_date", "first_stage3_date", "_cohort_row"],
        errors="ignore",
    )
    pooling = pooling.merge(
        cohort[
            [
                "PatientID",
                "lookback_start_date",
                "first_stage3_date",
                "_cohort_row",
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

    # Both endpoints are included. Do not recompute the two-year date here:
    # the cohort dataframe already contains the calendar-year-aware boundary.
    inside_window = (
        pooling["date"].ge(pooling["lookback_start_date"])
        & pooling["date"].le(pooling["first_stage3_date"])
    )
    n_outside_window = int((~inside_window).sum())
    visits = pooling.loc[inside_window].copy()

    duplicate_mask = visits.duplicated(["PatientID", "emb_id"], keep=False)
    duplicate_rows = visits.loc[duplicate_mask]

    if not duplicate_rows.empty:
        duplicate_consistency = (
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
        conflicts = duplicate_consistency.loc[
            (duplicate_consistency["n_dates"] > 1)
            | (duplicate_consistency["n_enc_ids"] > 1)
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
            ["_cohort_row", "emb_id", "date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    visits["raw_pool_position"] = (
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

    summarized_patients = set(summary["PatientID"])
    patients_without_rows = sorted(cohort_patients - summarized_patients)
    if patients_without_rows:
        raise ValueError(
            "Some cohort patients have no embeddings inside the inclusive "
            f"window: {patients_without_rows[:10]}"
        )

    stage_numeric = pd.to_numeric(visits["CKD_stage_numeric"], errors="coerce")
    is_stage3_on_index_date = (
        visits["date"].eq(visits["first_stage3_date"])
        & stage_numeric.eq(3)
    )
    has_index_stage3 = is_stage3_on_index_date.groupby(visits["PatientID"]).any()
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

    return (
        cohort,
        visits,
        summary,
        n_outside_window,
        n_duplicate_rows_removed,
    )


def _verify_saved_outputs(
    pooled_path: Path,
    patient_index_path: Path,
    patient_order: list[str],
    visits: pd.DataFrame,
    embedding_paths: dict[str, Path],
    spot_check_count: int,
    random_seed: int,
) -> None:
    """Reload saved files and independently recompute several patient rows."""
    saved_embeddings = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
    saved_index = pd.read_csv(patient_index_path, dtype={"PatientID": str})

    if saved_embeddings.shape[0] != len(saved_index):
        raise RuntimeError("Saved embedding and index row counts do not agree.")
    if not np.array_equal(
        saved_index["pooled_row"].to_numpy(),
        np.arange(len(saved_index)),
    ):
        raise RuntimeError("pooled_row does not match the saved matrix row order.")
    if saved_index["PatientID"].tolist() != patient_order:
        raise RuntimeError("Saved patient order does not match cohort_df order.")

    n_checks = min(max(int(spot_check_count), 0), len(saved_index))
    if n_checks == 0:
        return

    rng = np.random.default_rng(random_seed)
    rows_to_check = rng.choice(len(saved_index), size=n_checks, replace=False)

    for pooled_row in rows_to_check:
        patient_id = saved_index.loc[pooled_row, "PatientID"]
        emb_ids = visits.loc[
            visits["PatientID"].eq(patient_id), "emb_id"
        ].to_numpy(dtype=np.int64)
        patient_embeddings = np.load(
            embedding_paths[patient_id],
            mmap_mode="r",
            allow_pickle=False,
        )
        independently_pooled = np.asarray(
            patient_embeddings[emb_ids].max(axis=0),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            saved_embeddings[pooled_row],
            independently_pooled,
            err_msg=f"Saved pooled row failed verification for {patient_id}.",
        )


def maxpool_raw_embeddings(
    cohort_df: pd.DataFrame,
    pooling_index_df: pd.DataFrame,
    raw_embedding_root: PathLike = DEFAULT_RAW_EMBEDDING_ROOT,
    output_dir: PathLike | None = None,
    *,
    verify_saved: bool = True,
    spot_check_count: int = 5,
    random_seed: int = 2026,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Max-pool raw visit embeddings for each cohort patient.

    Parameters
    ----------
    cohort_df
        One row per retained patient. Must contain ``PatientID``,
        ``lookback_start_date``, and ``first_stage3_date``. All its columns,
        including outcome and split columns, are preserved in the patient index.
    pooling_index_df
        Visit-to-embedding bridge containing ``PatientID``, ``date``,
        ``CKD_stage_numeric``, ``emb_id``, and ``enc_id``.
    raw_embedding_root
        Directory containing ``<PatientID>/<PatientID>.npy`` files.
    output_dir
        Output directory. Defaults to
        ``<raw_embedding_root>/raw_maxpooled_2y_prestage3``.
    verify_saved
        Reload outputs and independently recompute a small patient sample.
    spot_check_count
        Number of randomly selected patients to recompute when verifying.
    random_seed
        Seed used only to select verification patients.
    verbose
        Print progress and a final summary.

    Returns
    -------
    pooled_embeddings, patient_index_df, visit_audit_df
        The first dimension of ``pooled_embeddings`` is aligned exactly with
        ``patient_index_df['pooled_row']``.
    """
    raw_embedding_root = Path(raw_embedding_root)
    if output_dir is None:
        output_dir = Path("/path/to/topk_SAE/ML4H/raw_maxpooled_2y_prestage3")
    else:
        output_dir = Path(output_dir)

    if not raw_embedding_root.is_dir():
        raise FileNotFoundError(
            f"Raw embedding root does not exist: {raw_embedding_root}"
        )

    (
        cohort,
        visits,
        summary,
        n_outside_window,
        n_duplicate_rows_removed,
    ) = _prepare_pooling_rows(cohort_df, pooling_index_df)

    patient_order = (
        cohort.sort_values("_cohort_row")["PatientID"].tolist()
    )
    visit_groups = {
        patient_id: patient_visits
        for patient_id, patient_visits in visits.groupby(
            "PatientID", sort=False
        )
    }
    embedding_paths = {
        patient_id: raw_embedding_root / patient_id / f"{patient_id}.npy"
        for patient_id in patient_order
    }

    missing_files = [
        str(path) for path in embedding_paths.values() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} patient embedding files were not found. "
            "First examples:\n"
            + "\n".join(missing_files[:10])
        )

    pooled_vectors: list[np.ndarray] = []
    embedding_dimension: int | None = None

    for patient_number, patient_id in enumerate(patient_order, start=1):
        emb_ids = visit_groups[patient_id]["emb_id"].to_numpy(dtype=np.int64)
        embedding_path = embedding_paths[patient_id]
        patient_embeddings = np.load(
            embedding_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if patient_embeddings.ndim != 2:
            raise ValueError(
                f"{embedding_path} has shape {patient_embeddings.shape}; "
                "expected a two-dimensional array."
            )
        if emb_ids.max() >= patient_embeddings.shape[0]:
            raise IndexError(
                f"Patient {patient_id}: largest emb_id is {emb_ids.max()}, "
                f"but {embedding_path.name} has only "
                f"{patient_embeddings.shape[0]} rows."
            )

        current_dimension = int(patient_embeddings.shape[1])
        if embedding_dimension is None:
            embedding_dimension = current_dimension
        elif current_dimension != embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for patient {patient_id}: "
                f"expected {embedding_dimension}, found {current_dimension}."
            )

        selected_embeddings = np.asarray(patient_embeddings[emb_ids])
        if not np.isfinite(selected_embeddings).all():
            raise ValueError(
                f"Patient {patient_id} has NaN or infinite selected values."
            )

        pooled_vectors.append(
            np.asarray(selected_embeddings.max(axis=0), dtype=np.float32)
        )

        if verbose and (
            patient_number % 100 == 0 or patient_number == len(patient_order)
        ):
            print(
                f"Processed {patient_number:,}/{len(patient_order):,} patients"
            )

    pooled_embeddings = np.stack(pooled_vectors, axis=0)

    patient_index = (
        cohort.sort_values("_cohort_row")
        .drop(columns="_cohort_row")
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
    patient_index["raw_embedding_dimension"] = embedding_dimension

    patient_to_pooled_row = {
        patient_id: pooled_row
        for pooled_row, patient_id in enumerate(patient_order)
    }
    visit_audit = visits.drop(columns="_cohort_row", errors="ignore").copy()
    visit_audit.insert(
        0,
        "pooled_row",
        visit_audit["PatientID"]
        .map(patient_to_pooled_row)
        .astype(np.int64),
    )
    visit_audit = visit_audit.sort_values(
        ["pooled_row", "raw_pool_position"], kind="mergesort"
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = output_dir / "raw_maxpooled_embeddings.npy"
    patient_index_path = output_dir / "raw_maxpooled_index.csv"
    visit_audit_path = output_dir / "raw_maxpool_visit_index.csv"

    np.save(pooled_path, pooled_embeddings)
    patient_index.to_csv(patient_index_path, index=False)
    visit_audit.to_csv(visit_audit_path, index=False)

    if verify_saved:
        _verify_saved_outputs(
            pooled_path=pooled_path,
            patient_index_path=patient_index_path,
            patient_order=patient_order,
            visits=visits,
            embedding_paths=embedding_paths,
            spot_check_count=spot_check_count,
            random_seed=random_seed,
        )

    if verbose:
        print("\nMax pooling complete.")
        print(f"Inclusive window: lookback_start_date <= date <= first_stage3_date")
        print(f"Patients: {len(patient_index):,}")
        print(f"Pooled matrix shape: {pooled_embeddings.shape}")
        print(f"Contributing embedding rows: {len(visit_audit):,}")
        print(f"Rows outside window excluded: {n_outside_window:,}")
        print(f"Repeated emb_id rows removed: {n_duplicate_rows_removed:,}")
        print(
            "Unique visit dates per patient: "
            f"min={summary['n_unique_visit_dates_pooled'].min():,}, "
            f"median={summary['n_unique_visit_dates_pooled'].median():.1f}, "
            f"max={summary['n_unique_visit_dates_pooled'].max():,}"
        )
        print(f"Saved pooled matrix: {pooled_path}")
        print(f"Saved patient index: {patient_index_path}")
        print(f"Saved visit audit: {visit_audit_path}")
        if verify_saved:
            print("Saved-output alignment checks passed.")

    return pooled_embeddings, patient_index, visit_audit


__all__ = [
    "DEFAULT_RAW_EMBEDDING_ROOT",
    "maxpool_raw_embeddings",
]
