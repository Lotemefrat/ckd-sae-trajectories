"""Three-month SAE trajectories using the existing CKD cohort and Top-K export.

Keep maxpool_sae_activations.py beside this file (included in the download).
Requires numpy and pandas. No model fitting or outcome reconstruction occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maxpool_sae_activations import (
    DEFAULT_SPLIT_DIRECTORY_MAP,
    _clean_patient_ids,
    _load_activation_indexes,
    _load_export_contract,
    _load_patient_sparse_file,
    _resolve_activations_root,
)


def _choose_column(frame, requested, candidates, label):
    if requested is not None:
        if requested not in frame:
            raise KeyError(f"Missing {label} column: {requested}")
        return requested
    available = [c for c in candidates if c in frame]
    if not available:
        raise KeyError(f"Specify {label}_column. Tried {candidates}")
    return available[0]


def _pool_bins(indices, values, row_numbers, bin_numbers, dimension):
    """Eight sparse maxima; implicit zero participates in each observed bin."""
    pooled = np.full((8, dimension), np.nan, dtype=np.float32)
    for b in np.unique(bin_numbers):
        selected = row_numbers[bin_numbers == b]
        pooled[b] = 0.0
        np.maximum.at(pooled[b], indices[selected].ravel(), values[selected].ravel())
    return pooled


def build_sae_temporal(
    cohort_df,
    selected_k_dir,
    output_dir="/path/to/topk_SAE/sae_temporal_3m_2y",
    *,
    split_column="split",
    outcome_column=None,
    time_column=None,
    prevalence_threshold=0.01,
    verbose=True,
):
    """Return patient CSV metadata (without loading the wide neuron columns).

    Cohort: one row per PatientID, first_stage3_date, split and survival outcome.
    Time must be DAYS from index until progression OR censoring.
    All supplied cohort patients are audited, including dropped patients.
    Scaling/filtering use ONLY retained training patients in this cohort.
    """
    if not 0 < prevalence_threshold <= 1:
        raise ValueError("prevalence_threshold must be in (0, 1].")
    cohort = _clean_patient_ids(cohort_df.reset_index(drop=True), "cohort_df")
    if cohort.empty or cohort.PatientID.duplicated().any():
        raise ValueError("Supply a nonempty cohort with exactly one row per patient.")
    outcome_column = _choose_column(cohort, outcome_column,
        ["event_observed", "progression", "event", "outcome"], "outcome")
    time_column = _choose_column(cohort, time_column,
        ["time_to_progression_or_censor_days", "followup_days", "time_to_outcome_days"], "time")
    cohort["first_stage3_date"] = pd.to_datetime(
        cohort["first_stage3_date"], errors="raise").dt.normalize()
    if cohort.first_stage3_date.isna().any():
        raise ValueError("Missing first_stage3_date.")
    splits = cohort[split_column].astype("string").str.strip().str.lower()
    cohort["_split"] = splits.map(DEFAULT_SPLIT_DIRECTORY_MAP)
    if cohort._split.isna().any():
        raise ValueError("Each patient needs train, validation (val), or test split.")
    outcomes = pd.to_numeric(cohort[outcome_column], errors="raise")
    times = pd.to_numeric(cohort[time_column], errors="raise")
    if not outcomes.isin([0, 1]).all():
        raise ValueError("Outcome must be 1=progression, 0=censored.")
    if not np.isfinite(times.to_numpy(dtype=float)).all() or (times < 0).any():
        raise ValueError("Follow-up days must be finite and nonnegative.")

    root = _resolve_activations_root(selected_k_dir)
    required_splits = sorted(cohort._split.unique())
    dimension, k, _ = _load_export_contract(root, required_splits)
    visit_index, _ = _load_activation_indexes(root, required_splits)
    visit_index = visit_index.loc[visit_index.PatientID.isin(cohort.PatientID)]
    groups = dict(tuple(visit_index.groupby(["_activation_split", "PatientID"], sort=False)))

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / ".raw_bins.tmp.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32,
                                  shape=(len(cohort), 8, dimension))
    train_max = np.zeros(dimension, dtype=np.float32)
    active_count = np.zeros(dimension, dtype=np.int64)
    n_train = 0
    patient_records, bin_records = [], []
    try:
        for i, patient in cohort.iterrows():
            index_date = patient.first_stage3_date
            # Direct offsets from index avoid drift at month ends/leap years.
            edges = pd.DatetimeIndex([index_date - pd.DateOffset(months=m)
                                      for m in range(24, -1, -3)])
            visits = groups.get((patient._split, patient.PatientID))
            if visits is None:
                visits = visit_index.iloc[:0].copy()
            visits = visits.loc[visits.date.between(edges[0], index_date)].copy()
            dates = visits.date.to_numpy(dtype="datetime64[ns]")
            # At each internal boundary, choose the older bin.
            # Keep index day in bin 0 and the exact 24-month endpoint in bin 7.
            chronological_bin = np.clip(np.searchsorted(edges.to_numpy(), dates,
                                                         side="left") - 1, 0, 7)
            bins = 7 - chronological_bin  # bin 0 is most recent; bin 7 oldest.
            counts = np.bincount(bins, minlength=8)
            observed = counts > 0
            has_early = bool((visits.date <= index_date - pd.DateOffset(months=6)).any())
            retained = has_early
            has_index_visit = bool(visits.date.eq(index_date).any())
            pooled = np.full((8, dimension), np.nan, dtype=np.float32)
            if len(visits):
                activation_path = root / patient._split / patient.PatientID / "sae_topk.npz"
                indices, values, saved_emb_ids = _load_patient_sparse_file(activation_path, k, dimension)
                rows = visits.activation_row.to_numpy(dtype=np.int64)
                if rows.max() >= len(saved_emb_ids):
                    raise IndexError(f"Activation row outside array for {patient.PatientID}")
                if not np.array_equal(saved_emb_ids[rows], visits.emb_id.to_numpy(dtype=np.int64)):
                    raise ValueError(f"Activation row/emb_id mismatch for {patient.PatientID}")
                if (values < 0).any():
                    raise ValueError("This pipeline requires nonnegative Top-K SAE activations.")
                pooled = _pool_bins(indices, values, rows, bins, dimension)
            raw[i] = pooled
            if retained and patient._split == "train":
                patient_max = pooled[observed].max(axis=0)
                np.maximum(train_max, patient_max, out=train_max)
                active_count += patient_max > 0
                n_train += 1
            patient_records.append({
                "patient_row": i, "PatientID": patient.PatientID, "split": patient._split,
                "status": "retain" if retained else "dropped",
                "drop_reason": "" if retained else (
                    "no_visits_in_2y_window" if not len(visits) else "no_visit_at_least_6m_before_index"),
                "first_stage3_date": index_date, "lookback_start_date": edges[0],
                "first_visit_in_window": visits.date.min(), "last_visit_in_window": visits.date.max(),
                "n_visits": len(visits), "n_observed_bins": int(observed.sum()),
                "has_visit_at_least_6m_before_index": has_early,
                "has_diagnosis_day_visit": has_index_visit,
                "has_index_bin": bool(observed[0]),
                "earliest_observed_bin": int(np.flatnonzero(observed).max()) if observed.any() else np.nan,
                "outcome": int(outcomes.iloc[i]), "time_to_outcome_or_censor_days": float(times.iloc[i]),
            })
            for b in range(8):
                bin_records.append({"patient_row": i, "PatientID": patient.PatientID,
                    "bin": b, "start_date": edges[7-b], "end_date": edges[8-b],
                    "start_inclusive": b == 7, "end_inclusive": True,
                    "n_visits": int(counts[b])})
            if verbose and ((i+1) % 100 == 0 or i+1 == len(cohort)):
                print(f"Pooled {i+1:,}/{len(cohort):,} patients", flush=True)
        raw.flush()
        if not n_train:
            raise ValueError("No retained training patients available for scaling/filtering.")
        prevalence = active_count / n_train
        keep = (prevalence >= prevalence_threshold) & (train_max > 0)
        neurons = np.flatnonzero(keep)
        if not len(neurons):
            raise ValueError("No neurons meet the prevalence threshold.")
        summary = pd.DataFrame(patient_records)
        scaled = np.lib.format.open_memmap(out / "bin_maxima.npy", mode="w+",
            dtype=np.float32, shape=(len(cohort), 8, len(neurons)))
        # Stream one patient at a time: avoid a large wide dataframe in memory.
        with (out / "patient_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            metadata_columns = list(patient_records[0])
            feature_columns = [name for neuron in neurons for name in
                               (f"neuron_{neuron}_change", f"neuron_{neuron}_average")]
            writer.writerow(metadata_columns + feature_columns)
            for i, record in enumerate(patient_records):
                patient_bins = raw[i][:, keep] / train_max[keep]
                scaled[i] = patient_bins
                observed = ~np.isnan(patient_bins[:, 0])
                mean = patient_bins[observed].mean(axis=0) if observed.any() else np.full(len(neurons), np.nan)
                change = (patient_bins[0] - patient_bins[np.flatnonzero(observed).max()]
                          if observed[0] else np.full(len(neurons), np.nan))
                metadata = ["" if pd.isna(record[col]) else record[col]
                            for col in metadata_columns]
                features = np.column_stack([change, mean]).ravel()
                writer.writerow(metadata + ["" if np.isnan(value) else format(value, ".8g")
                                            for value in features])
        scaled.flush()
        del scaled
        pd.DataFrame(bin_records).to_csv(out / "bin_index.csv", index=False)
        pd.DataFrame({"neuron": np.arange(dimension), "train_active_patients": active_count,
            "train_patients": n_train, "train_prevalence": prevalence,
            "train_max": train_max, "retained": keep,
            "feature_column": np.where(keep, np.cumsum(keep)-1, -1),
        }).to_csv(out / "neuron_filter.csv", index=False)
        settings = {"selected_k_dir": str(root.parent), "k": k,
            "n_patients": len(cohort), "n_retained_patients": int(summary.status.eq("retain").sum()),
            "n_retained_train": n_train, "n_retained_neurons": len(neurons),
            "outcome_source_column": outcome_column, "time_source_column": time_column,
            "time_unit": "days", "prevalence_threshold": prevalence_threshold,
            "scaling": "max over all in-window visits of retained training patients; no clipping",
            "bin_order": "0 most recent through 7 oldest; internal boundaries go to older bin",
            "bin_boundaries": "(start_date, end_date], except oldest bin includes both endpoints",
            "window": "[index - 24 calendar months, index], including diagnosis day",
            "retention": "at least one in-window visit on or before index - 6 calendar months",
            "change": "index-bin maximum minus earliest observed-bin maximum; missing if index bin absent",
            "average": "mean of observed-bin maxima; empty bins omitted; observed zeros included",
            "dropped_patient_features": "computed when available, using retained-train scale/filter",
            "array_alignment": "patient_summary patient_row; neuron_filter retained rows sorted by neuron",
        }
        (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        if verbose:
            print(f"Retained {settings['n_retained_patients']:,}/{len(cohort):,} patients; "
                  f"{len(neurons):,}/{dimension:,} neurons.\nSaved {out / 'patient_summary.csv'}", flush=True)
        return summary
    finally:
        del raw
        raw_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, help="Patient-level cohort CSV or parquet, including splits/outcomes.")
    parser.add_argument("--selected-k-dir", required=True, help="Selected k directory or its activations subdirectory.")
    parser.add_argument("--output-dir", default="/path/to/topk_SAE/sae_temporal_3m_2y")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--outcome-column")
    parser.add_argument("--time-column", help="Follow-up DAYS to progression or censoring.")
    args = parser.parse_args()
    path = Path(args.cohort)
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, dtype={"PatientID": "string"})
    build_sae_temporal(frame, args.selected_k_dir, args.output_dir,
        split_column=args.split_column, outcome_column=args.outcome_column, time_column=args.time_column)


if __name__ == "__main__":
    main()
