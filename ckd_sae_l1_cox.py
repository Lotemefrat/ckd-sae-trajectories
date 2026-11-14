"""Train-only L1 Cox model for max-pooled SAE activations.

Dependencies: numpy, pandas, scikit-survival, joblib.
Rows are aligned using pooled_row in the companion patient index.
Scaling and the >=1% prevalence filter use training patients only.
The validation set selects alpha; the frozen model evaluates test once.
Neuron numbers are zero-based activation-matrix column indices.
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


DEFAULT_DIR = Path(
    "/path/to/topk_SAE/sae_maxpooled_2y_prestage3"
)


def c_index(y, risk):
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def run(args):
    raw = np.load(args.activations, mmap_mode="r", allow_pickle=False)
    index = pd.read_csv(args.index, dtype={"PatientID": str})
    required = ["PatientID", "pooled_row", args.split_col, args.time_col, args.event_col]
    missing = [col for col in required if col not in index]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {index.columns.tolist()}")
    if raw.ndim != 2 or len(index) != raw.shape[0]:
        raise ValueError("Expected one index row per patient in the 2D activation matrix.")
    if index["PatientID"].isna().any() or index["PatientID"].duplicated().any():
        raise ValueError("PatientID must be present and unique across all splits.")

    index["pooled_row"] = pd.to_numeric(index["pooled_row"], errors="raise")
    index = index.sort_values("pooled_row").reset_index(drop=True)
    if not np.array_equal(index["pooled_row"], np.arange(len(index))):
        raise ValueError("pooled_row must contain every matrix row exactly once (0..N-1).")

    splits = index[args.split_col].astype(str).str.strip().str.lower()
    splits = splits.replace({"training": "train", "val": "validation"})
    if not splits.isin(["train", "validation", "test"]).all():
        raise ValueError(f"Unexpected split labels: {splits.unique().tolist()}")
    masks = {name: splits.eq(name).to_numpy() for name in ["train", "validation", "test"]}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Train, validation, and test must all contain patients.")

    time = pd.to_numeric(index[args.time_col], errors="raise").to_numpy(float)
    event = pd.to_numeric(index[args.event_col], errors="raise").to_numpy()
    if not np.isfinite(time).all() or (time <= 0).any():
        raise ValueError("Follow-up time must be finite and strictly positive.")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("Event must be 0 (censored) or 1 (progressed).")
    outcomes = Surv.from_arrays(event.astype(bool), time)
    y = {name: outcomes[mask] for name, mask in masks.items()}

    # Fit preprocessing on TRAIN only, using already max-pooled activations.
    train = np.asarray(raw[masks["train"]], dtype=float)
    if not np.isfinite(train).all() or (train < 0).any():
        raise ValueError("SAE activations must be finite and nonnegative.")
    active_counts = np.count_nonzero(train, axis=0)
    train_max = train.max(axis=0)
    keep = (active_counts * 100 >= len(train)) & (train_max > 0)
    neurons = np.flatnonzero(keep)
    if len(neurons) == 0:
        raise ValueError("No neurons meet the >=1% training prevalence threshold.")
    scale = train_max[neurons]
    X = {"train": train[:, neurons] / scale}
    del train
    for name in ["validation", "test"]:
        values = np.asarray(raw[masks[name]][:, neurons], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid SAE activations in {name}.")
        X[name] = values / scale  # Same TRAIN maxima; do not clip held-out values.

    print(f"Retained {len(neurons):,}/{raw.shape[1]:,} neurons", flush=True)
    model = CoxnetSurvivalAnalysis(
        l1_ratio=1.0, normalize=False, n_alphas=args.n_alphas,
        alpha_min_ratio=args.alpha_min_ratio, max_iter=args.max_iter,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X["train"], y["train"])

    # Each path column is a TRAIN-fitted model. Choose alpha using validation only.
    validation_risk = X["validation"] @ model.coef_ - model.offset_
    scores = np.array([
        c_index(y["validation"], validation_risk[:, j])
        for j in range(len(model.alphas_))
    ])
    if not np.isfinite(scores).all():
        raise ValueError("Validation C-index could not be calculated.")
    best = int(np.argmax(scores))  # Ties favor the first (stronger) penalty.
    alpha = float(model.alphas_[best])
    beta = model.coef_[:, best]

    # No refit on validation or test, and no further tuning after test evaluation.
    metrics = pd.DataFrame([
        {"split": name, "patients": int(mask.sum()),
         "events": int(y[name]["event"].sum()),
         "c_index": c_index(y[name], model.predict(X[name], alpha=alpha)),
         "selected_alpha": alpha, "retained_neurons": len(neurons),
         "selected_neurons": int(np.count_nonzero(beta))}
        for name, mask in masks.items()
    ])
    selected = pd.DataFrame({
        "sae_neuron": neurons, "coefficient": beta,
        "abs_coefficient": np.abs(beta),
        "hazard_ratio_per_train_max": np.exp(beta),
        "train_activation_prevalence": active_counts[neurons] / masks["train"].sum(),
        "train_max_activation": scale,
    })
    selected = selected.loc[selected["coefficient"] != 0].sort_values(
        "abs_coefficient", ascending=False, kind="stable"
    )
    top25 = selected.head(25)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "c_indices.csv", index=False)
    selected.to_csv(args.output_dir / "selected_neurons.csv", index=False)
    top25.to_csv(args.output_dir / "top25_neurons.csv", index=False)
    pd.DataFrame({
        "alpha": model.alphas_, "validation_c_index": scores,
        "selected_neurons": np.count_nonzero(model.coef_, axis=0),
    }).to_csv(args.output_dir / "validation_alpha_path.csv", index=False)
    pd.DataFrame({
        "sae_neuron": np.arange(raw.shape[1]),
        "active_train_patients": active_counts,
        "train_activation_prevalence": active_counts / masks["train"].sum(),
        "train_max_activation": train_max, "retained": keep,
    }).to_csv(args.output_dir / "neuron_preprocessing.csv", index=False)
    joblib.dump({
        "model": model, "selected_alpha": alpha, "neurons": neurons,
        "train_max": scale, "input_neuron_count": raw.shape[1],
        "time_column": args.time_col, "event_column": args.event_col,
    }, args.output_dir / "cox_model.joblib")
    print(metrics.to_string(index=False))
    print("\nTop selected neurons (up to 25; ranked by absolute coefficient):")
    print(top25.to_string(index=False))
    print(f"\nOutputs: {args.output_dir}")
    return metrics, selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path,
                        default=DEFAULT_DIR / "sae_maxpooled_activations.npy")
    parser.add_argument("--index", type=Path,
                        default=DEFAULT_DIR / "sae_maxpooled_index.csv")
    parser.add_argument("--time-col", default="time_to_progression_or_censor_days")
    parser.add_argument("--event-col", default="progression")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR / "l1_cox_clean")
    parser.add_argument("--n-alphas", type=int, default=100)
    parser.add_argument("--alpha-min-ratio", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=100000)
    run(parser.parse_args())
