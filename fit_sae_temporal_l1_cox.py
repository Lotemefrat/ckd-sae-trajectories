"""Fit L1 Cox to patient_summary.csv from the CKD three-month SAE pipeline.

Install: python -m pip install scikit-survival pandas joblib
Run:
  python fit_sae_temporal_l1_cox.py \
    --input /path/to/topk_SAE/sae_temporal_3m_2y/patient_summary.csv

Use already-scaled neuron_<ID>_change and neuron_<ID>_average predictors.
Fit on train, select alpha by validation Harrell C-index, evaluate once on test.
Report coefficients/HR from that frozen TRAIN fit, not a new fit on test.
HR = exp(coefficient), per +1 in the input's scaled feature, with other
features held fixed. No additional scaling or prevalence filtering is applied.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sksurv
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


FEATURE_PATTERN = re.compile(r"^neuron_(\d+)_(change|average)$")
TIME_COLUMN = "time_to_outcome_or_censor_days"
DEFAULT_INPUT = "/path/to/topk_SAE/sae_temporal_3m_2y/patient_summary.csv"


def c_index(frame, risk):
    return float(concordance_index_censored(
        frame["outcome"].to_numpy(dtype=bool),
        frame[TIME_COLUMN].to_numpy(dtype=float), risk)[0])


def fit_sae_temporal_l1_cox(input_csv=DEFAULT_INPUT, output_dir=None,
                          *, n_alphas=100, alpha_min_ratio=0.001,
                          top_n=50, max_iter=100000):
    """Return metrics; write validation scores, coefficients, HR and test risks.

    Only status='retain' enters modeling. Patients missing any predictor or
    outcome, or with nonpositive/nonfinite follow-up, are excluded and audited.
    Features constant in the analyzed TRAIN set are removed from all splits.
    Ties in validation C-index favor the largest alpha (stronger penalty).
    """
    if n_alphas < 2 or not 0 < alpha_min_ratio < 1 or top_n < 1 or max_iter < 1:
        raise ValueError("Require n_alphas >= 2, 0 < alpha_min_ratio < 1, top_n/max_iter >= 1.")
    input_csv = Path(input_csv).expanduser().resolve()
    out = (Path(output_dir).expanduser().resolve() if output_dir is not None
           else input_csv.parent / "l1_cox_change_average")
    header = pd.read_csv(input_csv, nrows=0).columns.tolist()
    features = [col for col in header if FEATURE_PATTERN.fullmatch(col)]
    metadata = ["PatientID", "split", "status", "outcome", TIME_COLUMN]
    missing = set(metadata) - set(header)
    if missing or not features:
        raise ValueError(f"Missing required columns: {sorted(missing)}; neuron features found: {len(features)}.")
    data = pd.read_csv(input_csv, usecols=metadata + features,
                       dtype={"PatientID": "string"})
    if data.PatientID.isna().any() or data.PatientID.duplicated().any():
        raise ValueError("Input must have one row per nonmissing PatientID.")
    data["split"] = data["split"].astype("string").str.strip().str.lower().replace(
        {"training": "train", "val": "validation"})
    data["status"] = data["status"].astype("string").str.strip().str.lower()
    if not data.status.isin(["retain", "dropped"]).all():
        raise ValueError("Status must be 'retain' or 'dropped'.")
    if not data.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Split must be train, validation (val), or test.")
    reason = pd.Series("", index=data.index, dtype="string")
    reason.loc[data.status.eq("dropped")] = "dropped_by_temporal_cohort"
    retained = data.loc[data.status.eq("retain")].copy()
    retained["outcome"] = pd.to_numeric(retained.outcome, errors="raise")
    if not retained.outcome.dropna().isin([0, 1]).all():
        raise ValueError("Outcome must be 1=progression or 0=censored.")
    retained[TIME_COLUMN] = pd.to_numeric(retained[TIME_COLUMN], errors="raise")
    valid_outcome = retained.outcome.notna()
    valid_time = np.isfinite(retained[TIME_COLUMN]) & retained[TIME_COLUMN].gt(0)
    valid_features = np.isfinite(retained[features].to_numpy(dtype=float)).all(axis=1)
    reason.loc[retained.index[~valid_outcome]] = "missing_outcome"
    reason.loc[retained.index[valid_outcome & ~valid_time]] = "invalid_or_nonpositive_followup"
    reason.loc[retained.index[valid_outcome & valid_time & ~valid_features]] = "missing_or_nonfinite_neuron_summary"
    patients = retained.loc[valid_outcome & valid_time & valid_features].copy()
    split_frames = {split: patients.loc[patients.split.eq(split)].copy()
                    for split in ["train", "validation", "test"]}
    for split, frame in split_frames.items():
        if len(frame) < 2 or not frame.outcome.eq(1).any():
            raise ValueError(f"{split} needs >=2 usable patients and at least one progression event.")
    train = split_frames["train"]
    variable = train[features].nunique().gt(1)
    constant_features = variable.index[~variable].tolist()
    features = variable.index[variable].tolist()
    if not features:
        raise ValueError("No nonconstant training features remain.")
    x_train = train[features].to_numpy(dtype=np.float64)
    y_train = Surv.from_arrays(train.outcome.to_numpy(dtype=bool),
                               train[TIME_COLUMN].to_numpy(dtype=float))
    model = CoxnetSurvivalAnalysis(l1_ratio=1.0, normalize=False,
        n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio, max_iter=max_iter)
    # Do not silently report coefficients from an unconverged solution path.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x_train, y_train)
        except ConvergenceWarning as error:
            raise RuntimeError("Cox path did not converge. Increase --max-iter or --alpha-min-ratio.") from error
    if not np.isfinite(model.coef_).all():
        raise RuntimeError("The fitted coefficient path contains nonfinite values.")
    validation = split_frames["validation"]
    x_validation = validation[features].to_numpy(dtype=np.float64)
    path_rows = []
    for column, alpha in enumerate(model.alphas_):
        score = c_index(validation, model.predict(x_validation, alpha=float(alpha)))
        path_rows.append({"alpha": float(alpha), "validation_c_index": score,
                          "n_nonzero_coefficients": int(np.count_nonzero(model.coef_[:, column]))})
    scores = pd.DataFrame(path_rows)
    # Actual fitted alphas are ordered largest first, so argmax breaks ties that way.
    best_column = int(np.argmax(scores.validation_c_index.to_numpy()))
    best_alpha = float(model.alphas_[best_column])
    scores["selected"] = np.arange(len(scores)) == best_column
    beta = model.coef_[:, best_column]
    with np.errstate(over="ignore"):
        hazard_ratios = np.exp(beta)
    coefficients = pd.DataFrame({
        "feature": features,
        "neuron": [int(FEATURE_PATTERN.fullmatch(col).group(1)) for col in features],
        "summary_type": [FEATURE_PATTERN.fullmatch(col).group(2) for col in features],
        "coefficient": beta, "HR_per_1_scaled_unit": hazard_ratios,
        "abs_coefficient": np.abs(beta), "selected": beta != 0,
    }).sort_values("abs_coefficient", ascending=False, kind="stable")
    selected = coefficients.loc[coefficients.selected].copy()
    top = selected.head(top_n)

    # Alpha is frozen before touching test predictions/performance.
    test = split_frames["test"]
    risk_test = model.predict(test[features].to_numpy(dtype=np.float64), alpha=best_alpha)
    test_c = c_index(test, risk_test)
    train_c = c_index(train, model.predict(x_train, alpha=best_alpha))
    split_counts = {split: {"n_patients": len(frame),
                    "n_events": int(frame.outcome.sum()),
                    "n_excluded": int((data.split.eq(split) & reason.ne("")).sum())}
                    for split, frame in split_frames.items()}
    metrics = {
        "input_csv": str(input_csv), "best_alpha": best_alpha,
        "train_c_index": train_c,
        "validation_c_index": float(scores.iloc[best_column].validation_c_index),
        "test_c_index": test_c, "metric": "Harrell concordance index",
        "n_input_features": len(features) + len(constant_features),
        "n_modeled_features": len(features), "n_selected_features": len(selected),
        "n_selected_neurons": int(selected.neuron.nunique()),
        "constant_train_features_removed": constant_features,
        "n_fitted_alphas": len(model.alphas_), "alpha_min_ratio": alpha_min_ratio,
        "selected_at_smallest_alpha": best_column == len(model.alphas_) - 1,
        "splits": split_counts, "scikit_survival_version": sksurv.__version__,
        "coefficient_source": "train-fitted pure L1 Cox at validation-selected alpha",
        "feature_scaling": "use upstream train-max-scaled summaries directly; no additional scaling",
        "HR_definition": "exp(coefficient); per +1 scaled input unit, other predictors held fixed",
    }
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "alpha_validation.csv", index=False)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    top.to_csv(out / "top_coefficients.csv", index=False)
    predictions = test[metadata].copy()
    predictions["risk_score"] = risk_test
    predictions.to_csv(out / "test_predictions.csv", index=False)
    excluded = data.loc[reason.ne(""), metadata].copy()
    excluded["exclusion_reason"] = reason.loc[reason.ne("")]
    excluded.to_csv(out / "excluded_patients.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Reload: saved=joblib.load(...); saved['model'].predict(X[saved['features']], alpha=saved['alpha'])
    joblib.dump({"model": model, "alpha": best_alpha, "features": features,
                 "coefficient_column": best_column, "metrics": metrics}, out / "model.joblib")
    print(f"Best alpha: {best_alpha:.8g}")
    print(f"Train C-index: {train_c:.4f}")
    print(f"Validation C-index: {metrics['validation_c_index']:.4f}")
    print(f"Test C-index: {test_c:.4f}")
    print(f"Selected {len(selected)} feature coefficients across {selected.neuron.nunique()} neurons.")
    print("\nTop coefficients from the frozen training model:")
    print(top[["feature", "coefficient", "HR_per_1_scaled_unit"]].to_string(index=False))
    print(f"\nSaved results: {out}")
    if metrics["selected_at_smallest_alpha"]:
        print("Best alpha is at the smallest searched value; the validation search may benefit from a lower --alpha-min-ratio.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Temporal patient_summary.csv")
    parser.add_argument("--output-dir", help="Default: l1_cox_change_average beside input CSV")
    parser.add_argument("--n-alphas", type=int, default=100)
    parser.add_argument("--alpha-min-ratio", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=100000)
    args = parser.parse_args()
    fit_sae_temporal_l1_cox(args.input, args.output_dir, n_alphas=args.n_alphas,
        alpha_min_ratio=args.alpha_min_ratio, top_n=args.top_n, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
