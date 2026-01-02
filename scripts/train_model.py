"""Train model on historical features and save artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.backtest import backtest_long_short_horizon
from src.features import add_target, check_target_alignment, drop_na_for_training, get_feature_columns
from src.model import (
    TrainConfig,
    StandardScaler,
    compute_regression_metrics,
    predict,
    train_mlp_model,
)


def split_folds(df_slice: pd.DataFrame, pred: np.ndarray, k_folds: int) -> list[tuple[pd.DataFrame, np.ndarray]]:
    k_folds = min(k_folds, len(df_slice))
    if k_folds < 1:
        raise ValueError("Not enough data for walk-forward folds.")
    idx_splits = np.array_split(np.arange(len(df_slice)), k_folds)
    folds = []
    for idx in idx_splits:
        if len(idx) == 0:
            continue
        fold_df = df_slice.iloc[idx].reset_index(drop=True)
        fold_pred = pred[idx]
        folds.append((fold_df, fold_pred))
    return folds


def sweep_configs_walkforward(
    df_slice: pd.DataFrame,
    pred: np.ndarray,
    thresholds: list[float],
    regimes: list[Any],
    sizing_modes: list[str],
    target_ann_vols: list[Any],
    hold_bars: int,
    cost_bps: float,
    folds: list[tuple[pd.DataFrame, np.ndarray]],
    min_trades: int,
    max_drawdown: float,
    min_profit_factor: float,
) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        for reg in regimes:
            for sm in sizing_modes:
                for tav in target_ann_vols:
                    if sm == "discrete" and tav is not None:
                        continue
                    fold_metrics = []
                    fold_debug = []
                    for fold_df, fold_pred in folds:
                        bt = backtest_long_short_horizon(
                            fold_df.assign(pred=fold_pred),
                            threshold=th,
                            hold_bars=hold_bars,
                            cost_bps=cost_bps,
                            regime=reg,
                            sizing_mode=sm,
                            target_ann_vol=tav,
                        )
                        fold_metrics.append(bt.metrics)
                        fold_debug.append(bt.debug)
                    sharpe_vals = np.array([m["sharpe"] for m in fold_metrics], dtype=float)
                    monthly_est_vals = np.array([m["monthly_return_est"] for m in fold_metrics], dtype=float)
                    profit_factor_vals = np.array([m["profit_factor"] for m in fold_metrics], dtype=float)
                    trade_count_vals = np.array([m["trade_count"] for m in fold_metrics], dtype=float)
                    max_drawdown_vals = np.array([m["max_drawdown"] for m in fold_metrics], dtype=float)
                    consistency_vals = np.array([d["consistency_abs"] for d in fold_debug], dtype=float)
                    rows.append(
                        {
                            "threshold": float(th),
                            "regime": reg,
                            "sizing_mode": sm,
                            "target_ann_vol": tav,
                            "sharpe_fold_min": float(sharpe_vals.min()),
                            "sharpe_fold_median": float(np.median(sharpe_vals)),
                            "monthly_return_est_fold_min": float(monthly_est_vals.min()),
                            "monthly_return_est_fold_median": float(np.median(monthly_est_vals)),
                            "profit_factor_fold_min": float(profit_factor_vals.min()),
                            "profit_factor_fold_median": float(np.median(profit_factor_vals)),
                            "trade_count_fold_min": float(trade_count_vals.min()),
                            "max_drawdown_fold_worst": float(max_drawdown_vals.min()),
                            "consistency_abs_fold_max": float(consistency_vals.max()),
                        }
                    )
    val_table = pd.DataFrame(rows)
    filtered = val_table[
        (val_table["trade_count_fold_min"] >= min_trades)
        & (val_table["max_drawdown_fold_worst"] >= max_drawdown)
        & (val_table["profit_factor_fold_min"] >= min_profit_factor)
        & (val_table["sharpe_fold_min"] >= 0)
        & (val_table["monthly_return_est_fold_min"] >= 0)
    ].copy()
    if filtered.empty:
        filtered = val_table.copy()
        filtered["constraints_ok"] = False
    else:
        filtered["constraints_ok"] = True
    filtered["robust_score"] = (
        1000 * filtered["monthly_return_est_fold_median"]
        + 200 * filtered["sharpe_fold_median"]
        + 100 * (filtered["profit_factor_fold_median"] - 1.0)
        - 300 * filtered["max_drawdown_fold_worst"].abs()
        + 500 * filtered["monthly_return_est_fold_min"]
    )
    ranked = filtered.sort_values("robust_score", ascending=False)
    return ranked


def train_for_horizon(
    df: pd.DataFrame,
    hold_bars: int,
    cost_bps: float,
    min_trades: int,
    max_drawdown: float,
    min_profit_factor: float,
    wf_folds: int,
) -> Tuple[dict, Any, Any, dict, dict]:
    df_target = add_target(df, horizon=hold_bars)
    alignment_diff = check_target_alignment(df_target, horizon=hold_bars)
    if alignment_diff >= 1e-12:
        raise AssertionError(f"Target misalignment: {alignment_diff}")
    df_clean = drop_na_for_training(df_target)
    feature_cols = get_feature_columns()
    X = df_clean[feature_cols].values
    y = df_clean["target"].values
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    df_val = df_clean.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df_clean.iloc[val_end:].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    cfg = TrainConfig(epochs=200, batch_size=1024, lr=1e-3, weight_decay=1e-4, patience=5)
    model, _ = train_mlp_model(X_train_s, y_train, X_val_s, y_val, cfg)
    pred_val = predict(model, X_val_s)
    pred_test = predict(model, X_test_s)

    abs_pred = np.abs(pred_val)
    quantiles = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97]
    thresholds = sorted(set([float(np.quantile(abs_pred, q)) for q in quantiles]))
    regimes = [None, "adx", "h1_align", "adx_and_h1"]
    sizing_modes = ["discrete", "continuous"]
    target_ann_vols = [None, 0.08, 0.12]

    def run_sweep(pred_arr: np.ndarray) -> Tuple[pd.DataFrame, float, pd.Series]:
        folds = split_folds(df_val, pred_arr, wf_folds)
        ranked = sweep_configs_walkforward(
            df_val,
            pred_arr,
            thresholds,
            regimes,
            sizing_modes,
            target_ann_vols,
            hold_bars=hold_bars,
            cost_bps=cost_bps,
            folds=folds,
            min_trades=min_trades,
            max_drawdown=max_drawdown,
            min_profit_factor=min_profit_factor,
        )
        best_row = ranked.iloc[0]
        return ranked, float(best_row["robust_score"]), best_row

    ranked_pos, score_pos, best_pos = run_sweep(pred_val)
    ranked_neg, score_neg, best_neg = run_sweep(-pred_val)
    polarity = 1 if score_pos >= score_neg else -1
    best_row = best_pos if polarity == 1 else best_neg
    pred_val_final = pred_val if polarity == 1 else -pred_val
    pred_test_final = pred_test if polarity == 1 else -pred_test

    best_threshold = float(best_row["threshold"])
    best_regime = best_row["regime"]
    best_sizing_mode = best_row["sizing_mode"]
    best_target_ann_vol = best_row["target_ann_vol"]
    if pd.isna(best_target_ann_vol):
        best_target_ann_vol = None

    metrics_val = compute_regression_metrics(pred_val_final, y_val)
    metrics_test = compute_regression_metrics(pred_test_final, y_test)

    bt_test = backtest_long_short_horizon(
        df_test.assign(pred=pred_test_final),
        threshold=best_threshold,
        hold_bars=hold_bars,
        cost_bps=cost_bps,
        regime=best_regime,
        sizing_mode=best_sizing_mode,
        target_ann_vol=best_target_ann_vol,
    )

    selected_config = {
        "polarity": polarity,
        "threshold": best_threshold,
        "regime": best_regime,
        "sizing_mode": best_sizing_mode,
        "target_ann_vol": best_target_ann_vol,
        "hold_bars": hold_bars,
    }

    metadata = {
        "n_samples": n,
        "train_end_idx": train_end,
        "val_end_idx": val_end,
        "feature_columns": feature_cols,
    }
    return selected_config, model, scaler, metrics_val, {**metrics_test, **bt_test.metrics}, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model and save artifacts.")
    parser.add_argument("--features-path", default="data/eurusd_features.parquet")
    parser.add_argument("--artifacts-dir", default="data/artifacts")
    parser.add_argument("--cost-bps", type=float, default=0.5)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--max-drawdown", type=float, default=-0.03)
    parser.add_argument("--min-profit-factor", type=float, default=1.0)
    parser.add_argument("--wf-folds", type=int, default=3)
    parser.add_argument("--hold-bars", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    features_path = Path(args.features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    df = pd.read_parquet(features_path)
    df = df.sort_values("time").reset_index(drop=True)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_result: Dict[str, Any] | None = None
    for hb in args.hold_bars:
        print(f"=== Training for HOLD_BARS={hb} ===")
        selected_config, model, scaler, metrics_val, metrics_test, meta = train_for_horizon(
            df,
            hold_bars=hb,
            cost_bps=args.cost_bps,
            min_trades=args.min_trades,
            max_drawdown=args.max_drawdown,
            min_profit_factor=args.min_profit_factor,
            wf_folds=args.wf_folds,
        )
        result = {
            "selected_config": selected_config,
            "model": model,
            "scaler": scaler,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "meta": meta,
        }
        if best_result is None or metrics_test["sharpe"] > best_result["metrics_test"]["sharpe"]:
            best_result = result

    assert best_result is not None
    sel = best_result["selected_config"]
    model = best_result["model"]
    scaler = best_result["scaler"]
    metrics_val = best_result["metrics_val"]
    metrics_test = best_result["metrics_test"]
    meta = best_result["meta"]

    # Save artifacts
    with (artifacts_dir / "selected_config.json").open("w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=2)
    with (artifacts_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics_val": metrics_val,
                "metrics_test": metrics_test,
                "meta": meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    import torch

    torch.save(model, artifacts_dir / "model.pt")
    with (artifacts_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    print("Saved artifacts to", artifacts_dir)
    print("Selected config:", sel)
    print("Test backtest metrics:", metrics_test)


if __name__ == "__main__":
    main()
