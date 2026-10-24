"""Ensemble of strategy experts + learned combiner, trained walk-forward.

Why not a random 30/70 split: price bars are a time series. A random split
lets the model "see the future" (neighbouring bars are nearly identical),
which makes it look great in testing and fail live. So:

  1. Everything up to train_end (31 Dec 2025) is the training period.
     Inside it we do PURGED WALK-FORWARD folds: train on the past, validate
     on the next block, with a gap so no label overlaps the validation block.
  2. Each expert (trend, ICT, order flow, reversal, news) is a small,
     regularised gradient-boosting model on its own feature group.
  3. Their out-of-fold probabilities feed a logistic-regression combiner;
     its coefficients ARE the learned weights of each expert.
  4. The confidence threshold is chosen on the last validation fold
     (data the combiner was not fitted on) by maximising expectancy.
  5. 2026 data is never touched here - it is the real out-of-sample test.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TF_MINUTES
from .features import EXPERTS, feature_groups

CLASSES = np.array([0, 1, 2])  # short win, none, long win


@dataclass
class EnsembleModel:
    groups: dict
    experts: dict = field(default_factory=dict)
    meta: object = None
    meta_cols: list = field(default_factory=list)
    threshold: float = 0.5
    tradeable: bool = False
    report: dict = field(default_factory=dict)
    settings_snapshot: dict = field(default_factory=dict)

    # ------------------------------------------------------------- inference
    def expert_probs(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for g, mdl in self.experts.items():
            p = _proba(mdl, X[self.groups[g]])
            out[f"{g}_short"] = p[:, 0]
            out[f"{g}_long"] = p[:, 2]
        return pd.DataFrame(out, index=X.index)

    def meta_input(self, X: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
        ctx = X[self.groups["context"]].fillna(0.0)
        return pd.concat([ep, ctx], axis=1)[self.meta_cols]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ep = self.expert_probs(X)
        p = _proba(self.meta, self.meta_input(X, ep))
        res = pd.DataFrame({"p_short": p[:, 0], "p_none": p[:, 1], "p_long": p[:, 2]}, index=X.index)
        res["direction"] = np.where(res["p_long"] >= res["p_short"], 1, -1)
        res["confidence"] = res[["p_long", "p_short"]].max(axis=1)
        return pd.concat([res, ep], axis=1)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "EnsembleModel":
        return joblib.load(path)


def _proba(model, X) -> np.ndarray:
    """predict_proba that always returns 3 columns in CLASSES order."""
    p = model.predict_proba(X)
    cls = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    full = np.zeros((len(X), 3))
    for j, c in enumerate(cls):
        full[:, int(c)] = p[:, j]
    return full


def _expert(params: dict):
    return HistGradientBoostingClassifier(
        max_depth=params["max_depth"], learning_rate=params["learning_rate"],
        max_iter=params["max_iter"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"], early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25, random_state=7)


def _meta(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1), labels=CLASSES)


# --------------------------------------------------------------- threshold
def evaluate_threshold(pred: pd.DataFrame, r_long, r_short, th: float):
    long_m = (pred["p_long"] >= th) & (pred["p_long"] >= pred["p_short"])
    short_m = (pred["p_short"] >= th) & (pred["p_short"] > pred["p_long"])
    r = np.r_[np.asarray(r_long)[long_m.to_numpy()], np.asarray(r_short)[short_m.to_numpy()]]
    if len(r) == 0:
        return 0, 0.0, 0.0, 0.0
    return len(r), float(r.mean()), float(r.sum()), float((r > 0).mean())


def select_threshold(pred, r_long, r_short, months: float, cfg: dict):
    lo, hi, step = cfg["threshold_grid"]
    rows = []
    for th in np.arange(lo, hi + 1e-9, step):
        n, exp, tot, wr = evaluate_threshold(pred, r_long, r_short, th)
        rows.append({"threshold": round(float(th), 3), "trades": n, "trades_per_month": n / max(months, 1e-9),
                     "expectancy_r": exp, "total_r": tot, "win_rate": wr})
    table = pd.DataFrame(rows)
    ok = table[(table["trades_per_month"] >= cfg["min_trades_per_month"]) &
               (table["expectancy_r"] >= cfg["min_expectancy_r"])]
    if len(ok):
        best = ok.loc[ok["total_r"].idxmax()]
        return float(best["threshold"]), True, table
    # nothing meets the bar: pick the most selective useful threshold but flag as not tradeable
    cand = table[table["trades"] >= 30]
    best = cand.loc[cand["expectancy_r"].idxmax()] if len(cand) else table.iloc[-1]
    return float(best["threshold"]), False, table


# --------------------------------------------------------------- training
def walk_forward_folds(times: pd.Series, n_splits: int, purge: pd.Timedelta):
    """Expanding-window folds by calendar time. Yields (train_idx, val_idx)."""
    t0, t1 = times.min(), times.max()
    edges = pd.date_range(t0, t1, periods=n_splits + 2)
    for i in range(1, n_splits + 1):
        vs, ve = edges[i], edges[i + 1]
        tr = np.where(times < vs - purge)[0]
        va = np.where((times >= vs) & (times < ve))[0] if i < n_splits else np.where(times >= vs)[0]
        if len(tr) and len(va):
            yield i, tr, va


def train_ensemble(ds: pd.DataFrame, settings: dict, verbose: bool = True) -> EnsembleModel:
    tcfg = settings["training"]
    train = ds[(ds["time"] <= pd.Timestamp(settings["train_end"]) + pd.Timedelta(days=1)) & ds["y"].notna()]
    train = train.reset_index(drop=True)
    groups = feature_groups(train.columns)
    y = train["y"].astype(int).to_numpy()
    bar_min = TF_MINUTES[settings["timeframe"]]
    purge = pd.Timedelta(minutes=bar_min * (settings["labels"]["horizon_bars"] + tcfg["embargo_bars"]))
    prior = np.bincount(y, minlength=3) / len(y)
    base_ll = _ll(y, np.tile(prior, (len(y), 1)))

    if verbose:
        print(f"Training rows: {len(train):,}  ({train['time'].min()} -> {train['time'].max()})")
        print(f"Class mix short/none/long: {np.round(prior, 3)}   baseline log-loss {base_ll:.4f}")

    oof = pd.DataFrame(index=train.index, dtype=float)
    fold_id = pd.Series(0, index=train.index)
    diag = {g: [] for g in EXPERTS}
    for i, tr, va in walk_forward_folds(train["time"], tcfg["cv_splits"], purge):
        t = time.time()
        for g in EXPERTS:
            cols = groups[g]
            m = _expert(tcfg["expert_params"]).fit(train.loc[tr, cols], y[tr])
            p_tr = _proba(m, train.loc[tr, cols]); p_va = _proba(m, train.loc[va, cols])
            oof.loc[va, f"{g}_short"] = p_va[:, 0]
            oof.loc[va, f"{g}_long"] = p_va[:, 2]
            diag[g].append({"fold": i, "train_ll": _ll(y[tr], p_tr), "val_ll": _ll(y[va], p_va),
                            "val_base_ll": _ll(y[va], np.tile(prior, (len(va), 1))), "iters": m.n_iter_})
        fold_id.loc[va] = i
        if verbose:
            print(f"  fold {i}: train {len(tr):,} rows, validate {len(va):,} rows "
                  f"({train.loc[va[0], 'time'].date()} -> {train.loc[va[-1], 'time'].date()})  {time.time() - t:.0f}s")

    # --------- fit/diagnose the combiner on out-of-fold expert outputs
    has = fold_id > 0
    ctx_cols = groups["context"]
    meta_cols = list(oof.columns) + ctx_cols
    M = pd.concat([oof, train[ctx_cols].fillna(0.0)], axis=1)[meta_cols]
    last = fold_id.max()
    sel_mask = (fold_id == last).to_numpy()
    fit_mask = (has & (fold_id < last)).to_numpy()

    meta_sel = _meta(tcfg["meta_C"]).fit(M[fit_mask], y[fit_mask])
    p_fit = _proba(meta_sel, M[fit_mask]); p_sel = _proba(meta_sel, M[sel_mask])
    pred_sel = pd.DataFrame({"p_short": p_sel[:, 0], "p_long": p_sel[:, 2]})
    sel_rows = train[sel_mask]
    months = (sel_rows["time"].max() - sel_rows["time"].min()).days / 30.4
    th, tradeable, th_table = select_threshold(
        pred_sel, sel_rows["r_long"].to_numpy(), sel_rows["r_short"].to_numpy(), months, tcfg)

    # final combiner on all out-of-fold rows, final experts on all training rows
    meta = _meta(tcfg["meta_C"]).fit(M[has.to_numpy()], y[has.to_numpy()])
    experts = {g: _expert(tcfg["expert_params"]).fit(train[groups[g]], y) for g in EXPERTS}

    # --------- report: weights, over/under-fit checks
    coefs = meta[-1].coef_                      # shape (3, n_features), on standardised inputs
    weight = {}
    for g in EXPERTS:
        idx = [meta_cols.index(f"{g}_short"), meta_cols.index(f"{g}_long")]
        weight[g] = float(np.abs(coefs[:, idx]).sum())
    weight["context"] = float(np.abs(coefs[:, [meta_cols.index(c) for c in ctx_cols]]).sum())
    tot = sum(weight.values())
    weight = {k: round(100 * v / tot, 1) for k, v in weight.items()}

    fit_report = {}
    warn = tcfg["overfit_gap_warn"]
    for g in EXPERTS:
        d = pd.DataFrame(diag[g])
        tr_ll, va_ll, b_ll = d["train_ll"].mean(), d["val_ll"].mean(), d["val_base_ll"].mean()
        verdict = "ok"
        if va_ll - tr_ll > warn:
            verdict = "OVERFIT (train much better than validation)"
        elif va_ll >= b_ll:
            verdict = "UNDERFIT / no edge (not better than guessing)"
        fit_report[g] = {"train_ll": round(tr_ll, 4), "val_ll": round(va_ll, 4),
                         "baseline_ll": round(b_ll, 4), "skill_pct": round(100 * (b_ll - va_ll) / b_ll, 2),
                         "verdict": verdict}
    meta_tr_ll, meta_va_ll = _ll(y[fit_mask], p_fit), _ll(y[sel_mask], p_sel)
    n, exp, tot_r, wr = evaluate_threshold(pred_sel, sel_rows["r_long"], sel_rows["r_short"], th)

    report = {
        "train_period": [str(train["time"].min()), str(train["time"].max())],
        "rows": int(len(train)), "class_prior": prior.round(4).tolist(),
        "experts": fit_report, "expert_weights_pct": weight,
        "combiner": {"fit_ll": round(meta_tr_ll, 4), "selection_ll": round(meta_va_ll, 4),
                     "selection_period": [str(sel_rows['time'].min()), str(sel_rows['time'].max())]},
        "threshold": th, "tradeable": tradeable,
        "selection_fold_at_threshold": {"trades": n, "expectancy_r": round(exp, 3),
                                        "total_r": round(tot_r, 1), "win_rate": round(wr, 3)},
        "threshold_table": th_table.round(4).to_dict("records"),
    }
    model = EnsembleModel(groups=groups, experts=experts, meta=meta, meta_cols=meta_cols,
                          threshold=th, tradeable=tradeable, report=report,
                          settings_snapshot={k: settings[k] for k in ("timeframe", "labels", "train_end", "symbols")})
    return model
