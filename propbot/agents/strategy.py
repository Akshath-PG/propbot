"""Strategy agents: the ML experts (trend, ICT, order flow, reversal, news,
geo) and the learned combiner that weighs them (see model.py). This wrapper
trains them on a chosen period and returns per-bar long / short probabilities.
"""
from __future__ import annotations

import copy

import pandas as pd

from ..model import EnsembleModel, train_ensemble


class StrategyAgents:
    def __init__(self, model: EnsembleModel):
        self.model = model

    @classmethod
    def train(cls, ds: pd.DataFrame, settings: dict, train_end: str, verbose: bool = True) -> "StrategyAgents":
        s = copy.deepcopy(settings)
        s["train_end"] = train_end
        return cls(train_ensemble(ds, s, verbose=verbose))

    def signals(self, ds: pd.DataFrame) -> pd.DataFrame:
        X = ds.reset_index(drop=True)
        return self.model.predict(X)

    @property
    def threshold(self) -> float:
        return self.model.threshold

    def weights(self) -> dict:
        return self.model.report["expert_weights_pct"]
