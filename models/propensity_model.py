"""
Propensity scoring model for outbound lead prioritization.

Trains an XGBoost classifier to estimate conversion probability,
then derives T50/T90/T95 threshold metrics for contact scheduling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PropensityConfig:
    """Model hyperparameters and training settings."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    scale_pos_weight: float = 1.0          # set to neg/pos ratio for imbalanced data
    early_stopping_rounds: int = 50
    eval_metric: str = "aucpr"
    n_calibration_folds: int = 5
    random_state: int = 42
    features_to_drop: list[str] = field(default_factory=lambda: ["lead_id", "converted", "partition_date"])


@dataclass
class ThresholdMetrics:
    """T-percentile metrics: contact threshold that captures N% of converters."""

    t50: float   # score cutoff covering 50% of converters
    t90: float   # score cutoff covering 90% of converters
    t95: float   # score cutoff covering 95% of converters
    base_rate: float
    leads_at_t50: float   # % of total leads to contact at T50 threshold
    leads_at_t90: float
    leads_at_t95: float

    def summary(self) -> str:
        return (
            f"Base rate: {self.base_rate:.2%}\n"
            f"T50 → score ≥ {self.t50:.3f} | contact {self.leads_at_t50:.1%} of leads | capture 50% of converters\n"
            f"T90 → score ≥ {self.t90:.3f} | contact {self.leads_at_t90:.1%} of leads | capture 90% of converters\n"
            f"T95 → score ≥ {self.t95:.3f} | contact {self.leads_at_t95:.1%} of leads | capture 95% of converters\n"
        )


class PropensityModel:
    """
    End-to-end propensity scoring pipeline.

    Wraps XGBoost with probability calibration (Platt scaling via CV),
    computes T50/T90/T95 thresholds, and exposes BigQuery-ready scoring output.

    Usage
    -----
    >>> model = PropensityModel(config=PropensityConfig(scale_pos_weight=30))
    >>> model.fit(df_train, label_col="converted")
    >>> scores = model.predict_proba(df_score)
    >>> print(model.thresholds.summary())
    """

    def __init__(self, config: Optional[PropensityConfig] = None):
        self.config = config or PropensityConfig()
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list[str] = []
        self.thresholds: Optional[ThresholdMetrics] = None
        self._eval_results: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, label_col: str = "converted") -> "PropensityModel":
        """
        Train and calibrate the propensity model.

        Parameters
        ----------
        df : DataFrame with features + label column.
        label_col : binary target (0/1).
        """
        X, y = self._split_features(df, label_col)
        self.feature_names = X.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.config.random_state
        )

        base_xgb = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            scale_pos_weight=self.config.scale_pos_weight,
            eval_metric=self.config.eval_metric,
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_state=self.config.random_state,
            use_label_encoder=False,
            n_jobs=-1,
        )

        # Fit with early stopping on validation set
        base_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Isotonic/Platt calibration via cross-validation
        calibrated = CalibratedClassifierCV(
            base_xgb, method="isotonic", cv=self.config.n_calibration_folds
        )
        calibrated.fit(X, y)

        self.pipeline = Pipeline([("model", calibrated)])

        # Compute T-metrics on full train set (will be validated on holdout)
        scores = self.predict_proba(df)
        self.thresholds = self._compute_thresholds(scores, y)

        logger.info("Model trained. %s", self.thresholds.summary())
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Return conversion probability for each row."""
        assert self.pipeline is not None, "Call .fit() before .predict_proba()"
        X = df[self.feature_names]
        probs = self.pipeline.predict_proba(X)[:, 1]
        return pd.Series(probs, index=df.index, name="propensity_score")

    def score_and_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with score + tier label for BigQuery export.

        Tier logic:
            HOT   → score >= T90 threshold
            WARM  → score >= T50 threshold
            COLD  → below T50
        """
        assert self.thresholds is not None, "Call .fit() first"

        scores = self.predict_proba(df)
        tiers = pd.cut(
            scores,
            bins=[-np.inf, self.thresholds.t50, self.thresholds.t90, np.inf],
            labels=["COLD", "WARM", "HOT"],
        )

        out = df[["lead_id"]].copy() if "lead_id" in df.columns else df.index.to_frame()
        out["propensity_score"] = scores.round(6)
        out["lead_tier"] = tiers
        out["score_date"] = pd.Timestamp.today().date()
        return out

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame, label_col: str = "converted") -> dict:
        """Compute AUROC, AUCPR, and lift metrics on a holdout set."""
        X, y = self._split_features(df, label_col)
        scores = self.predict_proba(df.assign(**{col: df[col] for col in df.columns}))

        metrics = {
            "auroc": roc_auc_score(y, scores),
            "aucpr": average_precision_score(y, scores),
            "base_rate": y.mean(),
        }

        # Lift at top decile
        threshold_top10 = np.percentile(scores, 90)
        top10_mask = scores >= threshold_top10
        metrics["lift_top10"] = (y[top10_mask].mean() / y.mean()) if y.mean() > 0 else 0

        logger.info(
            "Holdout — AUROC: %.4f | AUCPR: %.4f | Lift@10%%: %.2fx",
            metrics["auroc"], metrics["aucpr"], metrics["lift_top10"],
        )
        return metrics

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted by gain."""
        assert self.pipeline is not None, "Call .fit() first"

        # Navigate through CalibratedClassifierCV to reach XGBClassifier
        base_estimators = self.pipeline["model"].calibrated_classifiers_
        importances = np.zeros(len(self.feature_names))
        for est in base_estimators:
            xgb_model = est.estimator
            importances += xgb_model.feature_importances_

        importances /= len(base_estimators)

        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save pipeline + metadata via joblib."""
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / "pipeline.joblib")
        joblib.dump(self.thresholds, path / "thresholds.joblib")
        pd.Series(self.feature_names).to_csv(path / "feature_names.csv", index=False)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PropensityModel":
        import joblib
        path = Path(path)
        obj = cls()
        obj.pipeline = joblib.load(path / "pipeline.joblib")
        obj.thresholds = joblib.load(path / "thresholds.joblib")
        obj.feature_names = pd.read_csv(path / "feature_names.csv").iloc[:, 0].tolist()
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_features(self, df: pd.DataFrame, label_col: str):
        drop_cols = [c for c in self.config.features_to_drop + [label_col] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df[label_col].astype(int)
        return X, y

    def _compute_thresholds(self, scores: pd.Series, y: pd.Series) -> ThresholdMetrics:
        """
        Compute T50/T90/T95 score thresholds.

        For each T-value, finds the minimum score such that contacting
        leads above that score captures T% of all converters.
        """
        df = pd.DataFrame({"score": scores, "converted": y}).sort_values("score", ascending=False)
        df["cumulative_converters"] = df["converted"].cumsum() / df["converted"].sum()
        n = len(df)

        def threshold_at(capture_rate: float) -> tuple[float, float]:
            idx = (df["cumulative_converters"] >= capture_rate).idxmax()
            score_cut = df.loc[idx, "score"]
            leads_pct = (df.index.get_loc(idx) + 1) / n
            return float(score_cut), float(leads_pct)

        t50_score, t50_leads = threshold_at(0.50)
        t90_score, t90_leads = threshold_at(0.90)
        t95_score, t95_leads = threshold_at(0.95)

        return ThresholdMetrics(
            t50=t50_score,
            t90=t90_score,
            t95=t95_score,
            base_rate=float(y.mean()),
            leads_at_t50=t50_leads,
            leads_at_t90=t90_leads,
            leads_at_t95=t95_leads,
        )
