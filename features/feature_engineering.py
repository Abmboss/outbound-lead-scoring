"""
Feature engineering pipeline for outbound lead scoring.

Transforms raw CRM + behavioral + product data into a flat feature
matrix ready for XGBoost training or BigQuery scoring.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECENCY_WINDOWS = [7, 30, 90]          # days for rolling behavioral features
MONETARY_BINS = [0, 500, 2000, 5000, 15000, np.inf]
MONETARY_LABELS = ["micro", "low", "mid", "high", "premium"]


# ---------------------------------------------------------------------------
# RFM features
# ---------------------------------------------------------------------------

def build_rfm_features(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary features per lead.

    Expected columns: lead_id, last_contact_date, n_contacts_total, total_premium_brl
    """
    ref = reference_date or pd.Timestamp.today()

    out = df[["lead_id"]].copy()
    out["recency_days"] = (ref - pd.to_datetime(df["last_contact_date"])).dt.days.clip(0, 730)
    out["frequency_contacts"] = df["n_contacts_total"].clip(0, 50)
    out["monetary_premium"] = np.log1p(df["total_premium_brl"].clip(0))

    # Binned monetary tier (categorical → ordinal)
    out["monetary_tier"] = pd.cut(
        df["total_premium_brl"],
        bins=MONETARY_BINS,
        labels=range(len(MONETARY_LABELS)),
        right=False,
    ).astype(float)

    # RFM score (simple composite — not used as input to model, used for segment labeling)
    r_score = pd.cut(out["recency_days"], bins=[0, 7, 30, 90, 180, 730], labels=[5, 4, 3, 2, 1]).astype(float)
    f_score = pd.cut(out["frequency_contacts"], bins=[-1, 1, 3, 7, 15, 50], labels=[1, 2, 3, 4, 5]).astype(float)
    m_score = out["monetary_tier"]
    out["rfm_composite"] = (r_score + f_score + m_score) / 3

    return out


# ---------------------------------------------------------------------------
# Behavioral features (rolling window aggregations)
# ---------------------------------------------------------------------------

def build_behavioral_features(
    df_events: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Aggregate event-level data into per-lead behavioral features.

    Expected columns: lead_id, event_date, event_type, channel
    Event types: CALL_ANSWERED, CALL_MISSED, QUOTE_VIEWED, PROPOSAL_SENT,
                 EMAIL_OPENED, EMAIL_CLICKED, PORTAL_LOGIN
    """
    ref = reference_date or pd.Timestamp.today()
    df = df_events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["days_ago"] = (ref - df["event_date"]).dt.days

    features = []
    for lead_id, grp in df.groupby("lead_id"):
        row: dict = {"lead_id": lead_id}

        for window in RECENCY_WINDOWS:
            w = grp[grp["days_ago"] <= window]
            pfx = f"w{window}d_"

            row[pfx + "n_events"] = len(w)
            row[pfx + "n_calls_answered"] = (w["event_type"] == "CALL_ANSWERED").sum()
            row[pfx + "n_calls_missed"] = (w["event_type"] == "CALL_MISSED").sum()
            row[pfx + "n_quotes_viewed"] = (w["event_type"] == "QUOTE_VIEWED").sum()
            row[pfx + "n_proposals"] = (w["event_type"] == "PROPOSAL_SENT").sum()
            row[pfx + "n_email_open"] = (w["event_type"] == "EMAIL_OPENED").sum()
            row[pfx + "n_email_click"] = (w["event_type"] == "EMAIL_CLICKED").sum()
            row[pfx + "n_portal_login"] = (w["event_type"] == "PORTAL_LOGIN").sum()
            row[pfx + "channel_diversity"] = w["channel"].nunique()

        # Answer rate: signals lead responsiveness
        total_calls = (grp["event_type"].isin(["CALL_ANSWERED", "CALL_MISSED"])).sum()
        answered = (grp["event_type"] == "CALL_ANSWERED").sum()
        row["answer_rate_all_time"] = answered / total_calls if total_calls > 0 else 0

        # Days since last engagement
        engagement_events = ["CALL_ANSWERED", "QUOTE_VIEWED", "EMAIL_CLICKED", "PORTAL_LOGIN"]
        engaged = grp[grp["event_type"].isin(engagement_events)]
        row["days_since_last_engagement"] = (
            (ref - engaged["event_date"].max()).days if not engaged.empty else 999
        )

        features.append(row)

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# Product propensity features (cross-sell signals)
# ---------------------------------------------------------------------------

def build_product_features(df_products: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode active product portfolio per lead.

    Expected columns: lead_id, product_code, is_active
    """
    active = df_products[df_products["is_active"] == 1]
    pivot = (
        active.pivot_table(index="lead_id", columns="product_code", values="is_active", fill_value=0)
        .add_prefix("has_product_")
        .reset_index()
    )

    # Count of active products (breadth of relationship)
    pivot["n_active_products"] = pivot.filter(like="has_product_").sum(axis=1)
    return pivot


# ---------------------------------------------------------------------------
# CRM enrichment features
# ---------------------------------------------------------------------------

def build_crm_features(df_crm: pd.DataFrame) -> pd.DataFrame:
    """
    Encode CRM attributes: broker segment, lead source, campaign, region.

    Expected columns: lead_id, broker_segment, lead_source, campaign_id,
                      uf, customer_age, customer_income_band
    """
    out = df_crm[["lead_id"]].copy()

    # Ordinal income band
    income_map = {"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}
    out["income_band"] = df_crm["customer_income_band"].map(income_map).fillna(-1)

    # Age bins
    out["age_group"] = pd.cut(
        df_crm["customer_age"],
        bins=[0, 25, 35, 45, 60, 120],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # Dummies for categorical dimensions
    dummies = pd.get_dummies(
        df_crm[["broker_segment", "lead_source"]],
        prefix=["seg", "src"],
        drop_first=False,
        dtype=float,
    )
    out = pd.concat([out, dummies], axis=1)

    return out


# ---------------------------------------------------------------------------
# Master feature assembler
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df_leads: pd.DataFrame,
    df_events: pd.DataFrame,
    df_products: pd.DataFrame,
    df_crm: pd.DataFrame,
    label_col: Optional[str] = "converted",
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Join all feature sets into a single flat matrix.

    Parameters
    ----------
    df_leads    : base lead table (lead_id, last_contact_date, n_contacts_total, total_premium_brl)
    df_events   : event log (lead_id, event_date, event_type, channel)
    df_products : product portfolio (lead_id, product_code, is_active)
    df_crm      : CRM enrichment (lead_id, broker_segment, lead_source, ...)
    label_col   : target column in df_leads (None for scoring-only runs)
    """
    ref = reference_date or pd.Timestamp.today()
    logger.info("Building feature matrix for %d leads (ref_date=%s)", len(df_leads), ref.date())

    rfm = build_rfm_features(df_leads, ref)
    behavioral = build_behavioral_features(df_events, ref)
    products = build_product_features(df_products)
    crm = build_crm_features(df_crm)

    matrix = (
        df_leads[["lead_id"]].merge(rfm, on="lead_id", how="left")
        .merge(behavioral, on="lead_id", how="left")
        .merge(products, on="lead_id", how="left")
        .merge(crm, on="lead_id", how="left")
    )

    # Fill NaN from missing joins
    bool_cols = matrix.filter(like="has_product_").columns
    matrix[bool_cols] = matrix[bool_cols].fillna(0)
    matrix = matrix.fillna(-1)

    # Re-attach label for training runs
    if label_col and label_col in df_leads.columns:
        matrix[label_col] = df_leads.set_index("lead_id")[label_col].reindex(matrix["lead_id"]).values

    logger.info("Feature matrix shape: %s", matrix.shape)
    return matrix
