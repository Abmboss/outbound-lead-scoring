"""
Generate synthetic lead scoring dataset for demonstration.

Produces realistic-looking data without exposing any real records.
Run once to create data/sample/*.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_LEADS = 5_000
N_EVENTS = 80_000
OUT_DIR = Path(__file__).parent / "data" / "sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------

lead_ids = [f"L{i:06d}" for i in range(N_LEADS)]

converted = RNG.binomial(1, 0.04, N_LEADS)  # ~4% base conversion rate

days_since_last = RNG.exponential(scale=45, size=N_LEADS).astype(int).clip(1, 365)
last_contact = pd.Timestamp.today() - pd.to_timedelta(days_since_last, unit="D")

df_leads = pd.DataFrame({
    "lead_id": lead_ids,
    "last_contact_date": last_contact.strftime("%Y-%m-%d"),
    "n_contacts_total": RNG.integers(1, 30, N_LEADS),
    "total_premium_brl": np.abs(RNG.lognormal(mean=7.5, sigma=1.2, size=N_LEADS)).clip(0, 50_000),
    "converted": converted,
})
df_leads.to_csv(OUT_DIR / "leads.csv", index=False)
print(f"✓ leads.csv — {len(df_leads):,} rows | base rate: {converted.mean():.2%}")

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

event_types = [
    "CALL_ANSWERED", "CALL_MISSED", "QUOTE_VIEWED",
    "PROPOSAL_SENT", "EMAIL_OPENED", "EMAIL_CLICKED", "PORTAL_LOGIN",
]
event_weights = [0.15, 0.25, 0.20, 0.08, 0.18, 0.09, 0.05]
channels = ["PHONE", "EMAIL", "PORTAL", "WHATSAPP"]

# Converted leads get 2x more events (signal leakage guard: use only pre-conversion events in training)
event_lead_ids = RNG.choice(
    lead_ids,
    size=N_EVENTS,
    p=np.where(converted == 1, 2.0 / N_LEADS, 0.75 / N_LEADS) / (2.0 * converted.mean() + 0.75 * (1 - converted.mean())),
)

event_dates = pd.Timestamp.today() - pd.to_timedelta(
    RNG.integers(0, 90, N_EVENTS), unit="D"
)

df_events = pd.DataFrame({
    "lead_id": event_lead_ids,
    "event_date": event_dates.strftime("%Y-%m-%d"),
    "event_type": RNG.choice(event_types, N_EVENTS, p=event_weights),
    "channel": RNG.choice(channels, N_EVENTS, p=[0.40, 0.30, 0.20, 0.10]),
})
df_events.to_csv(OUT_DIR / "events.csv", index=False)
print(f"✓ events.csv — {len(df_events):,} rows")

# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

products = ["AUTO", "HOME", "LIFE", "HEALTH", "BUSINESS", "TRAVEL"]
rows = []
for lid in lead_ids:
    n_products = RNG.integers(0, 4)
    active_products = RNG.choice(products, n_products, replace=False)
    for p in products:
        rows.append({"lead_id": lid, "product_code": p, "is_active": int(p in active_products)})

df_products = pd.DataFrame(rows)
df_products.to_csv(OUT_DIR / "products.csv", index=False)
print(f"✓ products.csv — {len(df_products):,} rows")

# ---------------------------------------------------------------------------
# CRM enrichment
# ---------------------------------------------------------------------------

segments = ["INDIVIDUAL", "SME", "CORPORATE"]
sources = ["BROKER_REFERRAL", "DIGITAL_CAMPAIGN", "COLD_OUTBOUND", "RETENTION", "CROSS_SELL"]
ufs = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "PE", "CE"]
income_bands = ["A", "B", "C", "D", "E"]

df_crm = pd.DataFrame({
    "lead_id": lead_ids,
    "broker_segment": RNG.choice(segments, N_LEADS, p=[0.50, 0.35, 0.15]),
    "lead_source": RNG.choice(sources, N_LEADS, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
    "uf": RNG.choice(ufs, N_LEADS, p=[0.35, 0.15, 0.12, 0.08, 0.07, 0.06, 0.06, 0.04, 0.04, 0.03]),
    "customer_age": RNG.integers(18, 75, N_LEADS),
    "customer_income_band": RNG.choice(income_bands, N_LEADS, p=[0.10, 0.20, 0.40, 0.20, 0.10]),
    "campaign_id": RNG.choice([f"CMP_{i:04d}" for i in range(50)], N_LEADS),
})
df_crm.to_csv(OUT_DIR / "crm.csv", index=False)
print(f"✓ crm.csv — {len(df_crm):,} rows")

print(f"\nAll sample data written to {OUT_DIR.resolve()}")
