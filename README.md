# outbound-lead-scoring

> Production-grade propensity scoring pipeline for outbound lead prioritization — XGBoost · BigQuery · Dataform · Vertex AI.

---

## Overview

Insurance and financial services outbound teams face a fundamental prioritization problem: a contact list of thousands of leads but bandwidth for only a fraction of calls per day. Random or FIFO ordering wastes broker time on low-probability contacts and lets high-intent leads go cold.

This project implements an end-to-end **propensity scoring system** that ranks outbound leads by their estimated conversion probability. The scored output feeds directly into CRM queues, giving brokers a prioritized, tiered call list every morning.

**Key outputs per lead:**
- `propensity_score` — calibrated probability of conversion (0–1)
- `lead_tier` — HOT / WARM / COLD bucket based on T-metric thresholds
- `top_arguments` — personalized argumentation codes from the argumentation engine
- `broker_priority_rank` — rank within each broker's assigned portfolio
- `suggested_call_window` — MORNING / AFTERNOON / ANY based on historical answer patterns

---

## Architecture

```
Raw Sources                Feature Store              Model Layer             CRM Export
──────────                 ─────────────              ───────────             ──────────
CRM Events    ──▶  BigQuery SQL    ──▶  feat_behavioral   ──▶  XGBoost +     ──▶  mart_lead_scoring
Product Data  ──▶  (behavioral_      feat_rfm               Calibration          (Dataform SQLX)
CRM Attrs         features.sql)      feat_products          (Vertex AI)
                                     feat_crm              │
                                                           ▼
                                                    T50/T90/T95 thresholds
                                                    → tier assignment
```

**Data flow:**
1. Raw CRM events and product data land in BigQuery (partitioned by date)
2. Daily Dataform pipeline computes feature tables (`feat_*`) via rolling window aggregations
3. Vertex AI batch prediction job scores all active leads using the trained XGBoost pipeline
4. `stg_propensity_scores` validates and tiers the raw scores
5. `mart_lead_scoring` joins scores with argumentation, deduplication, and CRM context
6. Downstream: Power BI dashboard + CRM API sync pull from the mart

---

## T50 / T90 / T95 Metrics

Standard classification thresholds (0.5 cutoff) are inappropriate for imbalanced outbound lists with 3–5% base conversion rates. Instead, this project uses **T-percentile metrics**:

| Metric | Definition | Operational meaning |
|--------|-----------|-------------------|
| **T50** | Minimum score threshold that captures 50% of converters | Contact the top X% of leads to reach half of everyone who would convert |
| **T90** | Minimum score threshold that captures 90% of converters | The "safe" contact budget — captures nearly all converters with the smallest list |
| **T95** | Minimum score threshold that captures 95% of converters | Near-exhaustive capture — used for high-value product campaigns |

**Example interpretation:** If T90 = 0.42 and T90 covers 28% of the lead pool, contacting only the top 28% of leads captures 90% of conversions. This translates directly to **contact efficiency gains** reportable to business stakeholders.

Thresholds are recomputed after each model retrain and stored in `stg_propensity_scores.sqlx` for tier assignment.

---

## Project Structure

```
outbound-lead-scoring/
│
├── models/
│   └── propensity_model.py        # XGBoost pipeline, calibration, T-metric computation
│
├── features/
│   ├── feature_engineering.py     # Python feature builders (RFM, behavioral, product, CRM)
│   └── sql/                       # BigQuery SQL run before Python feature engineering
│
├── sql/features/
│   └── behavioral_features.sql    # Rolling window behavioral aggregations (7d/30d/90d)
│
├── dataform/definitions/
│   ├── staging/
│   │   └── stg_propensity_scores.sqlx   # Validates Vertex AI output, assigns tiers
│   └── marts/
│       └── mart_lead_scoring.sqlx       # Final CRM-ready mart with dedup + argumentation
│
├── data/
│   └── generate_sample.py         # Generates synthetic dataset for local development
│
├── .github/workflows/
│   ├── readme_generator.yml       # Auto-updates this README on every push (Claude API)
│   └── pr_reviewer.yml            # Posts automated code review on every PR (Claude API)
│
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/generate_sample.py
# Creates: data/sample/leads.csv, events.csv, products.csv, crm.csv
```

### 3. Build features and train

```python
import pandas as pd
from features.feature_engineering import build_feature_matrix
from models.propensity_model import PropensityModel, PropensityConfig

# Load sample data
df_leads   = pd.read_csv("data/sample/leads.csv")
df_events  = pd.read_csv("data/sample/events.csv")
df_products = pd.read_csv("data/sample/products.csv")
df_crm     = pd.read_csv("data/sample/crm.csv")

# Build feature matrix
df_features = build_feature_matrix(df_leads, df_events, df_products, df_crm)

# Train model
config = PropensityConfig(scale_pos_weight=24)  # neg/pos ratio
model = PropensityModel(config=config)
model.fit(df_features, label_col="converted")

# Inspect T-metric thresholds
print(model.thresholds.summary())

# Score and tier new leads
scored = model.score_and_tier(df_features)
print(scored.head())

# Persist
model.save("artifacts/model_v1")
```

### 4. Evaluate on holdout

```python
metrics = model.evaluate(df_holdout, label_col="converted")
# {'auroc': 0.82, 'aucpr': 0.31, 'base_rate': 0.04, 'lift_top10': 4.7}
```

---

## Key Design Decisions

### Probability calibration
Raw XGBoost scores are well-ranked but miscalibrated — the model's output of 0.7 does not mean "70% chance of converting". Isotonic regression via `CalibratedClassifierCV` corrects this, enabling T-metric thresholds to be operationally meaningful (and stable across retrains).

### Deduplication
The same customer may appear across multiple CRM pipelines (retention, cross-sell, cold outbound). `mart_lead_scoring.sqlx` deduplicates on CPF/CNPJ, keeping only the highest-scored occurrence. Without this, brokers would contact the same customer from different queues, degrading experience and inflating contact counts.

### `scale_pos_weight`
With a 3–5% base rate, naive XGBoost ignores the minority class. `scale_pos_weight ≈ neg/pos ratio` (~20–30x) forces the model to weight false negatives more heavily, recovering recall on converters at the cost of some precision — the right tradeoff for outbound, where missed converters are expensive.

### Rolling windows (7d / 30d / 90d)
Short windows (7d) capture recency signals — a lead who clicked an email yesterday is warm. Long windows (90d) capture frequency and historical engagement. Using all three windows simultaneously lets the model learn different temporal patterns without feature selection bias.

### AUCPR over AUROC
With heavy class imbalance, AUROC is optimistic (dominated by true negatives). Area Under the Precision-Recall Curve (AUCPR) is the correct metric — it directly measures performance on the minority class that matters.

---

## BigQuery / Dataform Integration

The Python model runs offline (local or Vertex AI). The BigQuery/Dataform layer handles:

- **Feature computation at scale** — `behavioral_features.sql` processes millions of events daily using partition-pruned queries, avoiding full table scans
- **Score validation** — `stg_propensity_scores.sqlx` rejects corrupt rows and enforces score bounds before they propagate downstream
- **CRM enrichment** — `mart_lead_scoring.sqlx` is the single source of truth consumed by Power BI and CRM API sync; it abstracts away all upstream complexity

To deploy the Dataform pipeline:
```bash
dataform init bigquery --project-id YOUR_PROJECT --location us-east1
dataform run --tags lead_scoring
```

---

## GitHub Actions Agents

This repo ships with two AI-powered automation workflows:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `readme_generator.yml` | Push to `main` | Reads all `.py`/`.sql`/`.sqlx` files, calls Claude API, auto-updates this README |
| `pr_reviewer.yml` | Pull request opened/updated | Diffs changed files, posts a structured code review comment (data leakage, NULL handling, partition pruning, etc.) |

**Setup required:** Add `ANTHROPIC_API_KEY` in GitHub → Settings → Secrets → Actions.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes — the PR reviewer agent will automatically review your diff
4. Open a pull request against `main`

Code style: `black` + `ruff`. SQL style: uppercase keywords, 4-space indentation, one CTE per logical step.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*README auto-generated by the `readme_generator` GitHub Actions agent using Claude.*
