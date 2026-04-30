# Data Understanding

**CRISP-DM Phase 2**
**Iteration 1 Deliverable**
*Document version 1.0*

---

## Objective

To thoroughly understand the REES46 e-commerce clickstream dataset before
any modelling begins. This phase answers the question: **what does the data
tell us?** The findings here directly inform every decision made in data
preparation and modelling.

---

## 1. Dataset Overview

**Name:** eCommerce Behavior Data from a Multi-Category Store
**Source:** Kaggle — published by REES46
**URL:** https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
**Coverage:** October 2019 and November 2019
**Combined raw size:** 13.67 GB (5.28 GB October, 8.39 GB November)
**Combined Parquet size:** 4.44 GB (1.61 GB October, 2.83 GB November)

**Schema:**

| Column | Data Type | Description |
|---|---|---|
| `event_time` | timestamp | Precise moment of the interaction (UTC) |
| `event_type` | string | view, cart, or purchase |
| `product_id` | integer | Unique product identifier |
| `category_id` | long | Category the product belongs to |
| `category_code` | string | Hierarchical category label — frequently missing |
| `brand` | string | Brand name — absent in some instances |
| `price` | float | Monetary value of the product |
| `user_id` | integer | Persistent identifier across sessions |
| `user_session` | string | Temporary session ID — resets after prolonged inactivity |

---

## 2. Data Quality Assessment

### 2.1 Missing Values

| Feature | Combined nulls | Percentage | Handling |
|---|---|---|---|
| `category_code` | 35,413,780 | 32.21% | Exclude — redundant with `category_id` |
| `brand` | 15,331,243 | 13.94% | Encode missing as "unknown" |
| `user_session` | 12 | 0.00% | Drop rows |
| All others | 0 | 0.00% | No action required |

**Key observation:** Missingness rates are consistent across both months
(category_code: 31.84% October, 32.44% November; brand: 14.40% October,
13.66% November), confirming this is a structural property of the product
catalogue rather than a data collection issue.

### 2.2 Duplicates

| File | Total rows | Duplicates | Percentage |
|---|---|---|---|
| October 2019 | 42,448,764 | 30,220 | 0.071% |
| November 2019 | 67,501,979 | 100,530 | 0.149% |
| Combined | 109,950,743 | 130,750 | 0.119% |

Duplicates are negligible in volume but will be removed in preprocessing
to avoid artificially inflating session event counts.

### 2.3 Anomalous Sessions

| Filter criterion | October | November | Combined |
|---|---|---|---|
| Sessions with < 3 events | 4,810,464 | 7,158,346 | 11,968,810 |
| Bot sessions (> 1 event/sec) | 4,941 | 9,063 | 14,004 |
| Sessions with > 100 distinct products | 359 | 836 | 1,195 |
| **Total to filter** | **4,811,040** | **7,159,996** | **11,971,036** |
| **Percentage of sessions** | **52.04%** | **51.97%** | **52.00%** |
| **Sessions remaining after filter** | ~4,433,381 | ~6,616,054 | ~11,049,435 |

**Key observation:** Approximately 52% of sessions have fewer than 3 events
and will be removed. This is expected — the median session length is 2 events,
meaning the majority of sessions are single or double-event visits with
insufficient behavioural signal for modelling.

---

## 3. Event Distribution

| Event Type | Combined Count | Percentage | October % | November % |
|---|---|---|---|---|
| Product view | 104,335,509 | 94.89% | 96.07% | 94.15% |
| Add to cart | 3,955,446 | 3.60% | 2.18% | 4.49% |
| Purchase | 1,659,788 | 1.51% | 1.75% | 1.36% |

**Key observation — Black Friday effect:** November shows a substantially
higher cart addition rate (4.49% vs 2.18%) but a lower purchase rate
(1.36% vs 1.75%) compared to October. This reflects classic Black Friday
promotional browsing behaviour — users adding items speculatively without
completing purchases. This pattern has direct implications for modelling:
the relationship between cart additions and purchase conversion differs
between the two months, and a month indicator feature should be included
to capture this effect.

---

## 4. Session-Level Statistics

| Statistic | October | November | Combined |
|---|---|---|---|
| Total sessions | 9,244,421 | 13,776,050 | 23,020,471 |
| Sessions with purchase | 629,560 | 773,214 | 1,402,774 |
| Sessions without purchase | 8,614,861 | 13,002,836 | 21,617,697 |
| Purchase session rate (%) | 6.81% | 5.61% | 6.09% |
| Mean events per session | 4.59 | 4.90 | 4.78 |
| Median events per session | 2 | 2 | 2 |
| Max events per session | 1,159 | 4,128 | 4,128 |
| Mean session duration (secs) | 1,041.88 | 931.10 | ~986 |
| Median session duration (secs) | 63.27 | 59.90 | ~61 |
| Mean views per session | 4.41 | 4.61 | 4.53 |
| Mean cart additions per session | 0.1002 | 0.2199 | 0.172 |
| Mean products per session | 2.99 | 3.03 | 3.01 |

**Key observations:**
- The dramatic gap between median (2 events) and mean (4.78 events)
  confirms a heavily right-skewed distribution. Most sessions are brief;
  a small number are extremely long.
- The maximum of 4,128 events in a single session is almost certainly
  automated activity and will be filtered.
- Mean session duration (986 seconds) is substantially higher than median
  (61 seconds), confirming outlier inflation. Log transformation will be
  required for linear models.

---

## 5. Session-Level Class Imbalance

| Metric | Value |
|---|---|
| Total sessions | 23,020,471 |
| Sessions with purchase | 1,402,774 |
| Sessions without purchase | 21,617,697 |
| Purchase session rate | 6.09% |
| Non-purchase session rate | 93.91% |
| Class imbalance ratio | 15.4 : 1 |

**Critical implication:** Raw accuracy is meaningless as an evaluation
metric — a classifier that predicts no purchase for every session would
achieve 93.91% accuracy while providing zero value. Primary evaluation
metrics are AUC-PR, F1-score (purchase class), Geometric Mean, and IBA.

**Imbalance handling strategy:**
- LightGBM: class weight adjustment via `is_unbalance=True`
- All other models: balanced subsampling of the majority class during
  training; evaluation on the full unbalanced test set

---

## 6. User-Level Statistics

| Statistic | Value |
|---|---|
| Total unique users | 5,316,649 |
| Users with at least 1 purchase | 697,470 (13.12%) |
| Users with no purchase | 4,619,179 (86.88%) |
| Mean purchases per user | 0.31 |
| Median purchases per user | 0 |
| Max purchases per user | 640 |

**Purchase concentration:**

| Segment | Users | Revenue | % of Total Revenue |
|---|---|---|---|
| 0 purchases | 4,619,179 | — | — |
| 1 purchase | 402,145 | 107,557,706.58 | 21.3% |
| 2 purchases | 138,677 | — | — |
| 3–5 purchases | 108,043 | — | — |
| > 5 purchases | 48,605 | 207,000,356.79 | 41.0% |
| **Total** | **5,316,649** | **505,152,392.77** | **100%** |

**Key observation:** Heavy buyers (>5 purchases, representing less than 1%
of all users) account for 41% of total revenue. One-time buyers account for
21.3%. This concentration pattern is important context for the intervention
simulation — high-value users responding to an intervention have
disproportionate revenue impact.

---

## 7. Temporal Patterns

### Hour of Day

| Hour | Sessions | Purchase Rate (%) |
|---|---|---|
| 0 | 161,388 | 3.05 |
| 1 | 318,215 | 2.94 |
| 2 | 620,359 | 3.84 |
| 3 | 898,793 | 5.79 |
| 4 | 1,141,140 | 6.63 |
| 5 | 1,267,810 | 6.96 |
| 6 | 1,332,672 | 7.04 |
| 7 | 1,359,781 | 7.09 |
| 8 | 1,405,942 | 7.40 |
| 9 | 1,372,560 | 7.75 |
| 10 | 1,336,487 | 7.56 |
| 11 | 1,298,479 | 7.26 |
| 12 | 1,283,080 | 6.80 |
| 13 | 1,369,354 | 6.15 |
| 14 | 1,464,288 | 5.61 |
| 15 | 1,474,351 | 5.10 |
| 16 | 1,391,928 | 4.83 |
| 17 | 1,233,413 | 4.73 |
| 18 | 927,200 | 4.09 |
| 19 | 597,206 | 4.26 |
| 20 | 341,583 | 4.43 |
| 21 | 198,014 | 4.79 |
| 22 | 122,348 | 4.89 |
| 23 | 100,259 | 4.37 |

**Key observation:** Purchase rate peaks between 08:00–11:00 UTC (7.40%–7.75%)
and drops sharply in the afternoon and evening. Session volume peaks between
14:00–16:00 UTC but purchase rate is lower at these hours, suggesting
afternoon traffic is more browsing-oriented. Hour of day is a meaningful
predictive feature.

### Day of Week

| Day | Sessions | Purchase Rate (%) |
|---|---|---|
| Sunday | 3,319,769 | 8.89 |
| Monday | 2,919,572 | 5.86 |
| Tuesday | 3,151,361 | 5.68 |
| Wednesday | 3,118,585 | 5.88 |
| Thursday | 3,185,740 | 5.60 |
| Friday | 3,636,044 | 4.85 |
| Saturday | 3,685,579 | 5.94 |

**Key observation:** Sunday has the highest purchase rate (8.89%) —
substantially above the weekday average. Friday has the lowest despite
the highest session volume alongside Saturday. The weekend effect is
likely driven by users having more time to complete purchase decisions
rather than just browsing. Day of week is a meaningful predictive feature.

---

## 8. Product and Category Insights

| Statistic | Value |
|---|---|
| Unique products | ~248,968 |
| Unique category IDs | ~892 |
| Unique brands | ~4,080 |
| Unique top-level categories | 12 |
| Mean price | 291.63 |
| Median price | 164.16 |
| 25th percentile price | 67.95 |
| 75th percentile price | 361.24 |
| 95th percentile price | 1,006.67 |
| Min price | 0.00 |
| Max price | 2,574.07 |
| Std dev price | 356.68 |

**Top categories by session volume:**

| Category | Sessions | Purchase Rate (%) |
|---|---|---|
| electronics | 10,076,624 | 7.77 |
| appliances | 2,313,812 | 5.98 |
| computers | 1,219,653 | 4.01 |
| apparel | 866,985 | 1.98 |
| furniture | 823,867 | 2.70 |
| auto | 445,303 | 3.90 |
| kids | 408,498 | 3.16 |
| construction | 362,192 | 4.11 |
| sport | 128,341 | 2.64 |
| accessories | 95,722 | 2.64 |
| medicine | 10,869 | 6.45 |
| stationery | 8,041 | 4.10 |
| country_yard | 8,028 | 1.92 |

**Key observations:**
- Electronics dominates both session volume (44% of all categorised
  sessions) and has the highest purchase rate (7.77%) among high-volume
  categories.
- Apparel has the lowest purchase rate (1.98%) despite being the 4th
  largest category — consistent with high browse-to-buy friction in
  fashion (sizing uncertainty, brand comparison).
- Medicine has a high purchase rate (6.45%) on low volume — users
  arriving with clear intent.
- Price distribution is highly right-skewed — median (164.16) is
  substantially below mean (291.63). Log transformation of price
  features will be needed for linear models.

**Price by event type:**

| Event Type | Mean Price | Median Price | Max Price |
|---|---|---|---|
| view | 291.11 | 163.41 | 2,574.07 |
| cart | 300.25 | 174.88 | 2,574.07 |
| purchase | 304.35 | 174.84 | 2,574.07 |

**Key observation:** Mean price increases from view to cart to purchase,
suggesting users who progress further in the funnel are interacting with
slightly higher-value products. The difference is modest but consistent.

---

## 9. Feature Correlation Analysis

Correlations with purchase outcome (`has_purchase`), computed on a 2%
stratified sample of 497,667 sessions (purchase rate: 6.20%):

| Feature | Correlation with has_purchase |
|---|---|
| `events_per_product` | +0.4370 |
| `cart_count` | +0.3321 |
| `event_count` | +0.1227 |
| `view_count` | +0.0479 |
| `product_count` | -0.0008 |
| `session_duration_secs` | -0.0036 |
| `avg_price` | -0.0058 |
| `distinct_categories` | -0.0106 |
| `price_range` | -0.0209 |

**Key observations:**

- `events_per_product` (0.437) is the strongest correlator. Sessions
  where users spend more events examining individual products are more
  likely to purchase — this reflects deliberate consideration rather
  than casual browsing. This is a primary feature for modelling.
- `cart_count` (0.332) is the second strongest correlator, confirming
  that cart addition is the most direct behavioural signal of purchase
  intent. `cart_to_view_ratio` should be a primary engineered feature.
- `event_count` (0.123) has a positive but moderate correlation —
  longer sessions are slightly more likely to purchase but the
  relationship is weak compared to the quality of engagement signals.
- `product_count` and `session_duration_secs` have near-zero
  correlation — raw counts and duration are poor predictors in isolation.
  Derived ratios (events per product, cart to view ratio) are far more
  informative.
- All price-related features show weak negative correlations — higher
  price range and average price are very slightly associated with lower
  purchase probability, consistent with higher-value items requiring
  more consideration.

---

## 10. Key Findings and Implications for Modelling

### What the data tells us

The REES46 dataset is a real-world e-commerce event log with clean core
fields, structural missingness in two columns, and a heavily imbalanced
binary classification target. The dataset covers a period of normal
e-commerce activity (October) followed by a promotional period (November
— Black Friday and Cyber Monday), which introduces meaningful behavioural
heterogeneity between the two months.

The most important finding for modelling is that **raw engagement metrics
are weak predictors of purchase**. Session duration, event count, and
product count have near-zero or weak correlations with purchase outcome.
The quality and intent of engagement — specifically events concentrated on
individual products and cart additions — are far more predictive. This
means feature engineering must prioritise ratio and interaction features
over raw counts.

### Implications for Data Preparation (Phase 3)

**1. Class imbalance handling is critical**
- Imbalance ratio: 15.4:1
- LightGBM: `is_unbalance=True`
- Other models: balanced subsampling during training
- Test set: full unbalanced distribution
- Primary metrics: AUC-PR, F1-score, Geometric Mean, IBA

**2. Session filtering required**
- Remove: sessions with < 3 events (52% of sessions — insufficient signal)
- Remove: sessions with > 1 event per second (bot activity)
- Remove: sessions with > 100 distinct products (anomalous)
- Remove: 12 null `user_session` rows
- Remove: duplicate rows (130,750 — 0.119%)
- Remove: events occurring after a purchase within a session (data leakage)
- Expected sessions after filtering: ~11,049,435

**3. Training dataset size**
- Full filtered dataset (~11M sessions) is too large for efficient
  training on Colab
- Strategy: keep all purchase sessions (~660,000); downsample
  non-purchase sessions to a 1:3 ratio (~1.98M non-purchase)
- Training set: ~2.64M sessions
- Test set: full filtered dataset (~11M sessions), unbalanced

**4. Feature transformations**
- Log-transform: `event_count`, `session_duration_secs`, `price`
  for linear models (LightGBM handles skew natively)
- Cap: sessions > 100 events truncated to 100 before feature computation

**5. Missing value handling**
- `category_code`: exclude, use `category_id` instead
- `brand`: encode missing as "unknown"

**6. Features to engineer**
Primary features (strongest signal):
- `events_per_product` — events divided by distinct products
- `cart_to_view_ratio` — cart additions divided by views
- `cart_count` — raw cart additions
- `has_cart` — binary: any cart addition in session

Session context features:
- `session_duration_secs`
- `event_count`
- `view_count`
- `product_count`
- `distinct_categories`
- `distinct_brands`
- `avg_price`, `max_price`, `price_range`
- `user_type` (first-time vs returning)
- `has_purchase_so_far` (prior purchase history)

Temporal features:
- `session_start_hour`
- `day_of_week`
- `is_weekend`
- `is_november` (month indicator — captures Black Friday effect)

Sequential features (for flattened and hybrid representations):
- Last N event types encoded as integers
- Time since last event (`time_since_last_event`)
- Price of last N products interacted with

**7. Data representations**
Three representations to be constructed:
- Aggregated: one row per session, summary statistics
- Flattened last N actions: last 1, 5, and 10 events per session,
  padded for shorter sessions
- Hybrid: concatenation of aggregated and flattened

**8. Train-test split**
- 90/10 stratified split preserving class distribution
- Temporal awareness: ensure test set includes sessions from both
  October and November proportionally

**9. Treat as single combined dataset**
- Consistent schema and missingness rates across months
- Add `is_november` binary feature to capture behavioural shift
- Do not model months separately

---

## 11. Notebook Reference

All analysis supporting this document:
- `notebooks/01_data_understanding.ipynb`

---

## 12. Reference

Tokuç, A. A. & Dağ, T. (2025). Predicting User Purchases From Clickstream
Data: A Comparative Analysis of Clickstream Data Representations and Machine
Learning Models. *IEEE Access*, 13, 43796–43817.
https://doi.org/10.1109/ACCESS.2025.3548267

---

*Completed in iteration 1. Updated if subsequent phases reveal findings
that require revision. See git commit history for version changes.*
