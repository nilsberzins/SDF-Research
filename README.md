# Stochastic Discount Factors via Machine Learning: An Anomaly Portfolio Approach

**Nils Berzins — UCLA Economics Thesis (Winter 2026)**

This project compares six machine learning and statistical models for predicting cross-sectional stock returns and derives Stochastic Discount Factor (SDF) kernels from each. Using 24 years of S&P 500 data and 100+ anomaly portfolio factors, the study evaluates whether deep learning methods outperform classical approaches in asset pricing.

---

## Research Question

Do machine learning models — particularly deep learning architectures — produce SDF kernels with superior cross-sectional explanatory power compared to linear baselines when applied to S&P 500 return prediction?

---

## Models Compared

| Model | Type |
|---|---|
| OLS | Linear regression baseline |
| LASSO | L1-regularized linear regression |
| PCR | Principal Component Regression |
| XGBoost | Gradient boosted trees |
| FNN | Feedforward Neural Network (2×64 hidden layers) |
| LSTM (short) | Long Short-Term Memory, 1-month lookback |
| LSTM (long) | Long Short-Term Memory, 18-month lookback, firm + industry embeddings |

All models are trained under a **walk-forward cross-validation** scheme (6 folds, 2000–2024) with strictly non-overlapping training and test windows to prevent look-ahead bias.

---

## SDF Methodology

For each model and fold, predicted returns are used to construct mean-variance optimal portfolio weights. The SDF realization is:

```
M_{t+1} = 1 - Σ_i  w_{i,t} · R^e_{i,t+1}
```

Firm-level **SDF exposures (β)** and **pricing errors (α)** are estimated via time-series regression using a minimum of 24 monthly observations per firm.

---

## Key Results

| Model | Mean\|α\| | Median\|α\| | Cross-Sectional R² |
|---|---|---|---|
| OLS | 0.01143 | 0.00918 | **0.1138** |
| LASSO | 0.01147 | 0.00920 | 0.0966 |
| XGBoost | 0.01146 | 0.00918 | 0.0946 |
| FNN | 0.01146 | 0.00924 | 0.0925 |
| LSTM (long) | 0.01144 | 0.00915 | 0.0595 |
| PCR | 0.01146 | 0.00919 | 0.0533 |

Simpler linear models achieve higher cross-sectional R² than deep learning approaches, suggesting neural networks may overfit the time-series dimension while missing cross-sectional patterns.

---

## Data Sources

- **WRDS CRSP** — daily S&P 500 stock prices, returns, and volume (107M+ rows, 2000–2024)
- **Fama-French Factors** — market, size, value, profitability, investment, and momentum factors
- **FRED API** — Federal Funds Rate (monthly)
- **Anomaly Portfolio Factors** — 100+ long-short portfolios across 6 categories:
  - Friction (FRIC)
  - Intangibles (INTAN)
  - Investment (INV)
  - Momentum (MOM)
  - Profitability (PROF)
  - Value / Growth (VVG)

Final panel: **749 S&P 500 firms × 138 months = ~65,600 observations**

---

## Repository Structure

```
SDF-Research/
├── SDF-AP Data Current.ipynb   # Main analysis notebook (full pipeline)
├── Anomaly Factors/            # Long-short portfolio factor data
│   ├── friction/
│   ├── intangible/
│   ├── investment/
│   ├── momentum/
│   ├── profitability/
│   └── value_growth/
├── model_cache/                # Cached model predictions and SDF results (gitignored)
├── sp500_full.csv              # Raw CRSP daily data (gitignored)
├── sp500_monthly.joblib        # Monthly panel cache (gitignored)
├── sic-codes.csv               # SIC industry classification
└── SIC_Major_Groups.csv        # SIC major group labels
```

---

## Notebook Pipeline

1. **Data Collection** — Pull CRSP, Fama-French, and FRED data via WRDS and API
2. **Anomaly Factor Integration** — Load and align 100+ anomaly portfolios
3. **Feature Engineering** — Compute rolling volatility, momentum, and reversal signals
4. **Data Aggregation** — Build monthly firm-level panel with joblib checkpointing
5. **Covariance Estimation** — Construct fold-specific covariance matrices (training window only)
6. **Model Training** — Walk-forward CV across 6 folds for all six models
7. **SDF Computation** — Derive SDF kernels from mean-variance optimal weights
8. **Exposure Analysis** — Estimate firm-level α and β via time-series regression
9. **Results & Comparison** — Cross-sectional R², pricing error tables, correlation heatmaps

---

## Requirements

```
pandas
numpy
scikit-learn
torch
xgboost
wrds
fredapi
joblib
matplotlib
seaborn
scipy
```

WRDS access credentials are required for data extraction. Cached `.joblib` files are excluded from version control due to size.

---

## Citation

> Berzins, N. (2026). *Stochastic Discount Factors via Machine Learning: An Anomaly Portfolio Approach*. UCLA Economics Honors Thesis.
