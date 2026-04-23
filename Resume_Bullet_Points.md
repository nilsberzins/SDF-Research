# Resume-Ready Bullet Points from SDF-AP Data Analysis

## Model Performance & Accuracy

• Developed and evaluated 6 machine learning models (OLS, LASSO, PCR, XGBoost, FNN, LSTM) for stock return prediction, achieving mean R² scores ranging from 0.54 to 0.88 across different models using walk-forward cross-validation

• Implemented LASSO regression achieving mean R² of 0.84 and XGBoost achieving mean R² of 0.75 on out-of-sample test data, demonstrating strong predictive performance for financial time series

• Built LSTM neural network models with firm and industry embeddings, processing 18-month lookback sequences to predict monthly stock returns with cross-validation across 6 time folds from 2010-2024

• Calculated pricing errors (alphas) for 749 firms across all models, achieving mean absolute alpha values of 0.0114-0.0115, indicating low pricing errors in Stochastic Discount Factor (SDF) models

• Implemented cross-sectional R² analysis achieving values up to 0.11, validating model performance in explaining cross-sectional variation in expected returns

## ETL & Data Engineering

• Engineered comprehensive ETL pipeline extracting 107+ million rows from WRDS CRSP database, integrating S&P 500 stock data, Fama-French factors, Federal Reserve economic data, and 100+ anomaly portfolio factors

• Developed automated data transformation pipeline converting daily, weekly, and monthly financial data, implementing log-return calculations, volatility measures (21-day rolling), and momentum factors (5-day reversal, 52-week momentum)

• Built robust data merging workflow combining multiple heterogeneous data sources (CRSP, FRED API, anomaly portfolios) with proper date alignment, handling missing values and ensuring data consistency across 24+ years of financial data

• Created efficient data caching system using joblib to store 745MB+ of trained models and predictions, reducing computation time from hours to seconds for iterative analysis

• Implemented time-series resampling functions converting daily data to weekly and monthly frequencies, preserving factor relationships and maintaining temporal integrity for panel data analysis

## Tools & Technical Skills

• **Programming Languages:** Python (pandas, numpy, scikit-learn, PyTorch)

• **Machine Learning:** XGBoost, LSTM neural networks, Feedforward Neural Networks, Linear Regression, LASSO, Principal Component Regression (PCR)

• **Data Sources & APIs:** WRDS (Wharton Research Data Services), FRED API (Federal Reserve Economic Data), SQL queries for database extraction

• **Data Processing:** Feature engineering, time-series analysis, panel data manipulation, data cleaning and validation

• **Model Evaluation:** Cross-validation, walk-forward validation, R² metrics, mean squared error, mean absolute error, pricing error analysis

• **Deep Learning:** PyTorch for LSTM implementation with custom embeddings, batch processing, GPU acceleration

## Visualization & Analysis

• Created correlation heatmaps using seaborn to visualize relationships between SDF kernels across 6 different models, revealing correlation patterns ranging from 0.02 to 0.98

• Developed time series visualizations overlaying multiple model predictions to compare SDF kernel trajectories across LSTM, OLS, and XGBoost models over 138 monthly observations

• Generated distribution plots for exposure (beta) coefficients and pricing errors (alpha) across 692 firms, enabling statistical comparison of model performance and risk characteristics

• Produced comprehensive summary statistics tables including mean, median, standard deviation, skewness, and kurtosis for SDF kernels, supporting quantitative model evaluation

## Statistical Analysis & Research

• Implemented Stochastic Discount Factor (SDF) computation using mean-variance optimization, calculating portfolio weights and SDF realizations for 749 firms across 138 monthly periods

• Conducted time-series regression analysis with HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors to calculate firm-level SDF exposures, ensuring robust statistical inference

• Performed cross-sectional analysis comparing model performance across different time periods and market conditions, analyzing 6 walk-forward validation folds spanning 2010-2024

• Calculated and analyzed pricing errors for asset pricing models, achieving consistent alpha distributions across models with standard deviations of approximately 0.014, demonstrating model reliability

## Data Management & Optimization

• Designed and implemented checkpoint system for model persistence, enabling efficient workflow management and reducing computational overhead for large-scale financial modeling

• Optimized data processing pipeline handling 65,655+ panel observations across 749 firms and 138 time periods, ensuring efficient memory usage and computational performance

• Created modular data processing functions for anomaly portfolio extraction, supporting daily and monthly frequencies with automated date parsing and log-return transformations

• Implemented robust error handling and data validation throughout ETL pipeline, ensuring data quality and consistency for downstream modeling tasks
