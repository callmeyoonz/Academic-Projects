# Portfolio with Copula and EVT

## Project Overview
This project performs a sophisticated risk assessment of a global investment portfolio consisting of 10 assets across major markets (USA, Europe, Japan, Hong Kong, and China). The objective is to model complex asset dependencies and estimate tail risk measures using advanced econometric frameworks.

## Key Methodologies
* **Volatility Modeling**: Implemented **GARCH** models to capture time-varying volatility and clustering effects in financial time series.
* **Tail Risk Analysis (EVT)**: Applied **Extreme Value Theory** using the **Generalized Pareto Distribution (GPD)** to accurately model the "fat tails" of the return distributions, which standard normal distributions often underestimate.
* **Dependence Modeling (Copulas)**: Utilized **Copula functions** (Gaussian, Student-t, and Archimedean) to analyze non-linear correlations and **lower-tail dependence** (co-movements during market crashes).
* **Factor Analysis (PCA/FA)**: Performed **Principal Component Analysis** and Factor Analysis to identify the underlying common drivers of portfolio returns and reduce dimensionality.
* **Risk Quantification**: Estimated **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** at multiple confidence levels, followed by rigorous backtesting to validate model accuracy.

## Tech Stack
* **Language**: Python
* **Key Libraries**: 
    * `arch`: For GARCH modeling and volatility analysis.
    * `pycop`: For implementing various Copula structures.
    * `scipy.stats` & `statsmodels`: For EVT-GPD fitting and statistical hypothesis testing.
    * `yfinance`: For automated retrieval of global market data.
    * `pandas` & `numpy`: For high-performance financial data manipulation.

## Key Insights
* Confirmed that EVT-based models significantly outperform standard parametric models in capturing extreme loss events.
* Identified strengthened tail dependence among global equities during periods of market stress, highlighting the importance of non-linear correlation modeling in risk management.
