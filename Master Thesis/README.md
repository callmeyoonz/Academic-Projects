# Master's Thesis: Tail Risk Dependencies in East Asian Markets
### A GARCH-EVT-Copula Approach Focused on the Korean Market

This repository contains the implementation of my Master's thesis in Financial Econometrics at Vrije Universiteit Amsterdam. 

## Research Overview
This study investigates the dependence structures of extreme financial risks (Tail Risk) across major East Asian markets (KOSPI, SSE, Nikkei 225) and the S&P 500. By integrating GARCH-family models with Extreme Value Theory (EVT) and Vine Copulas, the research identifies asymmetric tail dependence and contagion effects during periods of high market stress.

## Methodology & Features
- **Volatility Filtering:** Applied `GARCH`, `GJR-GARCH`, and `EGARCH` to capture conditional heteroskedasticity and leverage effects.
- **Tail Modeling:** Utilized **Generalized Pareto Distribution (GPD)** within the EVT framework to model the tails of standardized residuals.
- **Dependence Structure:** Implemented various **Copula models** (Gaussian, Student-t, Clayton, Gumbel, Frank) and **Vine Copulas** to analyze complex multivariate dependencies.
- **Risk Measurement:** Conducted **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** backtesting (Christoffersen test, etc.).


## Key Results
- The **EGARCH-EVT-Copula** model outperformed traditional models in capturing extreme downside risks.
- Strong lower-tail dependence was confirmed between the Korean (KOSPI) and other international markets during crisis periods.
