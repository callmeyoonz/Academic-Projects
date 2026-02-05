# Financial Econometrics Case Study: Volatility Modeling & Forecasting

This project explores advanced econometric strategies for modeling, measuring, and forecasting financial volatility using high-frequency stock data from **Cisco Systems, Inc. (CSCO)** between 2018 and 2025.

## Project Overview
The goal of this study is to capture market dynamics and microstructure effects by applying various **GARCH** and **GAS** models, integrated with **Realized Kernels (RK)**. We compare different distributions (Normal, Student-t, Skewed Student-t) to identify the most robust model for volatility forecasting.

## Key Features
- **High-Frequency Data Processing:** Cleaning and resampling TAQ data for precise volatility measurement.
- **Volatility Estimation:** Implementing Realized Kernel (RK) and Realized Volatility (RV) to account for market microstructure noise.
- **Econometric Modeling:**
  - **GARCH Family:** Standard GARCH, GJR-GARCH, EGARCH, and Realized GARCH.
  - **GAS (Generalized Autoregressive Score) Models:** Normal, Student-t, Skewed Student-t, and GED distributions.
- **Evaluation:** Model selection via AIC/BIC and performance evaluation using Loss Functions (MSE, QLIKE), Diebold-Mariano tests, and Dynamic Time Warping (DTW).

## Repository Structure
- `data/`: In-sample and Out-of-sample datasets (returns and realized measures).
- `src/`:
  - `data_preprocessing.py`: Data cleaning and resampling logic.
  - `realized_measures.py`: Calculation of Realized Kernel and Volatility.
  - `garch_modeling.py`: Estimation of GARCH-family models.
  - `gas_modeling.py`: Implementation of GAS and Realized GAS models.
  - `evaluation.py`: Forecasting performance metrics and statistical tests.
- `report/`: Full academic report in PDF format.

## Key Findings
- **Realized Kernel GAS (Normal)** model showed superior performance in terms of loss functions.
- **Realized Kernel GARCH (Student-t)** better captured long-term dynamic trends in the out-of-sample period.
- Incorporating high-frequency realized measures significantly improves forecasting accuracy compared to daily returns.

## Contributors
- Abe Tempelman
- Jiaxuan Zhu
- Yunji Eo
- Robin Klaasen
- Vân Lê
