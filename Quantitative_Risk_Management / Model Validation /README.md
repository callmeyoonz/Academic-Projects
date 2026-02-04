# Model Validation 

## Project Overview
This project involves a comprehensive validation and critique of a financial risk management report. The focus was on evaluating the robustness of risk models, the transparency of methodologies, and the accuracy of backtesting procedures.

## Key Tasks & Methodologies
* **Model Documentation Audit**: Evaluated the clarity and completeness of model assumptions, data foundations, and validation procedures.
* **Methodological Critique**: Analyzed the strengths and limitations of various risk frameworks, including:
    * **Historical Simulation (HS)**
    * **GARCH(1,1) - Constant Conditional Correlation (CCC)**
    * **Filtered Historical Simulation (FHS)**
* **Stability Analysis**: Implemented a **Rolling GARCH Beta** approach in Python to detect structural jumps and assess the stability of model parameters over time.
* **Backtesting & Stress Testing**: Reviewed the reliability of VaR forecasts and the effectiveness of stress testing scenarios during crisis periods.

## Technical Implementation (Python)
* **Volatility Analysis**: Used the `arch` library to fit GARCH models and analyze conditional volatility.
* **Statistical Testing**: Performed Kolmogorov-Smirnov tests and analyzed residual autocorrelation to validate model fit.
* **Visualization**: Developed rolling-window plots and correlation matrices to interpret model outcomes and stability.

## Key Findings
* Identified potential weaknesses in the GARCH-CCC model's assumption of constant correlation during market crises.
* Recommended improvements in methodological transparency, particularly regarding EWMA filtering and parameter specification.
