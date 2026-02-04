# Yelp Data Analysis: Business Trends & Statistical Inference

## Project Overview
This project focuses on extracting insights from the Yelp business dataset by integrating SQL and Python. The analysis includes large-scale database querying, exploratory data analysis (EDA), and rigorous statistical testing to identify trends in business ratings across categories and regions.

## Key Methodologies
* **Database Integration**: Connected Python to a MySQL server to perform complex relational queries on the Yelp dataset (businesses, reviews, and categories).
* **Descriptive Analytics**: Analyzed review trends and rating distributions for over 500,000 observations to establish baseline performance metrics.
* **Non-parametric Statistical Testing**: Conducted **Kolmogorov-Smirnov (K-S) tests** to compare rating distributions between different business sectors (e.g., 'Beauty & Spas' vs. 'Restaurants'), identifying statistically significant performance gaps.
* **Econometric Modeling**: Developed a multiple regression model using **state-wise dummy variables** to quantify the impact of geographic location on business ratings.
* **Hypothesis Testing**: Performed **F-tests** to evaluate the overall significance of regional factors, confirming that business ratings vary significantly across different states.

## Tech Stack
* **Languages**: SQL, Python
* **Database**: MySQL (Connector/Python)
* **Key Libraries**: 
    * `pandas` & `numpy`: For data cleaning and matrix operations.
    * `scipy.stats`: For Kolmogorov-Smirnov tests and statistical inference.
    * `statsmodels`: For regression analysis and F-statistics.
    * `matplotlib`: For visualizing rating distributions and regional trends.

## Key Insights
* Validated a significant difference in average ratings between service-oriented (Beauty & Spas: 3.89) and hospitality-oriented (Restaurants: 3.57) sectors.
* Proved through regression that regional factors play a statistically significant role in business performance, evidenced by a significant F-statistic (26.72) across 28,814 businesses.
