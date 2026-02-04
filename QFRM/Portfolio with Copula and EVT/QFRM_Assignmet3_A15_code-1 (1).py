
# Group A14 
# Assignment 3 

###########################################################
### Imports
import numpy as np
import pandas as pd
import yfinance as yf
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import scipy.stats as sts
import seaborn as sns
from scipy.stats import norm, t
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import LinearRegression
from pycop import  gaussian, student, archimedean, estimation, simulation
import scipy.stats as stats
from scipy.stats import genpareto
from arch import arch_model
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm 
from sklearn.decomposition import PCA

###########################################################
### Data Description & Copula part 
###########################################################

def process_stock_data(tickers, start_date, end_date):
    
    # Download adjusted close prices
    raw_stock_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    # Drop rows with too many missing values (more than 2)
    cleaned_stock_data = raw_stock_data.dropna(thresh=len(tickers) - 2)
    
    # Fill missing values using linear interpolation and backward fill for head
    cleaned_stock_data = cleaned_stock_data.interpolate(method='linear', axis=0).bfill(axis=0)
    
    return cleaned_stock_data

def plot_stock_data(stock_data):
    
    # Define the custom order for grouping (each pair per row)
    ordered_tickers = [
        '0700.HK', '0005.HK',        # Row 1: HKD
        '601318.SS', '600519.SS',    # Row 2: CNY 
        '6758.T', '9984.T',          # Row 3: JPY 
        'AAPL', 'MSFT',              # Row 4: USD
        'SAP.DE', 'ADS.DE'           # Row 5: EUR
    ]

    currency_map = {
        '0700.HK': 'HKD', '0005.HK': 'HKD',
        '601318.SS': 'CNY', '600519.SS': 'CNY',
        '6758.T': 'JPY', '9984.T': 'JPY',
        'AAPL': 'USD', 'MSFT': 'USD',
        'SAP.DE': 'EUR', 'ADS.DE': 'EUR'
    }

    # Ensure stock_data is ordered according to our custom layout
    stock_data = stock_data[ordered_tickers]

    num_stocks = len(ordered_tickers)
    ncols = 2
    nrows = num_stocks // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 4 * nrows))
    axes = axes.flatten()

    for i, column in enumerate(ordered_tickers):
        axes[i].plot(stock_data.index, stock_data[column], label=column, linewidth=2)
        axes[i].set_xlabel('Date', fontsize=12)
        axes[i].set_ylabel(f'Stock Price ({currency_map.get(column, "")})', fontsize=12)
        axes[i].set_title(f'{column} Stock Price Over Time', fontsize=14, fontweight='bold')
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
def compute_cont_comp_returns(stock_data):
    
    returns_df = np.log(stock_data / stock_data.shift(1)) * 100
    returns_df = returns_df.dropna()
    
    return returns_df

def plot_stock_returns(returns_df):
    
    # Define the custom order for grouping (each pair per row)
    ordered_tickers = [
        '0700.HK', '0005.HK',        # Row 1: HKD
        '601318.SS', '600519.SS',    # Row 2: CNY 
        '6758.T', '9984.T',          # Row 3: JPY 
        'AAPL', 'MSFT',              # Row 4: USD
        'SAP.DE', 'ADS.DE'           # Row 5: EUR
    ]

    # Ensure returns_df is ordered according to our custom layout
    returns_df = returns_df[ordered_tickers]

    num_stocks = len(ordered_tickers)
    ncols = 2
    nrows = num_stocks // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 4 * nrows), sharex=True)
    axes = axes.flatten()

    # Find global min and max for consistent y-axis scaling
    global_min = returns_df.min().min()
    global_max = returns_df.max().max()
    y_margin = 0.05 * (global_max - global_min)

    for i, column in enumerate(ordered_tickers):
        axes[i].plot(returns_df.index, returns_df[column], label=f'{column} Returns',
                     color='darkblue', linewidth=1.5)
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[i].set_ylabel('Returns (%)', fontsize=12)
        axes[i].set_xlabel('Date', fontsize=12)
        axes[i].set_title(f'{column} Continuously Compounded Returns', fontsize=14, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(global_min - y_margin, global_max + y_margin)

    plt.tight_layout()
    plt.show()
    
def descriptive_statistics(returns_df):
    
    desc_stats = returns_df.describe()
    skewness = returns_df.skew()
    kurtosis = returns_df.kurtosis()
    desc_stats.loc['skewness'] = skewness
    desc_stats.loc['kurtosis'] = kurtosis
    print("Descriptive Statistics:")
    print(desc_stats)
    
def plot_histograms_with_normal(returns_df):
    
    # Define the custom order for grouping (each pair per row)
    ordered_tickers = [
        '0700.HK', '0005.HK',        # Row 1: HKD
        '601318.SS', '600519.SS',    # Row 2: CNY 
        '6758.T', '9984.T',          # Row 3: JPY 
        'AAPL', 'MSFT',              # Row 4: USD
        'SAP.DE', 'ADS.DE'           # Row 5: EUR
    ]

    # Reorder the DataFrame
    returns_df = returns_df[ordered_tickers]

    num_stocks = len(ordered_tickers)
    ncols = 2
    nrows = num_stocks // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, column in enumerate(ordered_tickers):
        ax = axes[i]
        
        # Histogram of returns
        ax.hist(returns_df[column], bins=100, density=True, alpha=0.6, edgecolor='white')
        
        # Fit a normal distribution to the data
        mu, std = sts.norm.fit(returns_df[column])
        x = np.linspace(-10, 10, 1000)
        p = sts.norm.pdf(x, mu, std)
        ax.plot(x, p, color='orange', linewidth=3, linestyle='-')
        
        ax.set_xlim(-10, 10)
        ax.set_title(f'{column} Returns Histogram with Normal Fit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Returns (%)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(['Normal Fit', 'Histogram'])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
def plot_histograms_with_student_t(returns_df):
    
    # Define the custom order for grouping (each pair per row)
    ordered_tickers = [
        '0700.HK', '0005.HK',        # Row 1: HKD
        '601318.SS', '600519.SS',    # Row 2: CNY 
        '6758.T', '9984.T',          # Row 3: JPY 
        'AAPL', 'MSFT',              # Row 4: USD
        'SAP.DE', 'ADS.DE'           # Row 5: EUR
    ]
    
    # Reorder the DataFrame
    returns_df = returns_df[ordered_tickers]

    num_stocks = len(ordered_tickers)
    ncols = 2
    nrows = num_stocks // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, column in enumerate(ordered_tickers):
        ax = axes[i]
        
        # Histogram of returns
        ax.hist(returns_df[column], bins=100, density=True, alpha=0.6, edgecolor='white')
        
        # Fit a Student's t-distribution to the data
        params = t.fit(returns_df[column].dropna())
        df, loc, scale = params[0], params[1], params[2]
        
        # Generate the t-distribution PDF
        x = np.linspace(-10, 10, 1000)
        p = t.pdf(x, df, loc, scale)
        
        # Plot the fitted Student's t-distribution
        ax.plot(x, p, color='orange', linewidth=3, linestyle='-')
        
        # Set axis limits
        ax.set_xlim(-10, 10)
        ax.set_title(f'{column} Returns Histogram with Student\'s t Fit (nu={df:.2f})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Returns (%)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(['Student\'s t Fit', 'Histogram'])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_scatter_return_pairs(returns_df, pairs, figsize=(16, 16), wspace=0.1, hspace=0.1, hist_color='blue'):
    
    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(2, 2, figure=fig, wspace=wspace, hspace=hspace)

    for i, (vX1, vX2) in enumerate(pairs):
        row, col = divmod(i, 2)
        inner_gs = GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_gs[row, col])

        # Drop NaNs and align indexes
        x = returns_df[vX1]
        y = returns_df[vX2]
        df_pair = returns_df[[vX1, vX2]].dropna()
        x = df_pair[vX1]
        y = df_pair[vX2]

        # Compute 2D KDE density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # Sort by density for better visibility
        idx = z.argsort()
        x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

        ax_main = fig.add_subplot(inner_gs[1:4, 0:3])
        ax_xhist = fig.add_subplot(inner_gs[0, 0:3], sharex=ax_main)
        ax_yhist = fig.add_subplot(inner_gs[1:4, 3], sharey=ax_main)

        # Density scatter plot
        scatter = ax_main.scatter(x, y, c=z, s=15, cmap='viridis', edgecolor='none', alpha=0.8, label='Data')
        ax_main.set_xlabel(f'{vX1} Returns (%)', fontsize=10)
        ax_main.set_ylabel(f'{vX2} Returns (%)', fontsize=10)
        ax_main.set_title(f'{vX1} vs {vX2}', fontsize=12)
        ax_main.grid(True, linestyle='--', alpha=0.5)

        # Linear regression (OLS line)
        #slope, intercept, r_value, p_value, std_err = linregress(x, y)
        #line_x = np.linspace(x.min(), x.max(), 100)
        #line_y = slope * line_x + intercept
        #ax_main.plot(line_x, line_y, color='red', label='OLS', linewidth=2)

        # Add colorbar in the bottom-right corner of the subplot
        #cbar = fig.colorbar(scatter, ax=ax_main, pad=0.01, shrink=0.6)
        #cbar.set_label('Density')

        # Add the legend for OLS and Data Points
        ax_main.legend(loc='upper left', fontsize=8)

        # Histograms
        ax_xhist.hist(x, bins=60, color=hist_color, alpha=0.6)
        ax_xhist.axis('off')
        ax_yhist.hist(y, bins=60, orientation='horizontal', color=hist_color, alpha=0.6)
        ax_yhist.axis('off')

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    plt.show()
    
def plot_cdfs_of_pairs(returns_df, pairs):
   
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (vX1, vX2) in enumerate(pairs):
        ax = axes[i]

        # Drop missing data and align
        df_pair = returns_df[[vX1, vX2]].dropna()
        x = df_pair[vX1]
        y = df_pair[vX2]

        # Compute empirical CDFs
        ecdf_x = ECDF(x)
        ecdf_y = ECDF(y)

        # Plot the CDFs
        ax.plot(ecdf_x.x, ecdf_x.y, label=f'{vX1} ECDF', linewidth=2)
        ax.plot(ecdf_y.x, ecdf_y.y, label=f'{vX2} ECDF', linewidth=2)

        ax.set_title(f'Empirical CDFs: {vX1} & {vX2}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Returns (%)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
def estimate_all_copulas(returns_df, pairs, copula_families=["gaussian", "student_t", "gumbel", "clayton"]):

    results = []

    for vX1, vX2 in pairs:
        data = returns_df[[vX1, vX2]].dropna().T.values

        for family in copula_families:
            if family == "gaussian":
                cop = gaussian()
                param, cmle = estimation.fit_cmle(cop, data)
                results.append({
                    "pair": f"{vX1}-{vX2}",
                    "copula": "Gaussian",
                    "param1": param[0],  # rho
                    "param2": None,
                    "loglik": cmle
                })

            elif family == "student_t":
                cop = student()
                param, cmle = estimation.fit_cmle(cop, data)
                results.append({
                    "pair": f"{vX1}-{vX2}",
                    "copula": "Student-t",
                    "param1": param[0],  # rho
                    "param2": param[1],  # nu
                    "loglik": cmle
                })

            elif family == "gumbel":
                cop = archimedean(family="gumbel")
                param, cmle = estimation.fit_cmle(cop, data)
                results.append({
                    "pair": f"{vX1}-{vX2}",
                    "copula": "Gumbel",
                    "param1": param[0],  # theta
                    "param2": None,
                    "loglik": cmle
                })

            elif family == "clayton":
                cop = archimedean(family="clayton")
                param, cmle = estimation.fit_cmle(cop, data)
                results.append({
                    "pair": f"{vX1}-{vX2}",
                    "copula": "Clayton",
                    "param1": param[0],  # theta
                    "param2": None,
                    "loglik": cmle
                })

    return pd.DataFrame(results)

def evaluate_all_copulas_by_aic(copula_results_df):

    def get_num_params(row):
        return 2 if row["copula"] == "Student-t" else 1

    full_df = copula_results_df.copy()
    full_df["num_params"] = full_df.apply(get_num_params, axis=1)
    full_df["aic"] = 2 * full_df["num_params"] - 2 * full_df["loglik"]

    best_df = full_df.loc[full_df.groupby("pair")["aic"].idxmin()].reset_index(drop=True)

    return full_df.reset_index(drop=True), best_df

def compute_dependence_measures_for_pairs(returns_df, pairs):
   
    results = []

    for vX1, vX2 in pairs:
        df_pair = returns_df[[vX1, vX2]].dropna()
        x = df_pair[vX1]
        y = df_pair[vX2]

        rho_pearson = sts.pearsonr(x, y).statistic
        tau_kendall = sts.kendalltau(x, y).statistic
        rho_spearman = sts.spearmanr(x, y).statistic

        results.append({
            "Pair": f"{vX1} - {vX2}",
            "Pearson": rho_pearson,
            "Kendall": tau_kendall,
            "Spearman": rho_spearman
        })

    result_df = pd.DataFrame(results)
    print("\nDependence Measures (Pearson, Kendall, Spearman):")
    print(result_df.round(4))

    return result_df

def compute_tail_dependence(copula_df):
    
    tail_results = []

    for _, row in copula_df.iterrows():
        pair = row['pair']
        copula = row['copula']
        param1 = row['param1']
        param2 = row['param2']

        if copula == "Gaussian":
            lambda_l = lambda_u = 0.0

        elif copula == "Student-t":
            rho = param1
            nu = param2
            t_cdf = sts.t(df=nu + 1)
            z = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            lambda_l = lambda_u = 2 * t_cdf.cdf(z)

        elif copula == "Gumbel":
            theta = param1
            lambda_l = 0.0
            lambda_u = 2 - 2**(1.0 / theta)

        elif copula == "Clayton":
            theta = param1
            if theta > 0:
                lambda_l = 2 ** (-1 / theta)
            else:
                lambda_l = 0
            lambda_u = 0

        tail_results.append({
            "pair": pair,
            "copula": copula,
            "λ_L (Lower Tail)": round(lambda_l, 4),
            "λ_U (Upper Tail)": round(lambda_u, 4)
        })

    tail_df = pd.DataFrame(tail_results)
    print("\nTail Dependence Estimates:")
    print(tail_df)

    return tail_df

def sct_plt(u1, u2, vX1, vX2, family, rho=None, nu=None, theta=None, ax=None):
    
    # Transform u1, u2 from uniform to normal scale
    u1 = norm.ppf(u1)
    u2 = norm.ppf(u2)

    # If no axis is passed, create one
    if ax is None:
        ax = plt.gca()

    # Plot the scatter plot
    ax.scatter(u1, u2, color="purple", alpha=0.5, edgecolors='white', linewidth=0.5, label='Simulated Data')

    # Add a KDE contour plot
    kde = sns.kdeplot(x=u1, y=u2, ax=ax, cmap="Blues", fill=True, levels=10, alpha=0.3)

    # Add a colorbar
    cbar = plt.colorbar(kde.collections[0], ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=15)

    # Set labels and title
    ax.set_xlabel(f'{vX1} Returns (%)', fontsize=12)
    ax.set_ylabel(f'{vX2} Returns (%)', fontsize=12)
    ax.legend(loc='upper left')

    # Dynamically build the title
    title = f'{vX1} - {vX2} | {family.capitalize()} Copula'
    if rho is not None:
        title += f', ρ = {rho:.2f}'
    if nu is not None:
        title += f', ν = {nu:.2f}'
    if theta is not None:
        title += f', θ = {theta:.2f}'

    ax.set_title(title, fontsize=14, fontweight='bold')

def simulate_and_plot_copulas(returns_df, pairs, copula_families=["gaussian", "student_t", "gumbel", "clayton"]):
    
    for vX1, vX2 in pairs:
        copula_results = estimate_all_copulas(returns_df, [(vX1, vX2)], copula_families)
        copula_result = copula_results[(copula_results["pair"] == f"{vX1}-{vX2}")]

        # Create a 2x2 grid for the subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()  # Flatten the 2x2 array of axes for easy iteration
        
        # Plot each copula in the corresponding subplot
        for i, family in enumerate(copula_families):
            ax = axes[i]
            if family == "gaussian":
                rho = copula_result["param1"].iloc[0]
                corrMatrix = np.array([[1, rho], [rho, 1]])
                u1, u2 = simulation.simu_gaussian(2, len(returns_df), corrMatrix)
                sct_plt(u1, u2, vX1, vX2, family, rho=rho, ax=ax)

            elif family == "student_t":
                rho = copula_result["param1"].iloc[1]
                nu = copula_result["param2"].iloc[1]
                corrMatrix = np.array([[1, rho], [rho, 1]])
                u1, u2 = simulation.simu_tstudent(2, len(returns_df), corrMatrix, nu)
                sct_plt(u1, u2, vX1, vX2, family, rho=rho, nu=nu, ax=ax)

            elif family == "gumbel":
                theta = copula_result["param1"].iloc[2]
                u1, u2 = simulation.simu_archimedean("gumbel", 2, len(returns_df), theta=theta)
                sct_plt(u1, u2, vX1, vX2, family, theta=theta, ax=ax)

            elif family == "clayton":
                theta = copula_result["param1"].iloc[3]
                u1, u2 = simulation.simu_archimedean("clayton", 2, len(returns_df), theta=theta)
                sct_plt(u1, u2, vX1, vX2, family, theta=theta, ax=ax)

        plt.tight_layout()
        plt.show()

def main():
    
    #pd.set_option('display.max_columns', None)

    ## SETUP AND BASIC DATA ANALYSIS ##

    # Define the tickers
    tickers = [
        'AAPL', 'MSFT',           # USD: Apple, Microsoft
        'SAP.DE', 'ADS.DE',       # EUR: SAP, Adidas
        '6758.T', '9984.T',       # JPY: Sony, SoftBank Group
        '0700.HK', '0005.HK',     # HKD: Tencent, HSBC Holdings
        '601318.SS', '600519.SS'  # CNY: Ping An Insurance, Kweichow Moutai
    ] 
    
    # Define the period of interest (10 years)
    start_date = '2015-01-01'
    end_date = '2024-12-31'
    
    # Download and clean stock data
    stock_data = process_stock_data(tickers, start_date, end_date)
    
    # Plot stock data
    plot_stock_data(stock_data)
    
    # Compute continuously compounded returns
    returns_df = compute_cont_comp_returns(stock_data)
    
    # Plot stock returns
    plot_stock_returns(returns_df)
    
    # Calculate descriptive statistics 
    #descriptive_statistics(returns_df)
    
    # Plot histograms with normal and student t distribution fit
    plot_histograms_with_normal(returns_df)
    plot_histograms_with_student_t(returns_df)
    
    ## COPULAS ##
    
    pairs = [('AAPL', 'MSFT'), ('SAP.DE', 'ADS.DE'), ('0700.HK', '0005.HK'), ('6758.T', '9984.T')]

    # Plot scatter plots and histograms for all the return pairs for visualization
    plot_scatter_return_pairs(returns_df, pairs)
    
    # Plot empirical CDFs for pairs of return series 
    plot_cdfs_of_pairs(returns_df, pairs)
    
    # Estimate copulas
    copula_results = estimate_all_copulas(returns_df, pairs)
    print(copula_results)
    
    # Compute AIC 
    all_aics, best_copulas = evaluate_all_copulas_by_aic(copula_results)
    print("All Copulas with AIC:")
    print(all_aics[["pair", "copula", "aic"]])
    print("\nBest Copula per Pair (based on AIC):")
    print(best_copulas[["pair", "copula", "aic"]])
    
    # Compute dependence measures
    compute_dependence_measures_for_pairs(returns_df, pairs)
    
    # Tail dependence 
    compute_tail_dependence(copula_results)
    
    # Visualize copulas for each pair and copula family
    simulate_and_plot_copulas(returns_df, pairs)
    
    
### Call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("Execution took:",timedelta(seconds=end_time - start_time))


###########################################################
### PCA 
###########################################################

# Load returns data
returns_df = pd.read_csv('returns_data.csv', index_col=0, parse_dates=True)
returns_clean = returns_df.dropna()
print("Returns data loaded from 'returns_data.csv':")
print(returns_clean.head())

# PCA helper functions
# [2.1.B] Basis Selection: using covariance matrix of demeaned data

def remove_mean(data):
    """Center data by subtracting column means."""
    return data - np.mean(data, axis=0)


def compute_covariance(data):
    """Compute unbiased sample covariance matrix (1/(T-1)) * X^T X for centered data."""
    T = data.shape[0]  # Number of observations
    return (1.0 / (T - 1)) * np.dot(data.T, data)


def eigen_decomposition(cov_matrix):
    """Eigen-decompose covariance matrix, sorted descending."""
    vals, vecs = np.linalg.eig(cov_matrix)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def extract_principal_components(data, n_components):
    """Compute first n principal components for the dataset."""
    centered = remove_mean(data)
    cov = compute_covariance(centered)
    vals, vecs = eigen_decomposition(cov)
    pcs = np.dot(centered, vecs[:, :n_components])
    return pcs, vals, vecs


def plot_variance(eigenvalues):
    """Plot individual and cumulative explained variance ratios with 90% threshold marker."""
    ratios = eigenvalues / np.sum(eigenvalues)
    cum = np.cumsum(ratios)
    pcs = np.arange(1, len(ratios) + 1)
    
    plt.figure(figsize=(8, 5))
    # Bar for individual variance
    plt.bar(pcs, ratios, alpha=0.6, label='Individual')
    # Step plot for cumulative
    plt.step(pcs, cum, where='mid', linestyle='--', label='Cumulative')
    
    # 90% threshold line
    plt.axhline(y=0.9, color='red', linestyle='--', label='90% Threshold')
    
    # Find last PC where cumulative <= 0.9
    idx = np.where(cum <= 0.9)[0]
    if len(idx) > 0:
        k = idx[-1]  # zero based index
        plt.plot(pcs[k], cum[k], 'o', color='red', markersize=8,
                 label=f'PC{k+1} ({cum[k]*100:.1f}%)')
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Compute PCA
# Prepare data array
X = returns_clean.values
# Compute all eigenvalues/vectors
_, all_vals, all_vecs = extract_principal_components(X, X.shape[1])
# Plot variance
plot_variance(all_vals)

# [2.1.A]: Determine number of principal components to include based on the
# 90% cumulative variance threshold is chosen as this retains 7 PC's whereas the Kaiser rule maintains 9, since we want to leverage the dimension reduction
# capability of PCA we decided to follow the <=90% variance explained rule as otherwise we almost just as well could have kept all explanatory variables in.
explained_ratios = all_vals / np.sum(all_vals)

# Eigenvalues of each PC in descending order from largest to smallest, this also shows that by the Kaiser rule we selected every PC except for the 10th
print('\n Eigenvalues of the PCs in descending order:')
print(all_vals)

# The variance explained by each PC in descending order from most variance explained to least
print('\n Variance explained by each PC in descending order:')
print(explained_ratios)

cumulative_ratios = np.cumsum(explained_ratios)
keep = [(cum <= 0.9) for val, cum in zip(all_vals, cumulative_ratios)]
n_factors = sum(keep)
print(f"\n Retaining {n_factors} PC's based on 90% variance criteria.") # PC's is an abreviation for principal components

# [2.1.C]: Compute factor loadings and principal components
# Eigenvectors (columns of eigen_vecs) are the factor loadings
# Project data onto retained eigenvectors to get principal components
eigen_vals, eigen_vecs = all_vals[:n_factors], all_vecs[:, :n_factors]
pcs = np.dot(remove_mean(X), eigen_vecs)
# Convert PCs to dataframe for inspection
pc_df = pd.DataFrame(pcs, index=returns_clean.index,
                     columns=[f'PC{i+1}' for i in range(n_factors)])
print("\n Principal components:")
print(pc_df.head())

print("\n Factor loading of each asset's return in each of the selected PC's:")
loadings = pd.DataFrame(
    eigen_vecs,
    index=returns_clean.columns,
    columns=[f'PC{i+1}' for i in range(n_factors)]
)
print(loadings)

# [2.1.D]: R^2 of Asset returns on retained factors
# Regress each asset's returns on the retained principal components to quantify explained variance
# Higher R^2 implies the factors explain more of that asset's variability
# Compute R^2 for each asset

r2_results = {}
model = LinearRegression()
for i, asset in enumerate(returns_clean.columns):
    y = returns_clean.iloc[:, i].values
    model.fit(pc_df.values, y)
    r2_results[asset] = model.score(pc_df.values, y)
r2_df = pd.DataFrame.from_dict(r2_results, orient='index', columns=['R2'])
print("\n R^2 of each asset explained by the 9 selected PCs:")
print(r2_df)


###########################################################
### Factor Analysis 
###########################################################
# read data file 
returns_df = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/returns_data.csv') 
returns_df_value = returns_df.set_index('Date')  

# Fama French 5 factor data - global 
ff_factor = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Developed_5_Factors_Daily.csv', skiprows=6) 
ff_mom = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Developed_MOM_Factor_Daily.csv', skiprows=6) 

ff_factor['Date'] = pd.to_datetime(ff_factor['Unnamed: 0'].astype(str), format = '%Y%m%d')
ff_mom['Date'] = pd.to_datetime(ff_mom['Unnamed: 0'].astype(str), format = '%Y%m%d') 
ff_factor = ff_factor.set_index('Date')
ff_mom = ff_mom.set_index('Date') 

del ff_factor['Unnamed: 0']
del ff_mom['Unnamed: 0']
returns_df_value.index = pd.to_datetime(returns_df_value.index)

# match the index 
common_index = ff_factor.index.intersection(returns_df_value.index)
ff_factor_new = ff_factor.loc[common_index]
ff_mom_new = ff_mom.loc[common_index]
returns_new = returns_df_value.loc[common_index]

ff = ff_factor_new.merge(ff_mom_new, on=ff_factor_new.index, how='left') 
ff = ff.set_index('key_0')  

# VIX volatility index 
vix_index = pd.read_excel(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/cboe_volatility.xlsx')
vix_index_sp500 = vix_index[['Date', 'CBOE S&P500 Volatility Index - Close']] 
vix_index_nas = vix_index[['Date', 'CBOE NASDAQ Volatility Index - Close']] 

vix_index_sp500.index = pd.to_datetime(vix_index_sp500['Date'])
vix_index_nas.index = pd.to_datetime(vix_index_nas['Date'])

returns_df_value.index = pd.to_datetime(returns_df_value.index).normalize()
vix_index_sp500.index = pd.to_datetime(vix_index_sp500.index).normalize()
vix_index_nas.index = pd.to_datetime(vix_index_nas.index).normalize()

vix_index_sp500 = vix_index_sp500.reindex(returns_df_value.index)
vix_index_nas = vix_index_nas.reindex(returns_df_value.index)

vix_index_sp500 = vix_index_sp500.interpolate(method='linear')
vix_index_nas = vix_index_nas.interpolate(method='linear') 

vix = vix_index_sp500.merge(vix_index_nas, on=vix_index_nas.index, how='left') 
vix = vix[['Date_x', 'CBOE S&P500 Volatility Index - Close', 'CBOE NASDAQ Volatility Index - Close']] 

vix = vix.rename(columns={'Date_x':'Date'})
vix.index = pd.to_datetime(vix['Date'])
del vix['Date'] 

vix = vix.rename(columns={'CBOE S&P500 Volatility Index - Close':'S&P500', 'CBOE NASDAQ Volatility Index - Close':'NADQ'})
all_factor = ff.merge(vix, on=ff.index, how='left') 
all_factor = all_factor.rename(columns={'key_0':'Date'}) 
all_factor = all_factor.set_index('Date') 

# macro factor - global (overall)
factors = all_factor[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML', 'S&P500']] # delete Dadq 
excess_returns = returns_df_value.subtract(factors['RF'], axis=0) 
factor_explain = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML', 'S&P500']] 

results = {} 
for asset in excess_returns.columns: 
    y = excess_returns[asset] 
    X = sm.add_constant(factor_explain)
    model = sm.OLS(y, X, missing='drop').fit() 
    results[asset] = model 
    
loadings = pd.DataFrame({asset: res.params for asset, res in results.items()}).T
print(loadings) 

residuals = pd.DataFrame({asset: res.resid for asset, res in results.items()})
print(residuals) 

# residual correlation matrix 
residual_corr = residuals.corr()
print(residual_corr)  

r_squared_df = pd.Series({asset: res.rsquared for asset, res in results.items()})
print(r_squared_df)
pvalues_df = pd.DataFrame({asset: res.pvalues for asset, res in results.items()}).T
print(pvalues_df)

# adjusted R^2 for 7 factors 
n = 2466 
k = 7
r_squared_adj = 1 - (1 - r_squared_df)*(n - 1) / (n - k - 1)
print(r_squared_adj)  

# split the regressions by regions & countries 

# USD: Apple, Microsoft 'AAPL', 'MSFT',           
# EUR: SAP, Adidas 'SAP.DE', 'ADS.DE',       
# JPY: Sony, SoftBank Group '6758.T', '9984.T',       
# HKD: Tencent, HSBC Holdings '0700.HK', '0005.HK',     
# CNY: Ping An Insurance, Kweichow Moutai '601318.SS', '600519.SS'   

## read regional fama french 5 factor data 
## US part data 
ff_factor_us = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/North_America_5_Factors_Daily.csv', skiprows=6) 
ff_mom_us = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/North_America_MOM_Factor_Daily.csv', skiprows=6)  
ff_factor_us['Date'] = pd.to_datetime(ff_factor_us['Unnamed: 0'].astype(str), format = '%Y%m%d')
ff_mom_us['Date'] = pd.to_datetime(ff_mom_us['Unnamed: 0'].astype(str), format = '%Y%m%d') 
ff_factor_us = ff_factor_us.set_index('Date')
ff_mom_us = ff_mom_us.set_index('Date') 
del ff_factor_us['Unnamed: 0']
del ff_mom_us['Unnamed: 0']  

common_index = ff_factor_us.index.intersection(returns_df_value.index)
ff_factor_us = ff_factor_us.loc[common_index]
ff_mom_us = ff_mom_us.loc[common_index] 
ff_us = ff_factor_us.merge(ff_mom_us, on=ff_factor_us.index, how='left') 
ff_us = ff_us.set_index('key_0')  

# EU part data 
ff_factor_eu = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Europe_5_Factors_Daily.csv', skiprows=6) 
ff_mom_eu = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Europe_MOM_Factor_Daily.csv', skiprows=6)  

ff_factor_eu['Date'] = pd.to_datetime(ff_factor_eu['Unnamed: 0'].astype(str), format = '%Y%m%d')
ff_mom_eu['Date'] = pd.to_datetime(ff_mom_eu['Unnamed: 0'].astype(str), format = '%Y%m%d') 
ff_factor_eu = ff_factor_eu.set_index('Date')
ff_mom_eu = ff_mom_eu.set_index('Date') 
del ff_factor_eu['Unnamed: 0']
del ff_mom_eu['Unnamed: 0']  

common_index = ff_factor_eu.index.intersection(returns_df_value.index)
ff_factor_eu = ff_factor_eu.loc[common_index]
ff_mom_eu = ff_mom_eu.loc[common_index]

ff_eu = ff_factor_eu.merge(ff_mom_eu, on=ff_factor_eu.index, how='left') 
ff_eu = ff_eu.set_index('key_0') 

# Japan part data 
ff_factor_jp = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Japan_5_Factors_Daily.csv', skiprows=6) 
ff_mom_jp = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Japan_MOM_Factor_Daily.csv', skiprows=6)  

ff_factor_jp['Date'] = pd.to_datetime(ff_factor_jp['Unnamed: 0'].astype(str), format = '%Y%m%d')
ff_mom_jp['Date'] = pd.to_datetime(ff_mom_jp['Unnamed: 0'].astype(str), format = '%Y%m%d') 
ff_factor_jp = ff_factor_jp.set_index('Date')
ff_mom_jp = ff_mom_jp.set_index('Date') 
del ff_factor_jp['Unnamed: 0']
del ff_mom_jp['Unnamed: 0']  

common_index = ff_factor_jp.index.intersection(returns_df_value.index)
ff_factor_jp = ff_factor_jp.loc[common_index]
ff_mom_jp = ff_mom_jp.loc[common_index]

ff_jp = ff_factor_jp.merge(ff_mom_jp, on=ff_factor_jp.index, how='left') 
ff_jp = ff_jp.set_index('key_0') 

# Asia_Pacific_ex_Japan part 
ff_factor_asia = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Asia_Pacific_ex_Japan_5_Factors_Daily.csv', skiprows=6) 
ff_mom_asia = pd.read_csv(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/Asia_Pacific_ex_Japan_MOM_Factor_Daily.csv', skiprows=6)  

ff_factor_asia['Date'] = pd.to_datetime(ff_factor_asia['Unnamed: 0'].astype(str), format = '%Y%m%d')
ff_mom_asia['Date'] = pd.to_datetime(ff_mom_asia['Unnamed: 0'].astype(str), format = '%Y%m%d') 
ff_factor_asia = ff_factor_asia.set_index('Date')
ff_mom_asia = ff_mom_asia.set_index('Date') 
del ff_factor_asia['Unnamed: 0']
del ff_mom_asia['Unnamed: 0']  

common_index = ff_factor_asia.index.intersection(returns_df_value.index)
ff_factor_asia = ff_factor_asia.loc[common_index]
ff_mom_asia = ff_mom_asia.loc[common_index]

ff_asia = ff_factor_asia.merge(ff_mom_asia, on=ff_factor_asia.index, how='left') 
ff_asia = ff_asia.set_index('key_0') 

# import currency rate data 
currency_df = pd.read_excel(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/data/currency exchange rate.xlsx') 
currency_df['date'] = pd.to_datetime(currency_df['date'])
currency_df = currency_df.set_index('date') 

currency_df.index = pd.to_datetime(currency_df.index).normalize()
returns_df_value.index = pd.to_datetime(returns_df_value.index).normalize()

common_index = currency_df.index.intersection(returns_df_value.index)
currency_df = currency_df.loc[common_index] 

# plot for currency rate change time series 
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()  

currencies = ['Yuan/US', 'HK/US', 'JP/US', 'Euro/US']
colors = ['red', 'blue', 'pink', 'royalblue']
styles = ['-', '-', '-', '-']
titles = ['Yuan vs USD', 'HKD vs USD', 'JPY vs USD', 'Euro vs USD']

for i in range(4):
    axs[i].plot(currency_df.index, currency_df[currencies[i]], 
                linestyle=styles[i], color=colors[i], linewidth=1)
    axs[i].set_title(titles[i], fontsize=12)
    axs[i].grid(True, linestyle='--', alpha=0.5)

fig.supxlabel('Date')
fig.supylabel('Exchange Rate')

plt.tight_layout()
plt.savefig(r'/Users/nicolez/Desktop/VU/QFRM/Assignment 3/exchangerate.png') 
plt.show() 

# regressions for US part 
## for US part, use S&P as volatility proxy 
## factor analysis by regions - US 
ff_us_df = ff_us.merge(vix, on=vix.index, how='left') 
ff_us_df = ff_us_df.rename(columns={'key_0':'Date'}) 
returns_df_value_us = returns_df_value[['AAPL', 'MSFT']] 

ff_us_df = ff_us_df.set_index('Date')
factors = ff_us_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML','S&P500']] 

common_index = returns_df_value_us.index.intersection(factors.index)

returns_df_value_us = returns_df_value_us.loc[common_index]
factors = factors.loc[common_index] 

excess_returns = returns_df_value_us.subtract(factors['RF'], axis=0) 
factor_explain = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML','S&P500']]

results = {} 
for asset in excess_returns.columns: 
    y = excess_returns[asset] 
    X = sm.add_constant(factor_explain)
    model = sm.OLS(y, X, missing='drop').fit() 
    results[asset] = model  
    
loadings = pd.DataFrame({asset: res.params for asset, res in results.items()}).T
print(loadings)  

r_squared_df = pd.Series({asset: res.rsquared for asset, res in results.items()})
print(r_squared_df)

pvalues_df = pd.DataFrame({asset: res.pvalues for asset, res in results.items()}).T
print(pvalues_df)


# regressions for EU part 
ff_eu_df = ff_eu.merge(currency_df['Euro/US'], on=currency_df.index, how='left') 
ff_eu_df = ff_eu_df.rename(columns={'key_0':'Date'}) 
ff_eu_df = ff_eu_df.set_index('Date')

returns_df_value_eu = returns_df_value[['SAP.DE', 'ADS.DE']] 

factors = ff_eu_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML', 'Euro/US']] 
excess_returns = returns_df_value_eu.subtract(factors['RF'], axis=0) 
factor_explain = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML', 'Euro/US']]

results = {} 
for asset in excess_returns.columns: 
    y = excess_returns[asset] 
    X = sm.add_constant(factor_explain)
    model = sm.OLS(y, X, missing='drop').fit() 
    results[asset] = model  
    
loadings = pd.DataFrame({asset: res.params for asset, res in results.items()}).T
print(loadings)  

r_squared_df = pd.Series({asset: res.rsquared for asset, res in results.items()})
print(r_squared_df)

pvalues_df = pd.DataFrame({asset: res.pvalues for asset, res in results.items()}).T
print(pvalues_df)

# for JP part 
ff_jp_df = ff_jp.merge(currency_df['JP/US'], on=currency_df.index, how='left') 
ff_jp_df = ff_jp_df.rename(columns={'key_0':'Date'}) 
ff_jp_df = ff_jp_df.set_index('Date')

returns_df_value_jp = returns_df_value[['6758.T', '9984.T']] 

factors = ff_jp_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML', 'JP/US']] 
excess_returns = returns_df_value_jp.subtract(factors['RF'], axis=0) 
factor_explain = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML', 'JP/US']]

results = {} 
for asset in excess_returns.columns: 
    y = excess_returns[asset] 
    X = sm.add_constant(factor_explain)
    model = sm.OLS(y, X, missing='drop').fit() 
    results[asset] = model  
    
loadings = pd.DataFrame({asset: res.params for asset, res in results.items()}).T
print(loadings)  

r_squared_df = pd.Series({asset: res.rsquared for asset, res in results.items()})
print(r_squared_df)

pvalues_df = pd.DataFrame({asset: res.pvalues for asset, res in results.items()}).T
print(pvalues_df)

# for HK & CN currency - HK part 
ff_asia_df = ff_asia.merge(currency_df[['HK/US','Yuan/US']], on=currency_df.index, how='left') 

ff_asia_df = ff_asia_df.rename(columns={'key_0':'Date'}) 
ff_asia_df = ff_asia_df.set_index('Date')

returns_df_value_hk = returns_df_value[['0700.HK', '0005.HK']] 

factors = ff_asia_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML', 'HK/US']] 
excess_returns = returns_df_value_hk.subtract(factors['RF'], axis=0) 
factor_explain = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML', 'HK/US']]

results = {} 
for asset in excess_returns.columns: 
    y = excess_returns[asset] 
    X = sm.add_constant(factor_explain)
    model = sm.OLS(y, X, missing='drop').fit() 
    results[asset] = model  

loadings = pd.DataFrame({asset: res.params for asset, res in results.items()}).T
print(loadings)  

r_squared_df = pd.Series({asset: res.rsquared for asset, res in results.items()})
print(r_squared_df)

pvalues_df = pd.DataFrame({asset: res.pvalues for asset, res in results.items()}).T
print(pvalues_df)

# for HK & CN currency - CN part 
returns_df_value_cn = returns_df_value[['601318.SS', '600519.SS']] 

factors = ff_asia_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML', 'Yuan/US']] 
excess_returns = returns_df_value_cn.subtract(factors['RF'], axis=0) 
factor_explain = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML', 'Yuan/US']]

results = {} 
for asset in excess_returns.columns: 
    y = excess_returns[asset] 
    X = sm.add_constant(factor_explain)
    model = sm.OLS(y, X, missing='drop').fit() 
    results[asset] = model  
    
loadings = pd.DataFrame({asset: res.params for asset, res in results.items()}).T
print(loadings)  

r_squared_df = pd.Series({asset: res.rsquared for asset, res in results.items()})
print(r_squared_df)

pvalues_df = pd.DataFrame({asset: res.pvalues for asset, res in results.items()}).T
print(pvalues_df) 

###########################################################
# Alternative approach: use PCA as the inputs 
# 1. PCA factor 
X = returns_df_value - returns_df_value.mean()

K = 7 # use results from last q 
pca = PCA(n_components=K)
factors_pca = pd.DataFrame(pca.fit_transform(X), index=returns_df_value.index)

# 2. rotate factor 
from factor_analyzer.rotator import Rotator

rotator = Rotator(method='varimax')
factors_rotated = pd.DataFrame(rotator.fit_transform(factors_pca.values), index=returns_df_value.index)

# 3. regression 
results = {}
for asset in returns_df_value.columns:
    y = returns_df_value[asset]
    X = sm.add_constant(factors_rotated)
    model = sm.OLS(y, X, missing='drop').fit()
    results[asset] = model 
    
loadings = pd.DataFrame({asset: res.params for asset, res in results.items()}).T
print(loadings)  

r_squared_df = pd.Series({asset: res.rsquared for asset, res in results.items()})
print(r_squared_df)

pvalues_df = pd.DataFrame({asset: res.pvalues for asset, res in results.items()}).T
print(pvalues_df) 

residuals = pd.DataFrame({asset: res.resid for asset, res in results.items()})
print(residuals)  

residuals = residuals.dropna()
residuals_corr = residuals.corr() 
residuals_corr  

# adjusted R^2 for 7 factors 
n = 2466 
k = 5 
r_squared_adj = 1 - (1 - r_squared_df)*(n - 1) / (n - k - 1)
print(r_squared_adj)  

## correlation matrix for all involved factors 
tem = factors_rotated.merge(all_factor, on=all_factor.index, how='left') 
tem = tem.set_index('key_0') 

tem.corr()

###########################################################
### EVT 
########################################################### 
# Below we define the functions and helper functions for the EVT part of the assignment
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
    df = df.sort_values("Date")
    return df

def qq_plots(asset_columns, df):
    for asset in asset_columns:
        returns = df[asset].dropna()
        plt.figure()
        df_fit, loc, scale = t.fit(returns)
        stats.probplot(returns, dist="t", sparams=(df_fit,loc, scale), plot=plt)
        plt.title(f'QQ Plot: {asset} vs Student-t (df={df_fit:.2f})')
        plt.show()

def mean_excess_plot(losses, quantile_min=0.8, quantile_max=0.99, steps=50):
    thresholds = np.linspace(losses.quantile(quantile_min), losses.quantile(quantile_max), steps)
    mean_excess = []

    for u in thresholds:
        excess = losses[losses > u] - u
        mean_excess.append(np.mean(excess) if len(excess) > 0 else np.nan)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, mean_excess, marker='o', linestyle='-')
    plt.xlabel("Threshold u")
    plt.ylabel("Mean Excess e(u)")
    plt.title("Mean Excess Plot")
    plt.grid(True)
    plt.show()
    

def plot_gpd_fit(exceedances, xi, loc, beta):
    # Generate smooth x‐grid over the range of exceedances
    x = np.linspace(min(exceedances), max(exceedances), 200)
    pdf = genpareto.pdf(x, c=xi, loc=loc, scale=beta)

    # Set up figure
    plt.figure(figsize=(8, 5))
    
    # Histogram of empirical exceedances
    plt.hist(
        exceedances,
        bins=30,
        density=True,
        alpha=0.6,
        color='gray',
        edgecolor='black',
        label='Empirical excesses'
    )
    
    # Plot the fitted GPD density
    plt.plot(
        x,
        pdf,
        linewidth=2,
        color='navy',
        label=f'GPD fit (ξ={xi:.3f}, β={beta:.3f})'
    )
    
    # Labels, title, legend
    plt.xlabel(r'Exceedance over threshold $u$', fontsize=12)
    plt.ylabel('Probability density', fontsize=12)
    plt.title('Fit of Generalized Pareto Distribution to Excesses', fontsize=14, pad=12)
    plt.legend(frameon=True, fontsize=11)
    
    # Grid, ticks, layout
    plt.grid(linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.tight_layout()
    
    plt.show()

def VaR_ES(n, Nu, Fu, alpha, u, xi, beta):
    VaR = u + (beta / xi) * (((n/Nu)*(1-alpha)) ** (-xi) - 1)
    ES = (VaR + beta - xi*u) / (1 - xi)
    return VaR, ES

def hist_sim_VaR_ES(losses, alpha):
    losses_sorted = np.sort(losses)
    n = len(losses_sorted)
    k = int(np.ceil(n * alpha)) - 1  
    
    VaR = losses_sorted[k]
    ES = np.mean(losses_sorted[k:])
    
    return VaR, ES

def garch_var_es(returns, alpha):
    alpha = 1-alpha
    model = arch_model(returns, vol='Garch', dist='t')
    res = model.fit(disp='off')
    nu = res.params['nu']
    mu = res.params['mu']
    fcast = res.forecast(horizon=1)
    cond_vol =np.sqrt(fcast.variance.values[-1, 0])
    q = t.ppf(alpha, df=nu)
    var= mu + cond_vol * q * np.sqrt((nu - 2) / nu)
    es= mu - cond_vol * (t.pdf(q, df=nu) / alpha) * (nu + q**2) / (nu - 1) * np.sqrt((nu - 2) / nu)
    
    return -var, -es

# df = load_data(r"returns_data.csv")
df = pd.read_csv('returns_data.csv', index_col=0, parse_dates=True)
asset_columns = df.columns[df.columns != "Date"]
# qq_plots(asset_columns, df)

returns = df['MSFT'].dropna()
losses = -returns  # Losses = negative returns (positive values represent losses)
# 1. EVT Approach
print("EVT Method")
mean_excess_plot(losses, quantile_min=0.5, quantile_max=0.99, steps=50)

u = 2.6
exceedances = losses[losses > u] - u
xi, loc, beta = genpareto.fit(exceedances, floc=0)

print(f"GPD Parameters - xi: {xi:.4f}, beta: {beta:.4f}")
plot_gpd_fit(exceedances, xi, loc, beta)
n = len(losses)
Nu = len(exceedances)
Fu = Nu / n

alphas = [0.95, 0.99]
for alpha in alphas:
    VaR, ES = VaR_ES(n, Nu, Fu, alpha, u, xi, beta)
    print(f"Alpha={alpha:.2f}: VaR={VaR:.4f}, ES={ES:.4f}")
# 2. Historical Simulation
print("\nHistorical Simulation")
for alpha in alphas:
    VaR, ES = hist_sim_VaR_ES(losses, alpha)
    print(f"Alpha={alpha:.2f}: VaR={VaR:.4f}, ES={ES:.4f}")
# 3. GARCH with Student-t
print("\nGARCH with Student-t")
for alpha in alphas:
    VaR, ES = garch_var_es(returns, alpha)
    print(f"Alpha={alpha:.2f}: VaR={VaR:.4f}, ES={ES:.4f}")
        
        


