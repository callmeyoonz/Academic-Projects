import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Constants
C_STAR = 3.5134
TRADING_START = "09:30:00"
TRADING_END = "16:00:00"
SECONDS_IN_DAY = 23401  # 6.5 hours * 3600 + 1

# Fill missing seconds with backward and forward-filled prices
def fill_missing_seconds(data):
    # Forward-fill and backward-fill prices
    data.loc[:, 'PRICE'] = data['PRICE'].ffill().bfill()
    return data

# Compute log returns (scaled by 100)
def compute_log_returns(data):
    """Compute log returns."""
    data = data.copy()  # Ensure we work on a copy to avoid SettingWithCopyWarning
    data['log_price'] = np.log(data['PRICE'])
    data['log_ret'] = 100 * data['log_price'].diff()
    return data.dropna(subset=['log_ret'])
    # data['log_ret'] = data['log_ret'].fillna(method='ffill')
    # return data

# Compute Realized Volatility (RV)
def compute_rv(data):
    """Compute realized volatility (RV)."""
    return (data['log_ret'] ** 2).sum()

# Compute RV_sparse using 20-minute returns
def compute_rv_sparse(data):
    """Compute RV_sparse using 20-minute returns."""
    data = data.set_index('DATETIME')
    interval_seconds = 20 * 60
    rv_values = []
    for offset in range(interval_seconds):
        sampled = data.iloc[offset::interval_seconds]
        if sampled.empty:
            continue
        rv_values.append((sampled['log_ret'] ** 2).sum())
    return np.mean(rv_values) if rv_values else np.nan

# Compute RV_dense using q trades
def compute_rv_dense(data, q):
    """Compute RV_dense using steps of q trades."""
    returns = data['log_ret'].values
    rv_values = []
    for start in range(q):
        sampled_returns = returns[start::q]
        if sampled_returns.size == 0:
            continue
        rv_values.append((sampled_returns ** 2).sum())
    return rv_values

# Estimate omega_squared using RV_dense
def estimate_omega_squared(rv_dense_values, n_values):
    """Estimate omega^2 using RV_dense."""
    return np.mean([rv / (2 * n) for rv, n in zip(rv_dense_values, n_values) if n > 0]) if rv_dense_values else np.nan

# Compute bandwidth H
def compute_bandwidth(c_star, omega_squared, iv, n):
    """Compute the optimal bandwidth H."""
    if iv <= 0 or np.isnan(omega_squared):
        return np.nan
    xi_squared = omega_squared / iv
    h = c_star * ((xi_squared ** (2 / 5)) * (n ** (3 / 5)))
    return int(np.ceil(h)) if not np.isnan(h) else np.nan

# Compute the realized kernel using the Parzen kernel
def realized_kernel(data, h):
    """Compute the realized kernel using the Parzen kernel."""
    if np.isnan(h):
        print("Bandwidth (h) is NaN. Check the input data or computations.")
        return np.nan

    h = int(h)  # Convert h to an integer
    r = data['log_ret'].values
    kernel_value = 0

    def parzen_kernel(x):
        ax = abs(x)
        if ax <= 0.5:
            return 1 - 6 * (ax ** 2) + 6 * (ax ** 3)
        elif ax <= 1:
            return 2 * (1 - ax) ** 3
        return 0

    for lag in range(-h, h + 1):
        weight = parzen_kernel(lag / (h + 1))
        if lag == 0:
            kernel_value += weight * np.sum(r ** 2)
        else:
            idx = abs(lag)
            kernel_value += 2 * weight * np.sum(r[idx:] * r[:-idx])

    return kernel_value

# Compute Realized Variance (RV_5min) for comparison
def compute_realized_variance(data):
    """Compute realized variance using 5-minute returns."""
    data = data.set_index('DATETIME')
    sampled = data.resample('5min').last()  # Use 'min' instead of 'T'
    sampled['log_price'] = np.log(sampled['PRICE'])
    sampled['log_ret_5min'] = 100 * sampled['log_price'].diff()
    return (sampled['log_ret_5min'] ** 2).sum()

# Verify Realized Kernel
def verify_realized_kernel(results, df):
    """Verify the plausibility of realized kernel."""
    # Compute daily realized variance
    df['date'] = df['DATETIME'].dt.date
    rv_variance = []
    for date in results['date']:
        group = df[df['DATETIME'].dt.date == date]
        if group.empty:
            rv_variance.append(np.nan)
            continue
        rv_variance.append(compute_realized_variance(group))

    # Ensure length matches before assignment
    if len(rv_variance) != len(results):
        raise ValueError(
            f"Mismatch in lengths: results ({len(results)}) vs. rv_variance ({len(rv_variance)})"
        )

    # Add realized variance to results for comparison
    results['realized_variance'] = rv_variance

    # Compare means
    mean_rk = results['realized_kernel'].mean()
    mean_rv = results['realized_variance'].mean()
    print(f"Mean Realized Kernel: {mean_rk}")
    print(f"Mean Realized Variance: {mean_rv}")
    print(f"Ratio (RK / RV): {mean_rk / mean_rv}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(results['date'], results['realized_kernel'], label='Realized Kernel (RK)', alpha=0.7)
    plt.plot(results['date'], results['realized_variance'], label='Realized Variance (RV_5min)', alpha=0.7)
    plt.title('Comparison of Realized Kernel and Realized Variance')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

    # Inspect correlation with daily return volatility
    results['daily_return_volatility'] = results['realized_kernel'] ** 0.5
    print("Correlation between daily returns and realized kernel:")
    print(results[['realized_kernel', 'daily_return_volatility']].corr())

# Compute daily metrics
def compute_daily_metrics(df):
    """Compute daily metrics for realized volatility and kernel."""
    df['date'] = df['DATETIME'].dt.date
    results = []

    for date in df['date'].unique():
        #print(f"Processing date: {date}")  # Debug: Print current date
        group = df[df['DATETIME'].dt.date == date].copy()
        #print(f"Number of rows for {date}: {len(group)}")  # Debug: Print group size

        if group.empty:
            #print(f"No data for {date}. Skipping.")
            continue

        # Check if any critical column is missing or invalid
        if group['PRICE'].isna().all():
            #print(f"PRICE is entirely NaN for {date}. Skipping.")
            continue

        group = fill_missing_seconds(group)
        group = compute_log_returns(group)

        if group.empty:
            #print(f"Group is empty after computing log returns for {date}. Skipping.")
            continue

        rv = compute_rv(group)
        rv_sparse = compute_rv_sparse(group)

        n = len(group)
        #print(f"Computed n for {date}: {n}")  # Debug: Log n value
        if n == 0:
            #print(f"No valid data for {date} after processing. Skipping.")
            continue

        q = max(1, round(n / 195))  # Ensure q is at least 1
        rv_dense_values = compute_rv_dense(group, q)
        n_values = [len(group['log_ret'].iloc[i::q]) for i in range(q)]

        omega_squared = estimate_omega_squared(rv_dense_values, n_values)

        if np.isnan(rv_sparse):
            print(f"RV_sparse is NaN for date {date}.")
        if np.isnan(omega_squared):
            print(f"Omega_squared is NaN for date {date}.")

        h = compute_bandwidth(C_STAR, omega_squared, rv_sparse, n)

        if np.isnan(h):
            print(f"Bandwidth H is NaN for date {date}. Omega_squared: {omega_squared}, RV_sparse: {rv_sparse}, n: {n}")

        rk_value = realized_kernel(group, h)

        results.append({'date': date, 'realized_volatility': rv, 'realized_kernel': rk_value, 'H': h})

    return pd.DataFrame(results)

# Load data
file_path = "../data/new/1s_out.csv"
df = pd.read_csv(file_path)
df['DATETIME'] = pd.to_datetime(df['DATETIME'], errors='coerce')

# Fill missing seconds and compute metrics
results = compute_daily_metrics(df)

#Save results to CSV
output_path = r"C:\Users\\1000r\\Downloads\\1s_data_in.csv"
results.to_csv(output_path, index=False)

#Verify realized kernel
verify_realized_kernel(results, df)

#Plot results
plt.figure(figsize=(12, 6))
plt.plot(results['date'], results['realized_volatility'], label='Realized Volatility (RV)', alpha=0.7)
plt.plot(results['date'], results['realized_kernel'], label='Realized Kernel (RK)', alpha=0.7)
plt.title('Daily Realized Volatility and Realized Kernel')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

print(results.head())

results.to_csv('RK_out.csv', index=False)

plt.figure(figsize=(12, 4))
plt.plot(results['date'], results['realized_volatility'], label='Realized Volatility (RV)', alpha=1, linewidth=0.8)
plt.plot(results['date'], results['realized_kernel'], label='Realized Kernel (RK)', alpha=0.6, linewidth=0.8)
plt.title('Daily Realized Volatility and Realized Kernel')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()


results.to_csv('RK_out.csv', index=False)

plt.figure(figsize=(12,6)) 
plt.plot(results['date'], results['realized_kernel'], linestyle='-')
plt.title('realized kernel by day', fontsize=14) 
plt.xlabel('date', fontsize=12) 
plt.ylabel('realized kernel', fontsize=12) 
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()

# Verify realized kernel
#verify_realized_kernel(results, df)

# Plot results
plt.figure(figsize=(12, 4))
plt.plot(results['date'], results['realized_volatility'], label='Realized Volatility (RV)', alpha=1, linewidth=0.8)
plt.plot(results['date'], results['realized_kernel'], label='Realized Kernel (RK)', alpha=0.6, linewidth=0.8)
plt.title('Daily Realized Volatility and Realized Kernel')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()


