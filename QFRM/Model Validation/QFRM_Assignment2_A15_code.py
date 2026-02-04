import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import scipy.stats as sts
from arch import arch_model 
from scipy import stats
from scipy.stats import t
from scipy.stats import norm 
from scipy.stats import binom
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import ks_2samp

import os 

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.chdir('/Users/nicolez/Library/Mobile Documents/com~apple~CloudDocs/VU/P5/Quantitative Financial Risk Management/Assignment/A2')
file_path = '/Users/nicolez/Library/Mobile Documents/com~apple~CloudDocs/VU/P5/Quantitative Financial Risk Management/Assignment/A2' 

raw_data = pd.read_csv(r'./a14_135120_9019402_raw_data.csv') 
consolidate_data = pd.read_csv(r'./a14_135120_9019403_portfolio_returns.csv')  

portfolio_data = consolidate_data[['Date','^GSPC', '^IBEX', '^N225', 'LOAN_RETURN']].copy()
portfolio_data[['^GSPC', '^IBEX', '^N225', 'LOAN_RETURN']] *= 100 

portfolio_data = portfolio_data.set_index('Date') 
portfolio_weights = np.array([0.25, 0.20, 0.25, 0.30]) #indicate in report 
portfolio_returns = portfolio_data.dot(portfolio_weights)  

## define the crisis periods as the report did (graphically) 
# economy criris: 2015/06 - 2016/06 
# 2020 pandemic: 2020/02 - 2020/04 
# energy crisis + Russia - Ukraine conflict: 2021-2023  

########## reimplementation ########## 
##### GARCH-CCC ##### 
## rolling window 
def vc_multi_normal_VaR_ES(vW, vR, mS2, vAlpha):
    # Portfolio mean return
    port_mean = np.dot(vW.T, vR)
    
    # Portfolio standard deviation
    port_std = np.sqrt(np.dot(vW.T, np.dot(mS2, vW)))
    
    # VaR calculation
    dVaR0 = sts.norm.ppf(vAlpha)
    dVaR = -port_mean + port_std * dVaR0  
    
    # ES calculation 
    dES0 = (sts.norm.pdf(dVaR0) / (1 - vAlpha))
    dES = -port_mean + port_std * dES0
    
    return dVaR, dES

def rolling_GARCH_CCC_var_es(portfolio_data, weights, alphas, window):
    weights = weights.reshape(-1, 1)
    k_assets = portfolio_data.shape[1] - 1  # exclude cash
    df_out = pd.DataFrame(index=portfolio_data.index)

    for alpha in alphas:
        var_series = []
        es_series = []
        portfolio_volatility = [] 
        standardized_residual = [] # for standardized residual 

        for i in range(window, len(portfolio_data)):
            window_data = portfolio_data.iloc[i-window:i]
            asset_returns = window_data.iloc[:, :-1]
            cash_returns = window_data.iloc[:, -1]
            k = asset_returns.shape[1]

            conditional_vols = pd.DataFrame(index=asset_returns.index)
            for asset in asset_returns.columns:
                model = arch_model(asset_returns[asset], vol='Garch', p=1, q=1, dist='normal') 
                res = model.fit(disp='off')
                conditional_vols[asset] = res.conditional_volatility

            last_vols = conditional_vols.iloc[-1].values
            Delta = np.diag(last_vols)
            Y_t = asset_returns / conditional_vols
            P = Y_t.corr().values
            portfolio_z = Y_t.iloc[-1].values @ weights[:-1] 
            standardized_residual.append(portfolio_z.item()) 

            Sigma = Delta @ P @ Delta

            extended_Sigma = np.zeros((k+1, k+1))
            extended_Sigma[:k, :k] = Sigma
            extended_Sigma[-1, -1] = cash_returns.var() 
            
            port_std = np.sqrt(np.dot(weights.T, np.dot(extended_Sigma, weights))) 
            portfolio_volatility.append(port_std.item()) 

            mu_vec = window_data.mean().values.reshape(-1, 1)

            dVaR, dES = vc_multi_normal_VaR_ES(weights, mu_vec, extended_Sigma, alpha)

            var_series.append(dVaR.item())
            es_series.append(dES.item())

        df_out[f'VaR_{int(alpha*1000)}'] = [np.nan] * window + var_series
        df_out[f'ES_{int(alpha*1000)}'] = [np.nan] * window + es_series
        df_out['Portfolio_Volatility'] = [np.nan] * window + portfolio_volatility
        df_out['standardized_residual'] = [np.nan] * window + standardized_residual

    return df_out

rolling_df = rolling_GARCH_CCC_var_es(portfolio_data, portfolio_weights, alphas=[0.975, 0.99], window=261) # window defined in report 

df_m4 = pd.DataFrame({'Return': portfolio_returns, 
                      'Loss': -portfolio_returns}).join(rolling_df)   


df_m4.index = pd.to_datetime(df_m4.index) 
df_m4['year'] = df_m4.index.year 

## backtest ES and VAR 
def backtest_var_es_by_year(df, alphas, loss_col='Loss', date_col='Date'):
    df = df.copy()
    df['year'] = df['year']
    result_dict = {}

    for alpha in alphas: 
        var_col = f'VaR_{int(alpha * 1000)}'
        es_col = f'ES_{int(alpha * 1000)}'
        viol_col = f'Violation_{int(alpha * 1000)}'

        df[viol_col] = (df[loss_col] > df[var_col]).fillna(False).astype(bool)

        actual_violation = df.groupby('year')[viol_col].sum()
        total_counts = df.groupby('year').size()
        expected_violation = total_counts * (1 - alpha)

        results = pd.DataFrame({
            'Vio#': actual_violation,
            'Expected': expected_violation,
            'Vio%': actual_violation / total_counts,
            'Exceed%': actual_violation / total_counts * 100,
            'Expected%': (1 - alpha) * 100
        })

        viol_mask = df[viol_col]
        actual_ES = df.loc[viol_mask].groupby('year')[loss_col].mean()
        expected_ES = df.loc[viol_mask].groupby('year')[es_col].mean()

        results['Actual_ES'] = actual_ES
        results['Expected_ES'] = expected_ES

        result_dict[f'VaR_{int(alpha * 1000)}'] = results

    return result_dict

result_dict = backtest_var_es_by_year(df_m4, alphas=[0.975, 0.99])

for label, res in result_dict.items():
    print(f"\nBacktest Results for {label}:")
    print(res.round(4)) 

## VaR binomial test 
def binomial_var_backtest(df, alpha, level=0.05, year_col='year'):
    col_viol = f'Violation_{int(alpha * 1000)}'

    grouped = df.groupby('year')

    results = []
    for yr, group in grouped:
        T = len(group)
        I_hat = group[col_viol].sum()
        expected = T * (1 - alpha)

        cL = binom.ppf(level / 2, T, 1 - alpha)
        cU = binom.ppf(1 - level / 2, T, 1 - alpha)

        reject = not (cL <= I_hat <= cU)

        results.append({
            'Year': yr,
            'T': T,
            'I_hat': I_hat,
            'Expected': expected,
            'cL': cL,
            'cU': cU,
            'Reject H0': reject
        })

    return pd.DataFrame(results)

df_m4['Violation_975'] = (df_m4['Loss'] > df_m4['VaR_975']).astype(int)
df_m4['Violation_990'] = (df_m4['Loss'] > df_m4['VaR_990']).astype(int) 

binomial_test_m4_975 = binomial_var_backtest(df_m4, alpha = 0.975)
print(binomial_test_m4_975)

binomial_test_m4_99 = binomial_var_backtest(df_m4, alpha = 0.99)
print(binomial_test_m4_99)  # reject H0: violation no abnormal, true = discrepancies  

## violation residual ES t-test 
def test_es_residuals(df, alphas, loss_col='Loss'):
    srLoss = df[loss_col]
    dfTES = pd.DataFrame()
    dfK = pd.DataFrame(index=df.index)
    dMu = srLoss.mean()
    
    for alpha in alphas:
        col_viol = f'Violation_{int(alpha * 1000)}'
        col_es = f'ES_{int(alpha * 1000)}'
        col_label = f'ES_{int(alpha * 1000)}'
        
        dfK[col_label] = np.where(df[col_viol], 
                                 (srLoss - df[col_es]) / (df[col_es] - dMu), 
                                 0)
        
        vI = df[col_viol] == 1  
        vKv = dfK[col_label][vI] 
        
        print(f"No. of {col_viol}: {vI.sum()}")
        
        iDf = len(vKv)
        if iDf > 0:  
            dS2k = vKv.var() 
            dS2kavg = dS2k / iDf  
            dSv = np.sqrt(dS2kavg)  
            dMv = vKv.mean() 
            
            dT = (dMv - 0) / dSv if dSv != 0 else np.nan
            dPt0 = stats.t.cdf(dT, df=iDf) if iDf > 0 else np.nan
            dPt = 2 * min(dPt0, 1 - dPt0) if not np.isnan(dPt0) else np.nan
        else:
            dMv, dSv, dT, dPt = 0, 0, np.nan, np.nan
        
        dfTES[col_label] = {
            'K-mean': dMv,
            'K-sdev': dSv,
            't-ES': dT,
            'p-ES': dPt
        }
    
    return dfTES.T, dfK

alphas = [0.975, 0.99] 
dfTES, dfK = test_es_residuals(df_m4, alphas)
print("=== t-test for ES violation residuals ===")
print(dfTES.round(4))  

## qq-plot for this method 
dfViol = pd.DataFrame(index=df_m4.index)

for alpha in alphas:
    col_label = f'ES_{int(alpha * 1000)}'
    col_viol = f'Violation_{int(alpha * 1000)}'
    dfViol[col_label] = df_m4[col_viol] 

def plot_qq_residuals(dfK, dfViol, alpha):
    iC = dfK.shape[1]
    plt.figure(figsize=(6 * iC, 5))

    for i, sC in enumerate(dfK.columns):
        residuals = dfK.loc[dfViol[sC] == 1, sC]

        plt.subplot(1, iC, i + 1)
        stats.probplot(residuals, dist=stats.expon, plot=plt) 
        plt.title(f'QQ plot: ES residuals (α = {int(alpha[i]*100)}%)')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")

    plt.tight_layout()
    # plt.savefig('./qqplot_m4.png')
    plt.show() 

plot_qq_residuals(dfK, dfViol, alphas)

## plot for violation, VaR and ES, portfolio losses 
portfolio_returns_filtered = consolidate_data[['Date', 'PORTFOLIO']] 
portfolio_returns_filtered['PORTFOLIO'] = portfolio_returns_filtered['PORTFOLIO']*100 
portfolio_returns_filtered = portfolio_returns_filtered.set_index('Date') 

var_990_series = df_m4[['VaR_990']] 
var_975_series = df_m4[['VaR_975']]
es_990_series = df_m4[['ES_990']] 
es_975_series = df_m4[['ES_975']]

violations_990 = portfolio_returns_filtered['PORTFOLIO'].values > var_990_series['VaR_990'].values 
violations_975 = portfolio_returns_filtered['PORTFOLIO'].values > var_975_series['VaR_975'].values 

print(f"Violation (VaR_990): {violations_990.sum()}")
print(f"Violation (VaR_975): {violations_975.sum()}") 

plt.figure(figsize=(12, 6))
plt.vlines(x=df_m4.index, ymin=0, ymax=portfolio_returns_filtered['PORTFOLIO'].values, color='grey', linewidth=0.5)
plt.plot(df_m4.index, var_990_series.values, linestyle='-', color='red', label='VaR 99%', linewidth=1)
plt.plot(df_m4.index, var_975_series.values, linestyle='-', color='blue', label='VaR 97.5%', linewidth=1)
plt.plot(df_m4.index, es_990_series.values, linestyle='--', color='pink', label='ES 99%', linewidth=1)
plt.plot(df_m4.index, es_975_series.values, linestyle='--', color='royalblue', label='ES 97.5%', linewidth=1)

if violations_990.sum() > 0:
    plt.scatter(df_m4.index[violations_990], portfolio_returns_filtered[violations_990],
                facecolors='blue', edgecolors='black', label='Violation VaR 99%')

if violations_975.sum() > 0:
     plt.scatter(df_m4.index[violations_975], portfolio_returns_filtered[violations_975],
                facecolors='orange', edgecolors='orange', label='Violation VaR 97.5%', alpha=0.8) 

plt.xlabel('Year-Month')
plt.ylabel('Portfolio Loss %')
# plt.title('Portfolio Loss with VaR 99% and 97.5% Violations and ES for GARCH-CCC')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# plt.savefig('./method4_rolling.png') 
plt.show()

##### GARCH-CCC ##### 
## full sample: 1-day VaR and ES 
def GARCH_CCC_VaR_ES(portfolio_data, vW, vAlpha):
    # Exclude cash from GARCH modeling (last column is cash). Handles cash separately since it's not appropriate for GARCH modeling
    asset_returns = portfolio_data.iloc[:, :-1]
    cash_returns = portfolio_data.iloc[:, -1]
    k = asset_returns.shape[1]  # Number of assets
    
    # Step 1: Estimate GARCH(1,1) for each asset separately to get conditional variances
    garch_models = {}
    conditional_vols = pd.DataFrame(index=asset_returns.index)
    
    for asset in asset_returns.columns:
        # Fit GARCH(1,1) model
        model = arch_model(asset_returns[asset], vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(update_freq=0, disp='off')
        garch_models[asset] = res
        
        # Get conditional volatilities (standard deviations)
        conditional_vols[asset] = res.conditional_volatility
    
    # Step 2: Construct diagonal matrix of conditional volatilities
    # We'll represent this as a DataFrame where each row is diag(σ1,t, σ2,t, ..., σk,t)
    Delta_t = conditional_vols.copy()
    
    # Step 3: Get devolatilised process Y_t = Δ_t^{-1} X_t
    Y_t = asset_returns / conditional_vols
    
    # Step 4: Calculate constant correlation matrix P
    P = Y_t.corr()
    
    # For each day, construct conditional covariance matrix Σ_t = Δ_t P Δ_t
    # We'll calculate this for the last day to get the most recent covariance matrix
    last_Delta = np.diag(Delta_t.iloc[-1])
    Sigma_t = last_Delta @ P.values @ last_Delta
    
    # Add cash (assume zero covariance with other assets)
    # Create extended covariance matrix with cash as last element
    extended_Sigma = np.zeros((k+1, k+1))
    extended_Sigma[:k, :k] = Sigma_t
    # Cash variance (3-month Euribor has very small variance)
    extended_Sigma[-1, -1] = cash_returns.var()
    
    vR = np.array(portfolio_data.mean().values)  # Mean returns (including cash)
    
    print("\n=== GARCH(1,1) with Constant Conditional Correlation Method ===")
    for alpha in vAlpha:
        dVaR, dES = vc_multi_normal_VaR_ES(vW, vR, extended_Sigma, alpha)
        print(f'\nConfidence Level: {int(alpha*1000)}%')
        print(f'  VaR: {dVaR.item():.4f}%')  
        print(f'  ES : {dES.item():.4f}%')    

vW = np.array([0.25, 0.20, 0.25, 0.3]).reshape(-1, 1) 
vAlpha = np.array([0.975, 0.99])

df_m4_full = GARCH_CCC_VaR_ES(portfolio_data, vW, vAlpha) 

# define crisis periods as report 
portfolio_data.index = pd.to_datetime(portfolio_data.index)

portfolio_data['crisis_period_ind'] = portfolio_data.apply(lambda x: 1 if pd.Timestamp('2015-06-01') <= x.name <= pd.Timestamp('2016-06-01')
                                                            or pd.Timestamp('2020-02-01') <= x.name < pd.Timestamp('2020-05-01')
                                                            or pd.Timestamp('2021-01-01') <= x.name < pd.Timestamp('2023-01-01')
                                                            else 0, axis=1) 

portfolio_data_crisis = portfolio_data[portfolio_data['crisis_period_ind']==1] 
portfolio_data_non_crisis = portfolio_data[portfolio_data['crisis_period_ind']==0] 

del portfolio_data_crisis['crisis_period_ind'] 
del portfolio_data_non_crisis['crisis_period_ind'] 

vW = np.array([0.25, 0.20, 0.25, 0.3]).reshape(-1, 1) # Portfolio weights (including cash)
vAlpha = np.array([0.975, 0.99])

df_m4_crisis = GARCH_CCC_VaR_ES(portfolio_data_crisis, vW, vAlpha) 
df_m4_non_crisis = GARCH_CCC_VaR_ES(portfolio_data_non_crisis, vW, vAlpha)  

## Weibull LR test 
def weibull_lr_test_exceedances(exceedances):
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(exceedances, floc=0)
    LL_weibull = np.sum(stats.weibull_min.logpdf(exceedances, shape_weibull, loc=0, scale=scale_weibull))
    
    scale_exp = exceedances.mean()
    LL_exp = np.sum(stats.expon.logpdf(exceedances, loc=0, scale=scale_exp))
    
    LR_stat = 2 * (LL_weibull - LL_exp)
    p_value = 1 - stats.chi2.cdf(LR_stat, df=1)
    
    return {
        'Shape_Weibull': shape_weibull,
        'Scale_Weibull': scale_weibull,
        'LR_stat': LR_stat,
        'p_value': p_value,
        'Num_exceedances': len(exceedances)} 

# VaR 99% 
losses = df_m4['Loss']
VaR_series = df_m4['VaR_990']

mask_valid = VaR_series.notna()
losses = losses[mask_valid]
VaR_series = VaR_series[mask_valid]

exceedances_mask = losses > VaR_series
exceedances = losses[exceedances_mask] - VaR_series[exceedances_mask]
exceedances = exceedances[exceedances > 0]

print(f"Number of exceedances: {len(exceedances)}") 
results = weibull_lr_test_exceedances(exceedances)
print(results) 

# VaR 97.5% 
losses = df_m4['Loss']
VaR_series = df_m4['VaR_975']

mask_valid = VaR_series.notna()
losses = losses[mask_valid]
VaR_series = VaR_series[mask_valid]

exceedances_mask = losses > VaR_series
exceedances = losses[exceedances_mask] - VaR_series[exceedances_mask]
exceedances = exceedances[exceedances > 0]

print(f"Number of exceedances: {len(exceedances)}") 
results = weibull_lr_test_exceedances(exceedances)
print(results) 


##### Filtered Historical Simulation - EWMA ##### 
## rolling window 
def rolling_filtered_historical_simulation_EWMA(portfolio_data, weights, alphas=[0.975, 0.99], lambda_=0.94, window=522):
    T, N = portfolio_data.shape
    results = pd.DataFrame(index=portfolio_data.index[window:])
    all_z = []  # standardized residuals 

    weights = weights.reshape(-1, 1)

    for t in range(window, T):
        data_window = portfolio_data.iloc[t-window:t]
        mu = np.zeros((window, N))
        s2 = np.zeros((window, N))
        z = np.zeros((window, N))

        for i in range(N):
            mu[0, i] = data_window.iloc[:100, i].mean()
            s2[0, i] = np.var(data_window.iloc[:100, i] - mu[0, i])
            z[0, i] = (data_window.iloc[0, i] - mu[0, i]) / np.sqrt(s2[0, i])
            for j in range(1, window):
                mu[j, i] = lambda_ * mu[j - 1, i] + (1 - lambda_) * data_window.iloc[j - 1, i]
                s2[j, i] = lambda_ * s2[j - 1, i] + (1 - lambda_) * (data_window.iloc[j - 1, i] - mu[j - 1, i]) ** 2
                z[j, i] = (data_window.iloc[j, i] - mu[j, i]) / np.sqrt(s2[j, i])

        all_z.append(z[-1])  

        # Step 2: Correlation of residuals
        P = np.corrcoef(z.T)
        Delta_t = np.diag(np.sqrt(s2[-1, :]))
        S2_t = Delta_t @ P @ Delta_t
        vR = mu[-1, :]

        sigma_p = np.sqrt(weights.T @ S2_t @ weights).item()
        mu_p = float(weights.T @ vR.reshape(-1, 1)) 
        results.loc[portfolio_data.index[t], 'Portfolio_Volatility'] = sigma_p

        for alpha in alphas:
            z_alpha = norm.ppf(alpha)
            VaR = -mu_p + sigma_p * z_alpha
            ES = -mu_p + sigma_p * norm.pdf(z_alpha) / (1 - alpha)

            results.loc[portfolio_data.index[t], f'VaR_{int(alpha * 1000)}'] = VaR
            results.loc[portfolio_data.index[t], f'ES_{int(alpha * 1000)}'] = ES

    df_z = pd.DataFrame(all_z, index=portfolio_data.index[window:])
    results['standardized_residual'] = df_z @ weights 

    return results

del portfolio_data['crisis_period_ind'] 

portfolio_weights = np.array([0.25, 0.20, 0.25, 0.30])

ewma_df = rolling_filtered_historical_simulation_EWMA(portfolio_data, portfolio_weights, lambda_=0.94, window=261)
portfolio_returns = portfolio_data.dot(portfolio_weights) 

df_m5 = pd.DataFrame({'Return': portfolio_returns.squeeze(), 'Loss': -portfolio_returns.squeeze()}).join(ewma_df)  

# VaR backtest 
df_m5.index = pd.to_datetime(df_m5.index) 
df_m5['year'] = df_m5.index.year 

result_dict = backtest_var_es_by_year(df_m5, alphas=[0.975, 0.99])

for label, res in result_dict.items():
    print(f"\nBacktest Results for {label}:")
    print(res.round(4)) 

# VaR binomial test 
df_m5['Violation_975'] = (df_m5['Loss'] > df_m5['VaR_975']).astype(int)
df_m5['Violation_990'] = (df_m5['Loss'] > df_m5['VaR_990']).astype(int) 

binomial_test_m5_975 = binomial_var_backtest(df_m5, alpha = 0.975)
print(binomial_test_m5_975)

binomial_test_m5_99 = binomial_var_backtest(df_m5, alpha = 0.99)
print(binomial_test_m5_99)  # reject H0: violation no abnormal, true = discrepancies 

# ES test 
alphas = [0.975, 0.99] 
dfTES, dfK = test_es_residuals(df_m5, alphas)
print("=== t-test for ES violation residuals ===")
print(dfTES.round(4)) 

# qq-plot for this method 
dfViol = pd.DataFrame(index=df_m5.index)
alphas=[0.975, 0.99]

for alpha in alphas:
    col_label = f'ES_{int(alpha * 1000)}'
    col_viol = f'Violation_{int(alpha * 1000)}'
    dfViol[col_label] = df_m5[col_viol] 

def plot_qq_residuals(dfK, dfViol, alpha):
    iC = dfK.shape[1]
    plt.figure(figsize=(6 * iC, 5))

    for i, sC in enumerate(dfK.columns):
        residuals = dfK.loc[dfViol[sC] == 1, sC]

        plt.subplot(1, iC, i + 1)
        stats.probplot(residuals, dist=stats.expon, plot=plt) 
        plt.title(f'QQ plot: ES residuals (α = {int(alpha[i]*100)}%)')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")

    plt.tight_layout()
    # plt.savefig('./qqplot_m5.png')
    plt.show() 

plot_qq_residuals(dfK, dfViol, alphas) 

## plot for violation, VaR and ES, portfolio losses 
var_990_series = df_m5[['VaR_990']] 
var_975_series = df_m5[['VaR_975']]
es_990_series = df_m5[['ES_990']] 
es_975_series = df_m5[['ES_975']]

violations_990 = portfolio_returns_filtered['PORTFOLIO'].values > var_990_series['VaR_990'].values 
violations_975 = portfolio_returns_filtered['PORTFOLIO'].values > var_975_series['VaR_975'].values 

print(f"Violation (VaR_990): {violations_990.sum()}")
print(f"Violation (VaR_975): {violations_975.sum()}") 
 
plt.figure(figsize=(12, 6))

plt.vlines(x=df_m5.index, ymin=0, ymax=portfolio_returns_filtered['PORTFOLIO'].values, color='grey', linewidth=0.5)

plt.plot(df_m5.index, var_990_series.values, linestyle='-', color='red', label='VaR 99%', linewidth=1)
plt.plot(df_m5.index, var_975_series.values, linestyle='-', color='blue', label='VaR 97.5%', linewidth=1)
plt.plot(df_m5.index, es_990_series.values, linestyle='--', color='pink', label='ES 99%', linewidth=1)
plt.plot(df_m5.index, es_975_series.values, linestyle='--', color='royalblue', label='ES 97.5%', linewidth=1)

if violations_990.sum() > 0:
    plt.scatter(df_m5.index[violations_990], portfolio_returns_filtered[violations_990],
                facecolors='blue', edgecolors='black', label='Violation VaR 99%')

if violations_975.sum() > 0:
    plt.scatter(df_m5.index[violations_975], portfolio_returns_filtered[violations_975],
                facecolors='orange', edgecolors='orange', label='Violation VaR 97.5%', alpha=0.8) 

plt.xlabel('Year-Month')
plt.ylabel('Portfolio Loss %')
# plt.title('Portfolio Loss with VaR 99% and 97.5% Violations and ES for GARCH-CCC')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# plt.savefig('./method5_rolling_97.png') 
plt.show()

## full sample: 1-day VaR and ES 
def filtered_historical_simulation_EWMA(portfolio_data, vW, vAlpha, lambda_=0.94):
    T, N = portfolio_data.shape
    
    mu = np.zeros((T,N))
    s2 = np.zeros((T, N))
    z = np.zeros((T, N))

    for i in range(N):
        mu[0, i] = portfolio_data.iloc[:100, i].mean()
        s2[0, i] = np.var(portfolio_data.iloc[:100, i]- mu[0, i], axis=0)
        z[0, i] = (portfolio_data.iloc[0, i]- mu[0, i])/s2[0, i]
        for t in range(0, T - 1):
            mu[t+1, i] = lambda_ * mu[t, i] + (1 - lambda_) * portfolio_data.iloc[t, i]
            s2[t+1, i] = lambda_ * s2[t, i] + (1 - lambda_) * (portfolio_data.iloc[t, i] - mu[t, i])**2
            z[t+1, i] = (portfolio_data.iloc[t, i] - mu[t, i])/ np.sqrt(s2[t, i])
    
    # Step 2: Standardized residuals correlation matrix
    P = np.corrcoef(z.T)  # N x N correlation matrix

    for t in range(T):
        # Diagonal matrix of standard deviations
        Delta_t = np.diag(np.sqrt(s2[t, :]))
        
        # Covariance matrix
        S2_t = Delta_t @ P @ Delta_t
    
    vR = np.mean(mu, axis=0)
    print("\n=== Filtered Historical Simulation with EWMA ===")
    for alpha in vAlpha:
        dVaR, dES = vc_multi_normal_VaR_ES(vW, vR, S2_t, alpha)
        print(f'\nConfidence Level: {int(alpha*100)}%')
        print(f'  VaR: {dVaR.item():.4f}%')  
        print(f'  ES : {dES.item():.4f}%')   

vW = np.array([0.25, 0.20, 0.25, 0.3]).reshape(-1, 1) # Portfolio weights (including cash)
vAlpha = np.array([0.975, 0.99])

df_m5_full = filtered_historical_simulation_EWMA(portfolio_data, vW, vAlpha) 

# crisis & non-crisis 
df_m5_crisis = filtered_historical_simulation_EWMA(portfolio_data_crisis, vW, vAlpha) 
df_m5_non_crisis = filtered_historical_simulation_EWMA(portfolio_data_non_crisis, vW, vAlpha)  

# Weibull LR test 
# VaR 99 
losses = df_m5['Loss']
VaR_series = df_m5['VaR_990']

mask_valid = VaR_series.notna()
losses = losses[mask_valid]
VaR_series = VaR_series[mask_valid]

exceedances_mask = losses > VaR_series
exceedances = losses[exceedances_mask] - VaR_series[exceedances_mask]
exceedances = exceedances[exceedances > 0]

print(f"Number of exceedances: {len(exceedances)}") 
results = weibull_lr_test_exceedances(exceedances)
print(results) 

# VaR 97.5% 
losses = df_m5['Loss']
VaR_series = df_m5['VaR_975']

mask_valid = VaR_series.notna()
losses = losses[mask_valid]
VaR_series = VaR_series[mask_valid]

exceedances_mask = losses > VaR_series
exceedances = losses[exceedances_mask] - VaR_series[exceedances_mask]
exceedances = exceedances[exceedances > 0]

print(f"Number of exceedances: {len(exceedances)}")
results = weibull_lr_test_exceedances(exceedances)
print(results) 

########## calibration test ########## 
## compare with volatility 
# GARCH-CCC plot 
plt.figure(figsize=(12,6))
plt.plot(df_m4.index, df_m4['Portfolio_Volatility'], label='portfolio violatility')
plt.plot(df_m4.index, df_m4['VaR_975'], label='VaR_97.5%', alpha=0.8)
plt.plot(df_m4.index, df_m4['ES_975'], label='ES_97.5%')
plt.plot(df_m4.index, df_m4['VaR_990'], label='VaR_99%', alpha=0.8)
plt.plot(df_m4.index, df_m4['ES_990'], label='ES_99%', alpha=0.8)
plt.title('Rolling Portfolio Volatility & VaR and ES of GARCH-CCC')
plt.xlabel('Date') 
plt.ylabel('Volatility')
plt.legend()
# plt.grid()
# plt.savefig('./volatility_garchccc.png')
plt.show() 

# FHS-EWMA plot 
plt.figure(figsize=(12,6))
plt.plot(df_m5.index, df_m5['Portfolio_Volatility'], label='portfolio violatility')
plt.plot(df_m5.index, df_m5['VaR_975'], label='VaR_97.5%', alpha=0.8)
plt.plot(df_m5.index, df_m5['ES_975'], label='ES_97.5%')
plt.plot(df_m5.index, df_m5['VaR_990'], label='VaR_99%', alpha=0.8)
plt.plot(df_m5.index, df_m5['ES_990'], label='ES_99%', alpha=0.8)
plt.title('Rolling Portfolio Volatility & VaR and ES of FHS-EWMA')
plt.xlabel('Date') 
plt.ylabel('Volatility')
plt.legend()
# plt.grid()
# plt.savefig('./volatility_fhs_ewma.png')
plt.show() 

## change the estimation window from 1 year to 2 year 
# GARCH-CCC 
rolling_df_v2 = rolling_GARCH_CCC_var_es(portfolio_data, portfolio_weights, alphas=[0.975, 0.99], window=522) 

df_m4_v2 = pd.DataFrame({'Return': portfolio_returns, 
                      'Loss': -portfolio_returns}).join(rolling_df_v2) 
plt.figure(figsize=(12,6))
plt.plot(df_m4.index, df_m4['Portfolio_Volatility'], label='portfolio violatility_261day')
plt.plot(df_m4_v2.index, df_m4_v2['Portfolio_Volatility'], label='portfolio violatility_522day')
plt.plot(df_m4.index, df_m4['VaR_975'], label='VaR_97.5%_261day', alpha=0.8)
plt.plot(df_m4_v2.index, df_m4_v2['VaR_975'], label='VaR_97.5%_522day', alpha=0.8)
# plt.plot(df_m4.index, df_m4['ES_975'], label='ES_97.5%_261day')
# plt.plot(df_m4_v2.index, df_m4_v2['ES_975'], label='ES_97.5%_522day')
# plt.plot(df_m4.index, df_m4['VaR_990'], label='VaR_99%', alpha=0.8)
# plt.plot(df_m4.index, df_m4['ES_990'], label='ES_99%', alpha=0.8)
plt.title('Rolling Portfolio Volatility & VaR and ES of GARCH-CCC for different window')
plt.xlabel('Date') 
plt.ylabel('Volatility')
plt.legend()
# plt.grid()
# plt.savefig('./volatility_garchccc_windows.png')
plt.show()

# FHS-EWMA 
rolling_ewma_df_v2 = rolling_filtered_historical_simulation_EWMA(portfolio_data, portfolio_weights, alphas=[0.975, 0.99], window=522) 

df_m5_v2 = pd.DataFrame({'Return': portfolio_returns, 
                      'Loss': -portfolio_returns}).join(rolling_ewma_df_v2) 

plt.figure(figsize=(12,6))
plt.plot(df_m5.index, df_m5['Portfolio_Volatility'], label='portfolio violatility_261day')
plt.plot(df_m5_v2.index, df_m5_v2['Portfolio_Volatility'], label='portfolio violatility_522day')
plt.plot(df_m5.index, df_m5['VaR_975'], label='VaR_97.5%_261day', alpha=0.8)
plt.plot(df_m5_v2.index, df_m5_v2['VaR_975'], label='VaR_97.5%_522day', alpha=0.8)
plt.title('Rolling Portfolio Volatility & VaR and ES of GARCH-CCC for different window')
plt.xlabel('Date') 
plt.ylabel('Volatility')
plt.legend()
# plt.grid()
# plt.savefig('./volatility_fhs_ewma_windows.png')
plt.show()

########## assumption test ########## 
## change into student-t distribution 
# change distribution from normal to student t distribution 
def rolling_GARCH_CCC_var_es_t(portfolio_data, weights, alphas, window):
    weights = weights.reshape(-1, 1)
    k_assets = portfolio_data.shape[1] - 1  # exclude cash
    df_out = pd.DataFrame(index=portfolio_data.index)

    for alpha in alphas:
        var_series = []
        es_series = []
        portfolio_volatility = [] 
        standardized_residual = [] # for standardized residual 

        for i in range(window, len(portfolio_data)):
            window_data = portfolio_data.iloc[i-window:i]
            asset_returns = window_data.iloc[:, :-1]
            cash_returns = window_data.iloc[:, -1]
            k = asset_returns.shape[1]

            conditional_vols = pd.DataFrame(index=asset_returns.index)
            for asset in asset_returns.columns:
                model = arch_model(asset_returns[asset], vol='Garch', p=1, q=1, dist='t') 
                res = model.fit(disp='off')
                conditional_vols[asset] = res.conditional_volatility

            last_vols = conditional_vols.iloc[-1].values
            Delta = np.diag(last_vols)
            Y_t = asset_returns / conditional_vols
            P = Y_t.corr().values
            portfolio_z = Y_t.iloc[-1].values @ weights[:-1] 
            standardized_residual.append(portfolio_z.item()) 

            Sigma = Delta @ P @ Delta

            extended_Sigma = np.zeros((k+1, k+1))
            extended_Sigma[:k, :k] = Sigma
            extended_Sigma[-1, -1] = cash_returns.var() 
            
            port_std = np.sqrt(np.dot(weights.T, np.dot(extended_Sigma, weights))) 
            portfolio_volatility.append(port_std.item()) 

            mu_vec = window_data.mean().values.reshape(-1, 1)

            dVaR, dES = vc_multi_normal_VaR_ES(weights, mu_vec, extended_Sigma, alpha)

            var_series.append(dVaR.item())
            es_series.append(dES.item())

        df_out[f'VaR_{int(alpha*1000)}'] = [np.nan] * window + var_series
        df_out[f'ES_{int(alpha*1000)}'] = [np.nan] * window + es_series
        df_out['Portfolio_Volatility'] = [np.nan] * window + portfolio_volatility
        df_out['standardized_residual'] = [np.nan] * window + standardized_residual

    return df_out

rolling_df_t = rolling_GARCH_CCC_var_es_t(portfolio_data, portfolio_weights, alphas=[0.975, 0.99], window=261) # window defined in report 

df_m4_t = pd.DataFrame({'Return': portfolio_returns, 
                      'Loss': -portfolio_returns}).join(rolling_df_t)   

plt.figure(figsize=(12,6))
plt.plot(df_m4.index, df_m4['VaR_975'], label='VaR_97.5%_normal')
plt.plot(df_m4_t.index, df_m4_t['VaR_975'], label='VaR_97.5%_t', alpha=0.8)
plt.title('VaR of GARCH-CCC for normal and student-t distribution')
plt.xlabel('Date') 
plt.ylabel('Volatility')
plt.legend()
# plt.savefig('./comparison_garchccc_t_normal.png')
plt.show() 

## LB test for GARCH-CCC 
df_m4_lb = df_m4['standardized_residual'].squeeze().dropna() 
lb_test = acorr_ljungbox(df_m4_lb, lags=[10, 20], return_df=True)
print(lb_test)
# GARCH-CCC assumes standardized residuals follow i.i.d p>0.05, test pass, otherwise residuals have autocorrelation, model needs to be adjusted 

## distribution invariance for historical simulation - Kolmogorov-Smirnov test - for GARCH-CCC 
T = len(df_m4_lb)
portfolio_z_early = df_m4_lb[:T//2]   
portfolio_z_late = df_m4_lb[T//2:]   

ks_stat, ks_pvalue = ks_2samp(portfolio_z_early, portfolio_z_late)
print(f"KS-statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}") 

# for crisis and non-crisis 
z_df_m4 = df_m4[['standardized_residual']].dropna()
z_df_m4['crisis_period_ind'] = z_df_m4.apply(lambda x: 1 if pd.Timestamp('2015-06-01') <= x.name <= pd.Timestamp('2016-06-01')
                                                            or pd.Timestamp('2020-02-01') <= x.name < pd.Timestamp('2020-05-01')
                                                            or pd.Timestamp('2021-01-01') <= x.name < pd.Timestamp('2023-01-01')
                                                            else 0, axis=1)

portfolio_z_crisis_df_m4 = z_df_m4[z_df_m4['crisis_period_ind']==1]
portfolio_z_non_crisis_df_m4 = z_df_m4[z_df_m4['crisis_period_ind']==0] 

portfolio_z_crisis_m4 = portfolio_z_crisis_df_m4['standardized_residual'].squeeze()  
portfolio_z_non_crisis_m4 = portfolio_z_non_crisis_df_m4['standardized_residual'].squeeze() 

ks_stat, ks_pvalue = ks_2samp(portfolio_z_crisis_m4, portfolio_z_non_crisis_m4)
print(f"KS-statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")  


## LB test for FHS-EWMA 
tem = df_m5['standardized_residual'].squeeze().dropna() 
lb_test = acorr_ljungbox(tem, lags=[10, 20], return_df=True)
print(lb_test)
# h0: independence 
# h1: autocorrelation 
# p-value<0.05, cannot reject h0 --> possibly underestimate extreme risks 

## distribution invariance for historical simulation - Kolmogorov-Smirnov test - for FHS-EWMA 
T = len(tem)
portfolio_z_early = tem[:T//2]   
portfolio_z_late = tem[T//2:]   

ks_stat, ks_pvalue = ks_2samp(portfolio_z_early, portfolio_z_late)
print(f"KS-statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")  

# crisis and non crisis 
z_df = df_m5[['standardized_residual']].dropna()

z_df.index = pd.to_datetime(z_df.index)
z_df['crisis_period_ind'] = z_df.apply(lambda x: 1 if pd.Timestamp('2015-06-01') <= x.name <= pd.Timestamp('2016-06-01')
                                                            or pd.Timestamp('2020-02-01') <= x.name < pd.Timestamp('2020-05-01')
                                                            or pd.Timestamp('2021-01-01') <= x.name < pd.Timestamp('2023-01-01')
                                                            else 0, axis=1) 

portfolio_z_crisis_df = z_df[z_df['crisis_period_ind']==1]
portfolio_z_non_crisis_df = z_df[z_df['crisis_period_ind']==0] 

portfolio_z_crisis = portfolio_z_crisis_df['standardized_residual'].squeeze()  
portfolio_z_non_crisis = portfolio_z_non_crisis_df['standardized_residual'].squeeze() 

ks_stat, ks_pvalue = ks_2samp(portfolio_z_crisis, portfolio_z_non_crisis)
print(f"KS-statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")   

##############################################################################
### Sensitivity Analysis
def vc_multi_studentt_VaR_ES(vW, vR, mS2, ivDF, vAlpha):
    port_mean = vW.dot(vR)
    port_std  = np.sqrt(vW @ mS2 @ vW)
    dVaR0     = sts.t.ppf(vAlpha, df=ivDF)
    c         = port_std / np.sqrt(ivDF/(ivDF-2))
    dVaR      = -port_mean + c * dVaR0
    dES0      = sts.t.pdf(dVaR0, df=ivDF) * ((ivDF + dVaR0**2)/(ivDF-1)) / (1-vAlpha)
    dES       = port_mean + c * dES0
    return dVaR, dES


data_df = pd.read_csv(
    r"C:\Users\rwe20\OneDrive\문서\a14_135120_9019403_portfolio_returns.csv",
    parse_dates=['Date'], index_col='Date'
)


port_ret = data_df['PORTFOLIO']
vW       = np.array([1.0])
vR       = np.array([port_ret.mean()])
mS2      = np.array([[port_ret.var()]])

dfs     = [3,5,10]
lambdas = [0.94,0.97,0.99]
alphas  = [0.975,0.99]

results = []

# 1) VC-Student’s t
for df_ in dfs:
    for alpha in alphas:
        var_t, es_t = vc_multi_studentt_VaR_ES(vW, vR, mS2, df_, alpha)
        results.append({
            'method':'VC-StudentT','param':df_,
            'alpha':alpha,'VaR':float(var_t),'ES':float(es_t)
        })

asset_cols    = ['^GSPC','^IBEX','^N225','LOAN_RETURN']
asset_returns = data_df[asset_cols]
weights       = np.array([0.25,0.20,0.25,0.30])

for lam in lambdas:

    ewma_var = asset_returns.ewm(alpha=1-lam,adjust=False).var()
    
    valid = ewma_var.notna().all(axis=1)        
    ewma_var = ewma_var.loc[valid]
    R        = asset_returns.loc[valid]
    # --------------------------

    z_assets = R / np.sqrt(ewma_var)
    P        = z_assets.corr()
    
    sig_T    = np.sqrt(ewma_var.iloc[-1].values)
    Sigma_T  = np.diag(sig_T) @ P.values @ np.diag(sig_T)
    sig_p_T  = np.sqrt(weights @ Sigma_T @ weights)
    
    mu       = port_ret.mean()
    var_port = ewma_var.mul(weights**2,axis=1).sum(axis=1)
    s_t      = R.dot(weights) / np.sqrt(var_port)
    
    for alpha in alphas:
        q     = np.quantile(s_t,alpha)
        var_c = -mu + sig_p_T * q
        es_c  = -mu + sig_p_T * s_t[s_t>=q].mean()
        results.append({
            'method':'FHS-CCC','param':lam,
            'alpha':alpha,'VaR':float(var_c),'ES':float(es_c)
        })


df_res = pd.DataFrame(results)
#print(df_res)


###############################################################################
### Convergence & Stabiltiy Test ###
asset_cols = ['^GSPC', '^IBEX', '^N225', 'LOAN_RETURN']
portfolio_weights = np.array([0.25, 0.20, 0.25, 0.30])
portfolio_returns = portfolio_data[asset_cols].dot(portfolio_weights)

def rolling_beta_stability(asset_series: pd.Series, window: int = 261) -> pd.Series:
    betas = []
    dates = asset_series.index[window:]

    for i in range(window, len(asset_series)):
        window_ret = asset_series.iloc[i-window:i].dropna()
        try:
            res = arch_model(window_ret, vol='Garch', p=1, q=1, dist='normal').fit(disp='off')
            #rint(res)
            beta_val = res.params['beta[1]']
        except:
            beta_val = np.nan
        betas.append(beta_val)

    return pd.Series(betas, index=dates, name='beta')


def plot_beta_with_jumps(beta_series: pd.Series, label: str, threshold: float = 0.1):
    plt.figure(figsize=(12, 5))
    plt.plot(beta_series, label=f'{label} GARCH(1,1) beta', color='blue')

    jumps = beta_series[beta_series < threshold]
    plt.scatter(jumps.index, jumps.values, color='red', label=f'Beta < {threshold}', zorder=5)

    plt.title(f'{label} Rolling GARCH Beta with Jump Detection')
    plt.xlabel('Date')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True)
    plt.show()

for index_name in ['^GSPC', '^IBEX', '^N225']:
    asset_returns = portfolio_data[index_name]
    beta_series = rolling_beta_stability(asset_returns, window=261)
    plot_beta_with_jumps(beta_series, label=index_name, threshold=0.1)

# the first & last window for comparison - model technique part results 
# GARCH-CCC 
portfolio_first_window = portfolio_data.iloc[:261]
portfolio_last_window = portfolio_data.iloc[-261:]

m4_first_window = GARCH_CCC_VaR_ES(portfolio_first_window, vW, vAlpha) 
m4_last_window = GARCH_CCC_VaR_ES(portfolio_last_window, vW, vAlpha) 

# FHS-EWMA 
m5_first_window = filtered_historical_simulation_EWMA(portfolio_first_window, vW, vAlpha) 
m5_last_window = filtered_historical_simulation_EWMA(portfolio_last_window, vW, vAlpha) 
