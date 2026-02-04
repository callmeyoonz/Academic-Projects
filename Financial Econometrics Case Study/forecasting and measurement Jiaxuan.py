import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw


def rolling_forcast(Theta, fv, r, model_type, rm = None):
# One-step-ahead rolling forecast
    log_fv = np.log(fv)
    for t in range(1, len(r)):
        if model_type == 'GARCH':
            omega, alpha, beta = Theta
            # GARCH(1,1) formula: vH[t] = omega + alpha * vR[t-1]^2 + beta * vH[t-1]
            fv[t] = (
                omega + 
                alpha * r[t-1]**2 + 
                beta * fv[t-1]
            )

        elif model_type == 'GJR_GARCH':
            omega, alpha, beta, gamma = Theta
            leverage_effect = gamma * r[t - 1] ** 2
            fv[t] = omega + alpha * r[t - 1] ** 2 + leverage_effect + beta * fv[t - 1]

        elif model_type == 'EGARCH':
            omega, alpha, beta, gamma = Theta
            eta_t_1 = r[t - 1] / np.sqrt(fv[t - 1])
            log_fv[t] = (
                omega
                + alpha * (np.abs(eta_t_1) - np.sqrt(2 / np.pi))
                + gamma * eta_t_1
                + beta * log_fv[t - 1]
            )
            fv[t] = np.exp(log_fv[t])

        elif model_type == 'RK_GARCH':
            omega, beta, gamma = Theta
            log_fv[t] = omega + beta * log_fv[t-1] + gamma * np.log(rm[t-1])
            fv[t] = np.exp(log_fv[t])

        elif model_type == 'RV_GARCH':
            omega, beta, gamma = Theta
            log_fv[t] = omega + beta * log_fv[t-1] + gamma * np.log(rm[t-1])
            fv[t] = np.exp(log_fv[t])


    results = pd.DataFrame({
        'Returns': r,
        'Forecasted Variance': fv
    })

    return results

def distri_forecast(values, dates, Theta):
    # values = [returns, fv, rk, rv]
    returns, fv, rk, rv = values
    
    theta = Theta[0]
    results = rolling_forcast(theta, fv, returns, 'GARCH')
    results.insert(0, 'Date', dates)

    theta = Theta[1]
    results_gjr = rolling_forcast(theta, fv, returns, 'GJR_GARCH')
    results_gjr.insert(0, 'Date', dates)
    
    theta = Theta[2]
    results_e = rolling_forcast(theta, fv, returns, 'EGARCH')
    results_e.insert(0, 'Date', dates)    

    theta = Theta[3]
    results_rk = rolling_forcast(theta, fv, returns, 'RK_GARCH', rk)
    results_rk.insert(0, 'Date', dates)  

    theta = Theta[4]
    results_rv = rolling_forcast(theta, fv, returns, 'RV_GARCH', rv)
    results_rv.insert(0, 'Date', dates) 

    return [results, results_gjr, results_e, results_rk, results_rv]    

# ## MCMC
def MCMC_forecast(values, dates, Theta):
    returns, fv = values
    
    theta = Theta[0]
    results = rolling_forcast(theta, fv, returns, 'GARCH')
    results.insert(0, 'Date', dates)

    theta = Theta[1]
    results_gjr = rolling_forcast(theta, fv, returns, 'GJR_GARCH')
    results_gjr.insert(0, 'Date', dates)

    theta = Theta[2]
    results_e = rolling_forcast(theta, fv, returns, 'EGARCH')
    results_e.insert(0, 'Date', dates)

    return [results, results_gjr, results_e]
    #print('Forecast under MCMC:')
    #plot_forecasted(results, results_gjr, results_e)


def RiskMetrics(values, dates):
# riskmetrics model to compare with GARCH
    r, fv = values
    
    lambd = 0.94
    for t in range(1, len(r)):
        fv[t] = (  # forecasted volatility
            lambd * fv[t-1] + 
            (1-lambd) * r[t-1]**2)  #r: daily return
        
    results = pd.DataFrame({
        'Date': dates,
        'Returns': r,
        'Forecasted Variance': fv
    })

    return results

if __name__ == '__main__':
    # Given estimation results
    daily_return_data = pd.read_csv("../data/new/daily_return_out.csv")
    returns = daily_return_data['daily_return'].values * 100  # Extract daily returns as a NumPy array
    dates = pd.to_datetime(daily_return_data['DATE'])
    fv = np.zeros_like(returns)
    fv[0] = np.var(returns)

    realized_measure = pd.read_csv('../RK/RK_out_0122.csv')
    rk = realized_measure['realized_kernel']
    rv = realized_measure['realized_volatility']

    values = [returns, fv, rk, rv]

    # 0:GARCH, 1:GJR-GARCH, 2:EGARCH, 3:RK-GARCH, 4:RV-GARCH

    # student-t
    Theta = [[0.221294, 0.159166, 0.781679],
             [0.181942, 0.057099, 0.811979, 0.164458],
             [0.071338, 0.152478, 0.962315, -0.114940],
             [0.277519, 0.411077, 0.500738],
             [0.562679, 0.333040, 0.683164]]
    #print('Forecast under student-t distribution:')
    st = distri_forecast(values, dates, Theta)

    # skewed student-t
    Theta = [[0.221638, 0.159199, 0.782113],
             [0.181837, 0.057210, 0.812558, 0.164008],
             [0.0307896, 0.152493, 0.962350, -0.114812],
             [0.119213, 0.701145, 0.285914],
             [0.227566, 0.700000, 0.319707]]
    #print('Forecast under Skewed Student-t distribution:')
    sst = distri_forecast(values, dates, Theta)

    # normal
    Theta = [[0.100000, 0.124946, 0.855202],
             [0.395725, 0.086196, 0.700000, 0.197051],
             [0.085580, 0.207665, 0.928606, -0.113209],
             [0.099671, 0.763700, 0.233606],
             [0.099994, 0.857231, 0.173298]]
    #print('Forecast under noraml distribution:')
    n = distri_forecast(values, dates, Theta)
    
    values = [returns, fv]
    Theta = [[0.000199, 0.115761, 0.770667],
            [0.000122, 0.061590, 0.861920, 0.165839],
            [0.010645, 0.214352, 0.993497, -0.067853]]
    mcmc = MCMC_forecast(values, dates, Theta)

    rme = RiskMetrics(values, dates)

    GAS_result = pd.read_csv('../GAS/GAS_forecasts_0126.csv')
    
    plt.figure(figsize=(15, 4))
    plt.plot(dates, st[0]['Forecasted Variance'], label='GARCH-STD', color='blue', alpha=0.7)
    plt.plot(dates, st[1]['Forecasted Variance'], label='GJR-GARCH-STD', color='red', alpha=0.7)
    plt.plot(dates, st[2]['Forecasted Variance'], label='EGARCH-STD', color='green', alpha=0.7)
    plt.plot(dates, st[3]['Forecasted Variance'], label='RK-GARCH-STD', color='black', alpha=1)
    plt.plot(dates, st[4]['Forecasted Variance'], label='RV-GARCH-STD', color='turquoise', alpha=0.7)

    plt.plot(dates, sst[0]['Forecasted Variance'], label='GARCH-SSTD', color='purple', alpha=0.7)
    plt.plot(dates, sst[1]['Forecasted Variance'], label='GJR-GARCH-SSTD', color='orange', alpha=0.7)
    plt.plot(dates, sst[2]['Forecasted Variance'], label='EGARCH-SSTD', color='cyan', alpha=0.7)
    plt.plot(dates, sst[3]['Forecasted Variance'], label='RK-GARCH-SSTD', color='teal', alpha=0.7)
    plt.plot(dates, sst[4]['Forecasted Variance'], label='RV-GARCH-SSTD', color='gold', alpha=0.7)

    plt.plot(dates, n[0]['Forecasted Variance'], label='GARCH-Norm', color='yellow', alpha=1)
    plt.plot(dates, n[1]['Forecasted Variance'], label='GJR-GARCH-Norm', color='magenta', alpha=0.7)
    plt.plot(dates, n[2]['Forecasted Variance'], label='EGARCH-Norm', color='black', alpha=0.7)
    plt.plot(dates, n[3]['Forecasted Variance'], label='RK-GARCH-Norm', color='khaki', alpha=1)
    plt.plot(dates, n[4]['Forecasted Variance'], label='RV-EGARCH-Norm', color='orchid', alpha=0.7)

    plt.plot(dates, GAS_result['GAS-Norm'], label='GAS-Norm', color='crimson', alpha=0.7)
    plt.plot(dates, GAS_result['GAS-STD'], label='GAS-STD', color='navy', alpha=0.7)
    plt.plot(dates, GAS_result['GAS-SSTD'], label='GAS-SSTD', color='chartreuse', alpha=0.7)
    plt.plot(dates, GAS_result['R-GAS-Norm'], label='R-GAS-Norm', color='black', alpha=0.7)
    plt.plot(dates, GAS_result['R-GAS-STD'], label='R-GAS-STD', color='maroon', alpha=0.7)
    plt.plot(dates, GAS_result['R-GAS-SSTD'], label='R-GAS-SSTD', color='aqua', alpha=0.7)

    plt.plot(dates, mcmc[0]['Forecasted Variance'], label='GARCH-MCMC', color='brown', alpha=0.7)
    plt.plot(dates, mcmc[1]['Forecasted Variance'], label='GJR-GARCH-MCMC', color='pink', alpha=1)
    plt.plot(dates, mcmc[2]['Forecasted Variance'], label='EGARCH-MCMC', color='gray', alpha=0.7)
    
    plt.plot(dates, rme['Forecasted Variance'], label='RiskMetrics', color='lime', alpha=0.7)

    plt.plot(dates, rk, label='Realized Kernal', color='chocolate', alpha=0.7)
    #plt.plot(dates, rv, label='Realized Volatility', color='chocolate', alpha=0.7)

    #plt.plot(dates, returns, label='DailyReturn', color='black', alpha=0.7)

    plt.title('Volatility forecast for models')
    plt.ylabel('Volatility')
    plt.xlabel('Date')
    plt.legend(fontsize=7, borderpad=0.2)
    plt.show()

# ## ModelPerformanceMetrics
def error_metrics(rv, fv, model):
    mse = np.mean((fv - rv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(fv - rv))
    mape = np.mean(np.abs((rv - fv) / rv))
    
    result = pd.DataFrame({
        'Model':[model],
        'MSE':[mse],
        'RMSE':[rmse],
        'MAE':[mae],
        'MAPE':[mape]
    })
    
    return result

model_name = ['GARCH', 'GJR-GARCH', 'EGARCH', 'RK-GARCH']
distribution = ['STD', 'SSTD', 'Norm']

em_st = []
em_sst = []
em_n = []
em_mcmc = []
em_rme = []
for i in range(len(model_name)):
    em_st.append(error_metrics(rk, 
                          st[i]['Forecasted Variance'], 
                          f"{model_name[i]}-{distribution[0]}"))
    em_sst.append(error_metrics(rk, 
                           sst[i]['Forecasted Variance'], 
                           f"{model_name[i]}-{distribution[1]}"))
    em_n.append(error_metrics(rk, 
                         n[i]['Forecasted Variance'], 
                         f"{model_name[i]}-{distribution[2]}"))
    
GAS_name = GAS_result.columns[2:]
GAS_metrics = []
for col in GAS_name:
    GAS_metrics.append(error_metrics(
        rk,  
        GAS_result[col],  
        col
    ))

for i in range(3):
    em_mcmc.append(error_metrics(rk, 
                         mcmc[i]['Forecasted Variance'], 
                         f"{model_name[i]}-{'MCMC'}"))
    
em_rme = error_metrics(rk, 
                         rme['Forecasted Variance'], 
                         'RiskMetrics')

all_data = pd.concat(em_st+em_sst+em_n+em_mcmc+GAS_metrics, ignore_index=True)
all_data = pd.concat([all_data, em_rme], ignore_index=True)
print(all_data)

all_data["Score"] = all_data["MSE"] + all_data["RMSE"] + all_data["MAE"] + all_data["MAPE"]

top_score = all_data.nsmallest(22, "Score")
print("\nTop 22 Models by Combined Score")
print(top_score)

# get error series and all fv series
def get_error(rk, fv, model_name):
    er = rk - fv

    result_er = pd.DataFrame({
        model_name:er,
    })

    result_fv = pd.DataFrame({
        model_name:fv,
    })
    return result_er, result_fv

model_name = ['GARCH', 'GJR-GARCH', 'EGARCH', 'RK-GARCH']
distribution = ['STD', 'SSTD', 'Norm']

er_st = []
er_sst = []
er_n = []
er_mcmc = []

fv_st = []
fv_sst = []
fv_n = []
fv_mcmc = []

for i in range(len(model_name)):
    temp_er, temp_fv = get_error(rk, st[i]['Forecasted Variance'], f"{model_name[i]}-{distribution[0]}")
    er_st.append(temp_er)
    fv_st.append(temp_fv)

    temp_er, temp_fv = get_error(rk, 
                           sst[i]['Forecasted Variance'], 
                           f"{model_name[i]}-{distribution[1]}")
    er_sst.append(temp_er)
    fv_sst.append(temp_fv)

    temp_er, temp_fv = get_error(rk, 
                         n[i]['Forecasted Variance'], 
                         f"{model_name[i]}-{distribution[2]}")
    er_n.append(temp_er)
    fv_n.append(temp_fv)
    

er_gas = []
fv_gas = []
for col in GAS_name:
    temp_er, temp_fv = get_error(
        rk,  
        GAS_result[col],  
        col
    )
    er_gas.append(temp_er)
    fv_gas.append(temp_fv)

for i in range(3):
    temp_er, temp_fv = get_error(rk, 
                         mcmc[i]['Forecasted Variance'], 
                         f"{model_name[i]}-{'MCMC'}")
    er_mcmc.append(temp_er)
    fv_mcmc.append(temp_fv)

er_rme = []
fv_rme = []
temp_er, temp_fv = get_error(rk, rme['Forecasted Variance'], 'RiskMetrics')
er_rme.append(temp_er)
fv_rme.append(temp_fv)

all_er = pd.concat(er_st + er_sst + er_n + er_mcmc + er_gas + er_rme, axis=1)
all_fv = pd.concat(fv_st + fv_sst + fv_n + fv_mcmc + fv_gas + fv_rme, axis=1)
print(all_er)
print(all_fv)

all_fv.to_csv('all_fv.csv', index=False)
all_er.to_csv('all_er.csv', index=False)


results_DTW = []

# Calculate DTW distance for each model
for model_name, predictions in all_fv.items():
    dtw_distance = dtw.distance(rk, predictions)
    results_DTW.append({"Model": model_name, "DTW_Distance": dtw_distance})

results_DTW = pd.DataFrame(results_DTW)
# Sort results by DTW distance (smaller is better)
#results_DTW = results_DTW.sort_values(by="DTW_Distance", ascending=True)

# Display the sorted results
print(results_DTW)

all_data['Score'] += results_DTW['DTW_Distance']
top_score = all_data.nsmallest(22, "Score")
print("\nTop 22 Models by Combined Score")
print(top_score)


# Diebold-Mariano Test 
from scipy.stats import norm
from statsmodels.tsa.stattools import acovf

def diebold_mariano_test(e1, e2, metric):    
    if metric == 'MSE':
        d = e1**2 - e2**2
    elif metric == 'RMSE':
        d = np.sqrt(e1**2) - np.sqrt(e2**2)
    elif metric == 'MAE':
        d = np.abs(e1) - np.abs(e2)
    elif metric == 'MAPE':
        d = np.abs(e1 / (e1 + e2)) - np.abs(e2 / (e1 + e2)) 
    else:
        raise ValueError("Unsupported metric. Choose from 'MSE','MAPE'.")

    mean_d = np.mean(d)
    var_d = acovf(d, fft=False, adjusted=False)[0]
    n = len(d)

    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value

filtered_data = top_score[top_score['Score'] < 6.49748]
filter_er = all_er[filtered_data['Model'].values]

from itertools import combinations

metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
models_name_DM = filter_er.columns
DM_results = []
for model1, model2 in combinations(models_name_DM, 2):
    e1 = filter_er[model1].values
    e2 = filter_er[model2].values
    for metric in metrics:
        dm_stat, p_value = diebold_mariano_test(e1, e2, metric)
        DM_results.append({
            'Model 1': model1,
            'Model 2': model2,
            'Metric': metric,
            'DM Statistic': dm_stat,
            'P-Value': p_value
        })

DM_results = pd.DataFrame(DM_results)

def significance_level(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    else:
        return ''

DM_results['Significance'] = DM_results['P-Value'].apply(significance_level)
#print(DM_results)

DM_results.to_csv('DM results all.csv', index=False)