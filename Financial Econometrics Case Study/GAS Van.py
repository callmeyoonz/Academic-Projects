import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings

from scipy.optimize import minimize, basinhopping
from scipy.special import gamma, gammaln

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

data = pd.read_csv('daily_return_in.csv')
data['DATE'] = pd.to_datetime(data['DATE'])

realized_df = pd.read_csv('RK_in.csv')
realized_df.index = pd.to_datetime(realized_df['date'])
RK = realized_df.iloc[:,-2]
print(RK)
# with sns.plotting_context('paper', font_scale=1.2):
#     plt.figure(figsize=(18,6))
#     g = sns.lineplot(realized_df.iloc[:,1], color=('tab:blue'), label='Realized Kernel')
#     g = sns.lineplot(realized_df.iloc[:,2], color=('tab:red'), label='Realized Volatility')
#     plt.xlabel('Date')
#     plt.ylabel('Returns')
#     plt.title(f'Realized Kernel vs Realized Volatility')
#     plt.show()

def GAS_update(y, params, dist='gaussian', realized=False, X=None):
    omega, alpha, beta = params[:3]
    nu, xi = None, None

    if dist in ['student-t', 'GED']:
        nu = params[3]
    elif dist=='skewed student-t':
        nu, xi = params[3], params[4]

    f = [np.log(np.var(y))] + [np.nan]*(len(y))
    for t in range(len(y)):
        nabla_t = calc_nabla(y[t], f[t], dist, nu, xi, realized, X[t] if X is not None else None)
        S_t = 1
        s_t = S_t * nabla_t
        f[t+1] = omega + alpha*s_t + beta*f[t]

    return f[:-1]

def calc_nabla(y, f, dist='gaussian', nu=None, xi=None, varphi=None, realized=False, X=None):
    if dist=='gaussian':
        nabla = -0.5 + (y**2 / (2*np.exp(f)))
    elif dist=='student-t':
        nabla = -0.5 + ((nu+1)/2) * (y**2/((nu-2)*np.exp(f) + y**2))
    elif dist=='skewed student-t':
        m = (gamma((nu-1)/2) / gamma(nu/2)) * np.sqrt((nu-2)/np.pi) * (xi - 1/xi)
        s = np.sqrt(xi**2 + 1/xi**2 - 1 - m**2)
        I = np.where(s * (y/np.exp(f)**0.5) + m >= 0, 1, -1)
        z = s * y / np.sqrt(np.exp(f)) + m
        B = 1 + ((s * y / np.sqrt(np.exp(f)) + m) ** 2 / (nu - 2)) * xi ** (-2 * I)
        # nabla = -0.5 + (nu+1)/2*((s**2 * y**2) / (np.exp(f)**0.5 * (nu-2) * xi**(-2*I) * (1 + (s*y/np.exp(f)**0.5 + m)**2/((nu-2)*xi**(-2*I)))))
        # nabla = -0.5 + ((nu + 1) * z * s * y * xi ** (-2 * I)) / (2 * B * (nu - 2) * np.exp(f)**0.5)
        nabla = -0.5 + (nu+1)/2 * z * s * (y / np.sqrt(np.exp(f))) / ( (nu-2)* xi**(2*I) + z**2 )
    elif dist=='GED':
        _lambda = (gamma(1/nu)/(2**(2/nu) * gamma(3/nu)))
        nabla = -0.5 + (nu/4) * ((np.abs(y)**(nu-1) * y)/(_lambda**nu * np.exp(f)**(nu/2)))

    return nabla

def calc_LogL(y, f, params, dist='gaussian', realized=False, X=None):
    if dist=='gaussian':
        LogL = -0.5 * np.log(2*np.pi*np.exp(f)) - (y**2/np.exp(f))
    elif dist=='student-t':
        nu = params[3]
        LogL = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log((nu-2)*np.pi*np.exp(f)) - ((nu+1)/2)*np.log(1 + y**2/((nu-2)*np.exp(f)))
    elif dist=='skewed student-t':
        nu, xi = params[3], params[4]
        m = (gamma((nu-1)/2) / gamma(nu/2)) * np.sqrt((nu-2)/np.pi) * (xi - 1/xi)
        s = np.sqrt(xi**2 + 1/xi**2 - 1 - m**2)
        I = np.where(s * (y/np.exp(f)**0.5) + m >= 0, 1, -1)
        LogL = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log((nu-2)*np.pi*np.exp(f)) \
              + np.log(s) + np.log(2/(xi+1/xi)) \
              - ((nu+1)/2)*np.log(1 + (s*y/np.exp(f)**0.5 + m)**2 / (nu-2) * xi**(-2*I))
    elif dist=='GED':
        nu = params[3]
        _lambda = (gamma(1/nu)/(2**(2/nu) * gamma(3/nu)))
        LogL = -np.log(2**(1+(1/nu)) * gamma(1/nu) * _lambda) - 0.5*np.log(np.exp(f)) + np.log(nu) - 0.5*np.abs(y/(_lambda * np.exp(f)**0.5))**nu

    L2 = 1*np.sum(np.exp(f)**2)

    return np.sum(LogL) - L2

def GAS_optimize(y, dist='gaussian', realized=False, params0=[0, .1, .9, 5, 1], LB=[-10, -10, .1, 2.1, 0.2], UB=[10, 10, .95, 20, 5]):
    if dist == 'gaussian':
        params0, LB, UB = params0[:3], LB[:3], UB[:3]
    elif dist in ['student-t', 'GED']:
        params0, LB, UB = params0[:4], LB[:4], UB[:4]
    elif dist == 'skewed student-t':
        params0, LB, UB = params0[:5], LB[:5], UB[:5]
    def loss_func(params, y, dist):
        return -calc_LogL(y, GAS_update(y, params, dist), params, dist)

    result = basinhopping(loss_func, x0=params0, niter=10, seed=np.random.default_rng(1234),
                          minimizer_kwargs={'args':(y, dist), 'bounds':list(zip(LB, UB)), 'method':'L-BFGS-B', 'options':{'gtol':1e-5}})

    return result


df_GAS_params = pd.DataFrame(columns=['gaussian', 'student-t', 'skewed student-t', 'GED'], index=['omega', 'alpha', 'beta', 'nu', 'xi', 'LogL', 'AIC', 'BIC'])
for dist in ['gaussian', 'student-t', 'skewed student-t', 'GED']:
    result = GAS_optimize(data.iloc[:, -1], dist=dist)
    GAS_params = result.x

    y_pred = np.exp(GAS_update(data.iloc[:, -1], GAS_params, dist=dist))
    y_true = data.iloc[:, -1].values

    LogL = calc_LogL(y_true, y_pred, GAS_params)
    AIC = -2*LogL + 2*len(GAS_params)
    BIC = len(GAS_params) * np.log(len(data)) - 2*LogL

    hessian_inv = result.lowest_optimization_result.hess_inv.todense()  # Inverse of the Hessian
    se = np.sqrt(np.diag(hessian_inv))

    df_GAS_params.loc[:, dist][:len(GAS_params)] = GAS_params
    df_GAS_params.loc[['LogL', 'AIC', 'BIC'], dist] = [LogL, AIC, BIC]

print(df_GAS_params)

realized_df = pd.read_csv('RK_in.csv')
realized_df.index = pd.to_datetime(realized_df['date'])
RK = realized_df.iloc[:,-2]

def calc_nabla_R(y, f, dist, nu, xi, varphi, realized, X):
    # if X is None:
    #     print('in calc_nable_R, X is None.')

    if dist=='gaussian':
        nabla = -0.5 + (y**2 / (2*np.exp(f)))
    elif dist=='student-t':
        nabla = -0.5 + ((nu+1)/2) * (y**2/((nu-2)*np.exp(f) + y**2))
    elif dist=='skewed student-t':
        m = (gamma((nu-1)/2) / gamma(nu/2)) * np.sqrt((nu-2)/np.pi) * (xi - 1/xi)
        s = np.sqrt(xi**2 + 1/xi**2 - 1 - m**2)
        I = np.where(s * (y/np.exp(f)**0.5) + m >= 0, 1, -1)
        z = s * y / np.sqrt(np.exp(f)) + m
        B = 1 + ((s * y / np.sqrt(np.exp(f)) + m) ** 2 / (nu - 2)) * xi ** (-2 * I)

        # nabla = -0.5 + (nu+1)/2*((s**2 * y**2) / (np.exp(f)**0.5 * (nu-2) * xi**(-2*I) * (1 + (s*y/np.exp(f)**0.5 + m)**2/((nu-2)*xi**(-2*I)))))
        # nabla = -0.5 + ((nu + 1) * z * s * y * xi ** (-2 * I)) / (2 * B * (nu - 2) * np.exp(f)**0.5)
        nabla = -0.5 + (nu+1)/2 * z * s * (y / np.sqrt(np.exp(f))) / ( (nu-2)* xi**(2*I) + z**2 )

    elif dist=='GED':
        _lambda = (gamma(1/nu)/(2**(2/nu) * gamma(3/nu)))
        nabla = -0.5 + (nu/4) * ((np.abs(y)**(nu-1) * y)/(_lambda**nu * np.exp(f)**(nu/2)))

    if realized and X is not None:
        part_nabla = nabla
        RK_nabla = (varphi/2) * (X/np.exp(f)-1)
        nabla = part_nabla + RK_nabla
    return nabla

def RGAS_update(y, params, dist='gaussian', realized = True, X=RK):
    if X is None:
        print('in RGAS_update, X is None.')

    omega, alpha, beta, varphi = params[:4]
    nu, xi = None, None

    if dist in ['student-t', 'GED']:
        nu = params[4]
    elif dist=='skewed student-t':
        nu, xi = params[4], params[5]

    f = [np.log(np.var(y))] + [np.nan]*(len(y))
    for t in range(len(y)):
        nabla_t = calc_nabla_R(y[t], f[t], dist, nu, xi, varphi, realized, X[t])
        S_t = 1
        s_t = S_t * nabla_t
        f[t+1] = omega + alpha*s_t + beta*f[t]

    return f[:-1]


def calc_LogL_R(y, f, params, dist='gaussian', realized=True, X=RK):
    if X is None:
        print('in calc_LogL_R, X is None.')

    if dist=='gaussian':
        LogL = -0.5 * np.log(2*np.pi*np.exp(f)) - (y**2/np.exp(f))
    elif dist=='student-t':
        nu = params[4]
        LogL = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log((nu-2)*np.pi*np.exp(f)) - ((nu+1)/2)*np.log(1 + y**2/((nu-2)*np.exp(f)))
    elif dist=='skewed student-t':
        nu, xi = params[4], params[5]
        m = (gamma((nu-1)/2) / gamma(nu/2)) * np.sqrt((nu-2)/np.pi) * (xi - 1/xi)
        s = np.sqrt(xi**2 + 1/xi**2 - 1 - m**2)
        I = np.where(s * (y/np.exp(f)**0.5) + m >= 0, 1, -1)
        LogL = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log((nu-2)*np.pi*np.exp(f)) \
              + np.log(s) + np.log(2/(xi+1/xi)) \
              - ((nu+1)/2)*np.log(1 + (s*y/np.exp(f)**0.5 + m)**2 / (nu-2) * xi**(-2*I))
    elif dist=='GED':
        nu = params[4]
        _lambda = (gamma(1/nu)/(2**(2/nu) * gamma(3/nu)))
        LogL = -np.log(2**(1+(1/nu)) * gamma(1/nu) * _lambda) - 0.5*np.log(np.exp(f)) + np.log(nu) - 0.5*np.abs(y/(_lambda * np.exp(f)**0.5))**nu

    if realized and X is not None:
        varphi = params[3]
        part_ll = LogL
        u_ll = - gammaln(varphi/2) - (varphi/2)* np.log(2*np.exp(f)/varphi) + (varphi/2 - 1)*np.log(X) - (varphi*X/2*np.exp(f))
        ll = part_ll + u_ll

    L2 = 1*np.sum(np.exp(f)**2)

    return np.sum(LogL) - L2

#testing RGAS
def RGAS_optimize(y, dist='gaussian', params0=[0, .1, .9, 10, 5, 1], LB=[-10, .0001, .0001, 0.1, 2.1, 0.2], UB=[10, .95, .95, 50, 20, 5], realized=True, X=RK):
    if dist=='gaussian':
        params0, LB, UB = params0[:4], LB[:4], UB[:4]
    elif dist in ['student-t', 'GED']:
        params0, LB, UB = params0[:5], LB[:5], UB[:5]
    elif dist == 'skewed student-t':
        params0, LB, UB = params0[:6], LB[:6], UB[:6]
    def loss_func(params, y, dist):
        return -calc_LogL_R(y, RGAS_update(y, params, dist), params, dist, X=X)

    result = basinhopping(loss_func, x0=params0, niter=10, seed=np.random.default_rng(1234),
                          minimizer_kwargs={'args':(y, dist), 'bounds':list(zip(LB, UB)), 'method':'L-BFGS-B', 'options':{'gtol':1e-5}})

    return result

df_RGAS_params = pd.DataFrame(columns=['gaussian', 'student-t', 'skewed student-t', 'GED'], index=['omega', 'alpha', 'beta', 'varphi', 'nu', 'xi', 'LogL', 'AIC', 'BIC'])
for dist in ['gaussian', 'student-t', 'skewed student-t', 'GED']:
    result = RGAS_optimize(data.iloc[:, -1], dist=dist, realized=True, X= RK)
    RGAS_params = result.x
    hessian_inv = result.lowest_optimization_result.hess_inv.todense()  # Inverse of the Hessian
    se = np.sqrt(np.diag(hessian_inv))
    # # se = np.sqrt(np.diag(inv(-hessian)))
    # print(se)
    LogL = -result.fun
    AIC = -2*LogL + 2*len(RGAS_params)
    BIC = len(RGAS_params) * np.log(len(data)) - 2*LogL

    df_RGAS_params.loc[:, dist][:len(RGAS_params)] = RGAS_params
    df_RGAS_params.loc[['LogL', 'AIC', 'BIC'], dist] = [LogL, AIC, BIC]

print(df_RGAS_params)

dict_GAS_plot = {}
df_GAS_result = pd.DataFrame(columns=['gaussian', 'student-t', 'skewed student-t', 'GED'], index=['RMSE', 'MAPE', 'DM'])
for dist in ['gaussian', 'student-t', 'skewed student-t', 'GED']:
    params = df_GAS_params[dist][:5].values
    y_pred = np.exp(GAS_update(data.iloc[:, -1], params, dist=dist))
    y_true = data.iloc[:, -1].values

    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    dm = 0

    df_GAS_result.loc[['RMSE', 'MAPE', 'DM'], dist] = [rmse, mape, dm]
    dict_GAS_plot[dist] = pd.DataFrame({'true': y_true, 'pred': y_pred}, index=data['DATE'])

dict_RGAS_plot = {}
df_RGAS_result = pd.DataFrame(columns=['gaussian', 'student-t', 'skewed student-t', 'GED'], index=['RMSE', 'MAPE', 'DM'])
for dist in ['gaussian', 'student-t', 'skewed student-t', 'GED']:
    params = df_RGAS_params[dist][:6].values
    y_pred = np.exp(RGAS_update(data.iloc[:, -1], params, dist=dist))
    y_true = (data.iloc[:, -1]).values

    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    dm = 0

    df_RGAS_result.loc[['RMSE', 'MAPE', 'DM'], dist] = [rmse, mape, dm]
    dict_RGAS_plot[dist] = pd.DataFrame({'true': y_true, 'pred': y_pred}, index=data['DATE'])

test = pd.read_csv('daily_return_out.csv')
test['DATE'] = pd.to_datetime(test['DATE'])

dict_GAS_plot = {}
df_GAS_result = pd.DataFrame(columns=['gaussian', 'student-t', 'skewed student-t', 'GED'], index=['RMSE', 'MAPE', 'MAE', 'MSE', 'DM'])
for dist in ['gaussian', 'student-t', 'skewed student-t', 'GED']:
    params = df_GAS_params[dist][:5].values
    y_pred = np.exp(GAS_update(test.iloc[:, -1], params, dist=dist))
    y_true = test.iloc[:, -1].values

    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    dm = 0

    df_GAS_result.loc[['RMSE', 'MAPE', 'MAE', 'MSE', 'DM'], dist] = [rmse, mape, mae, mse, dm]
    dict_GAS_plot[dist] = pd.DataFrame({'true': y_true, 'pred': y_pred}, index=test['DATE'])

dict_RGAS_plot = {}
df_RGAS_result = pd.DataFrame(columns=['gaussian', 'student-t', 'skewed student-t', 'GED'], index=['RMSE', 'MAPE', 'MAE', 'MSE', 'DM'])
for dist in ['gaussian', 'student-t', 'skewed student-t', 'GED']:
    params = df_RGAS_params[dist][:6].values
    y_pred = np.exp(RGAS_update(test.iloc[:, -1], params, dist=dist))
    y_true = test.iloc[:, -1].values

    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    dm = 0

    df_RGAS_result.loc[['RMSE', 'MAPE', 'MAE', 'MSE', 'DM'], dist] = [rmse, mape, mae, mse, dm]
    dict_RGAS_plot[dist] = pd.DataFrame({'true': y_true, 'pred': y_pred}, index=test['DATE'])

# Create an empty DataFrame to store all predictions
combined_df = pd.DataFrame()

# Add the true values
combined_df['DATE'] = dict_GAS_plot['gaussian'].index  # Use the DATE index from one model
combined_df['True'] = dict_GAS_plot['gaussian']['true'].values  # True values from one model

# Add predicted series for each model in dict_GAS_plot
for dist, df in dict_GAS_plot.items():
    combined_df[f"GAS_{dist}_Pred"] = df['pred'].values

# Add predicted series for each model in dict_RGAS_plot
for dist, df in dict_RGAS_plot.items():
    combined_df[f"RGAS_{dist}_Pred"] = df['pred'].values

# Save the combined DataFrame to a single CSV file
combined_df.to_csv("GAS_forecasts.csv", index=False)