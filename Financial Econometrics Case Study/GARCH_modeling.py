#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, gamma
from numpy.linalg import inv
from scipy.stats import norm


def aic_bic(log_likelihood, num_params, num_observations):
    aic = -2 * log_likelihood + 2 * num_params
    bic = -2 * log_likelihood + num_params * np.log(num_observations)
    return {'AIC': aic, 'BIC': bic}

def compute_variance_garch(vR, vTheta, gjr=False):
    dOmega, dAlpha, dBeta = vTheta[:3]
    dGamma = vTheta[3] if gjr else 0
    iT = len(vR)
    vH = np.ones(iT) * np.var(vR)
    for t in range(1, iT):
        leverage_effect = dGamma * vR[t - 1] ** 2 if (gjr and vR[t - 1] < 0) else 0
        vH[t] = dOmega + dAlpha * vR[t - 1] ** 2 + leverage_effect + dBeta * vH[t - 1]
    return vH

def compute_variance_egarch(vR, vTheta):
    dOmega, dAlpha, dBeta, dNu, dGamma = vTheta
    iT = len(vR)
    vH = np.ones(iT) * np.var(vR)
    expected_abs_eta = np.sqrt(dNu) * gamma((dNu - 1) / 2) / (np.sqrt(np.pi) * gamma(dNu / 2))
    for t in range(1, iT):
        eta_t_1 = vR[t - 1] / np.sqrt(vH[t - 1])
        vH[t] = vH[t - 1] ** dBeta * np.exp(dOmega + dAlpha * (np.abs(eta_t_1) - expected_abs_eta) + dGamma * eta_t_1)
    return vH

def log_likelihood_student_t(vR, vH, dNu):
    return np.sum(gammaln((dNu + 1) / 2) - gammaln(dNu / 2) - 
                  0.5 * np.log((dNu - 2) * np.pi * vH) - 0.5 * (dNu + 1) * np.log(1 + (vR ** 2) / ((dNu - 2) * vH)))

def nll_garch_student_t(vTheta, vR):
    dNu = vTheta[-1]
    vH = compute_variance_garch(vR, vTheta)
    return -log_likelihood_student_t(vR, vH, dNu)

def nll_gjr_student_t(vTheta, vR):
    dNu = vTheta[-1]
    vH = compute_variance_garch(vR, vTheta, gjr=True)
    return -log_likelihood_student_t(vR, vH, dNu)

def nll_egarch_student_t(vTheta, vR):
    dNu = vTheta[3]
    vH = compute_variance_egarch(vR, vTheta)
    return -log_likelihood_student_t(vR, vH, dNu)

def compute_gradients(vTheta,vR, model="GARCH"):
    iT = len(vR)
    grad_matrix = np.zeros((iT, len(vTheta)))
    if model=="GARCH":
        dOmega, dAlpha, dBeta, dNu = vTheta
        vH = compute_variance_garch(vR, vTheta[:3])
        for t in range(1, iT):
            e_t = vR[t]
            h_t = vH[t]
            dlog_p_dOmega = 0.5 * (1 - (1 + (e_t**2) / ((dNu - 2) * h_t)) ** (-1) * e_t**2 / ((dNu - 2) * h_t**2))
            dlog_p_dAlpha = dlog_p_dOmega * vR[t - 1] ** 2
            dlog_p_dBeta = dlog_p_dOmega * vH[t - 1]
            dlog_p_dNu = 0.5 * (1 - (1 + (e_t**2) / ((dNu - 2) * h_t)) ** (-1) * (e_t**2 / ((dNu - 2) ** 2 * h_t)) - 1 / (dNu - 2))
            grad_matrix[t] = [dlog_p_dOmega, dlog_p_dAlpha, dlog_p_dBeta, dlog_p_dNu]
    elif model=="GJR-GARCH":
        dOmega, dAlpha, dBeta, dGamma, dNu = vTheta
        vH = compute_variance_garch(vR, vTheta[:4], gjr=True)
        for t in range(1, iT):
            e_t = vR[t]
            h_t = vH[t]
            dlog_p_dOmega = 0.5 * (1 - (1 + (e_t**2) / ((dNu - 2) * h_t)) ** (-1) * e_t**2 / ((dNu - 2) * h_t**2))
            dlog_p_dAlpha = dlog_p_dOmega * vR[t - 1] ** 2
            dlog_p_dBeta = dlog_p_dOmega * vH[t - 1]
            dlog_p_dGamma = dlog_p_dOmega * (vR[t - 1] ** 2 if vR[t - 1] < 0 else 0)
            dlog_p_dNu = 0.5 * (1 - (1 + (e_t**2) / ((dNu - 2) * h_t)) ** (-1) * (e_t**2 / ((dNu - 2) ** 2 * h_t)) - 1 / (dNu - 2))
            grad_matrix[t] = [dlog_p_dOmega, dlog_p_dAlpha, dlog_p_dBeta, dlog_p_dGamma, dlog_p_dNu]
    elif model == "EGARCH":
        dOmega, dAlpha, dBeta, dNu, dGamma = vTheta
        vH = compute_variance_egarch(vR, vTheta)
        expected_abs_z = 2 * np.sqrt((dNu-2)/np.pi) * gamma((dNu+1)/2) / gamma(dNu/2)
        for t in range(1, iT):
            eta_t_1 = vR[t - 1] / np.sqrt(vH[t - 1])
            e_t = vR[t]
            h_t = vH[t]
            log_h_t = np.log(h_t)
            dlog_p_dOmega = 0.5 * (1 - (1 + (e_t**2) / ((dNu - 2) * h_t)) ** (-1) * e_t**2 / ((dNu - 2) * h_t**2))
            dlog_p_dAlpha = dlog_p_dOmega * (np.abs(eta_t_1) - expected_abs_z)
            dlog_p_dBeta = dlog_p_dOmega * log_h_t
            dlog_p_dGamma = dlog_p_dOmega * eta_t_1
            dlog_p_dNu = 0.5 * (1 - (1 + (e_t**2) / ((dNu - 2) * h_t)) ** (-1) * (e_t**2 / ((dNu - 2) ** 2 * h_t)) - 1 / (dNu - 2))
            grad_matrix[t] = [dlog_p_dOmega, dlog_p_dAlpha, dlog_p_dBeta, dlog_p_dGamma, dlog_p_dNu]
    return grad_matrix[1:]

def compute_standard_errors(vTheta,vR,model):
    grad_matrix = compute_gradients(vTheta,vR, model)
    G = grad_matrix.T
    T = len(vR)
    mCovariance_OPG = inv(G @ G.T)
    return np.sqrt(np.diag(mCovariance_OPG))

def optimize_model(vR, nll_function, vTheta_ini, vTheta_bnds):
    result = minimize(nll_function, vTheta_ini, args=(vR,), method="L-BFGS-B", bounds=vTheta_bnds)
    return result.x, -result.fun

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    return data['daily_return'].values * 100

def main(file_path, model="GARCH"):
    vR = load_data(file_path)
    num_observations = len(vR)
    if model == "EGARCH":
        vTheta_ini = [0.1, 0.2498, 0.9505, 4.3648, 0.1]
        vTheta_bnds = [(0, 1), (-1, 1), (0.7, 1), (2.1, 10), (-1, 1)]
        nll_function = nll_egarch_student_t
        param_names = ["Omega", "Alpha", "Beta", "Nu", "Gamma"]
        num_params = len(param_names)
    elif model == "GARCH":
        vTheta_ini = [0.1, 0.05, 0.90, 6]
        vTheta_bnds = [(0, 1), (0, 1), (0.7, 0.999), (2.1, 30)]
        nll_function = nll_garch_student_t
        param_names = ["Omega", "Alpha", "Beta", "Nu"]
        num_params = len(param_names)
    elif model == "GJR-GARCH":
        vTheta_ini = [0.1, 0.05, 0.90, 0.1, 12]
        vTheta_bnds = [(1e-6, 1), (1e-6, 1), (0.7, 0.999), (0, 1), (2.1, 30)]
        nll_function = nll_gjr_student_t
        param_names = ["Omega", "Alpha", "Beta", "Gamma", "Nu"]
        num_params = len(param_names)
    else:
        raise ValueError("Unsupported model type!")
    vTheta_ML, log_likelihood = optimize_model(vR, nll_function, vTheta_ini, vTheta_bnds)
    standard_errors = compute_standard_errors(vTheta_ML,vR,model)
    
    print(f"\nModel: {model}")
    print("Log-likelihood: ", log_likelihood)
    info_criteria = aic_bic(log_likelihood, num_params, num_observations)
    print(f"AIC: {info_criteria['AIC']:.4f}")
    print(f"BIC: {info_criteria['BIC']:.4f}")
    print("Parameter Estimates:")
    print(f"{'Parameter':<10}{'Estimate':<15}{'Std. Error':<15}")
    for name, est, se in zip(param_names, vTheta_ML, standard_errors):
        print(f"{name:<10}{est:<15.6f}{se:<15.6f}")

if __name__ == "__main__":
    file_path = r"C:\Users\rwe20\OneDrive\Desktop\FECS\daily_return_in.csv"
    print("Student-t")
    main(file_path, model="GARCH")
    main(file_path, model="GJR-GARCH")
    main(file_path, model="EGARCH")


# In[2]:


def compute_variance_egarch_skew(vR, vTheta):
    dOmega, dAlpha, dBeta, dNu, dGamma, dSkew = vTheta
    iT = len(vR)
    vH = np.ones(iT) * np.var(vR)
    term = (4 * dSkew**2) / (dSkew**2 + 1 / dSkew**2)
    expected_abs_eta = term * np.sqrt((dNu - 2) / np.pi) * (gamma((1 + dNu) / 2) / gamma(dNu / 2))
    for t in range(1, iT):
        eta_t_1 = vR[t - 1] / np.sqrt(max(vH[t - 1], 1e-8))
        vH[t] = vH[t - 1] ** dBeta * np.exp(dOmega + dAlpha * (np.abs(eta_t_1) - expected_abs_eta) + dGamma * eta_t_1)

    return vH

def compute_gradient_skewt(vTheta, vR, model="GARCH"):
    iT = len(vR)
    grad_matrix = np.zeros((iT, len(vTheta)))
    if model == "GARCH":
        dOmega, dAlpha, dBeta, dNu, dX = vTheta
        vH = compute_variance_garch(vR, vTheta[:-2])
        m = gamma((dNu - 1) / 2) / gamma(dNu / 2) * np.sqrt((dNu - 2) / np.pi) * (dX - 1/dX)
        s = np.sqrt((dX**2 + 1/dX**2 - 1) - m**2)
        for t in range(1, iT):
            e_t, h_t = vR[t], vH[t]
            z_t = s * e_t / np.sqrt(h_t) + m
            I_t = 1 if z_t >= 0 else -1
            k_t = 1 + z_t**2 / ((dNu - 2) * dX**(-2 * I_t))
            common_factor = -0.5/h_t * (1 - (dNu + 1)/((dNu - 2)) * z_t**2/k_t)
            dlog_p_dOmega = common_factor
            dlog_p_dAlpha = common_factor * vR[t-1]**2
            dlog_p_dBeta = common_factor * vH[t-1]
            dlog_p_dNu = 0.5 * (gamma((dNu + 1)/2) - gamma(dNu/2) - 1/(dNu - 2) - np.log(k_t) + (dNu + 1) * z_t**2 / ((dNu - 2)**2 * k_t * dX**(-2 * I_t)))
            dm_ddX = gamma((dNu - 1)/2)/(gamma(dNu/2) * np.sqrt((dNu - 2) * np.pi)) * (1 + 1/dX**2)
            ds_ddX = (2 * dX - 2/(dX**3) - 2 * m * dm_ddX)/(2 * s)
            dz_t_ddX = ds_ddX * e_t/np.sqrt(h_t) + dm_ddX
            dlog_p_ddX = (1/s * ds_ddX - 2/(dX + 1/dX) * (1 - 1/dX**2)/(dX + 1/dX) - (dNu + 1)/(2 * k_t) * (2 * z_t * dz_t_ddX/((dNu - 2) * dX**(-2 * I_t)) + 2 * I_t * z_t**2/((dNu - 2) * dX**(-2 * I_t + 1))))
            grad_matrix[t] = [dlog_p_dOmega,dlog_p_dAlpha,dlog_p_dBeta,dlog_p_dNu,dlog_p_ddX]
    elif model == 'GJR-GARCH':
        dOmega, dAlpha, dBeta, dGamma, dNu, dX = vTheta
        vH = compute_variance_garch(vR, vTheta[:-2], gjr=True)
        m = gamma((dNu - 1) / 2) / gamma(dNu / 2) * np.sqrt((dNu - 2) / np.pi) * (dX - 1/dX)
        s = np.sqrt((dX**2 + 1/dX**2 - 1) - m**2)
        for t in range(1, iT):
            e_t, h_t = vR[t], vH[t]
            z_t = s * e_t / np.sqrt(h_t) + m
            I_t = 1 if z_t >= 0 else -1
            k_t = 1 + z_t**2 / ((dNu - 2) * dX**(-2 * I_t))
            common_factor = -0.5/h_t * (1 - (dNu + 1)/((dNu - 2)) * z_t**2/k_t)
            dlog_p_dOmega = common_factor
            dlog_p_dAlpha = common_factor * vR[t-1]**2
            dlog_p_dBeta = common_factor * vH[t-1]
            dlog_p_dGamma = common_factor * (vR[t-1]**2 if vR[t-1] < 0 else 0)
            dlog_p_dNu = 0.5 * (gamma((dNu + 1)/2) - gamma(dNu/2) - 1/(dNu - 2) - np.log(k_t) + (dNu + 1) * z_t**2 / ((dNu - 2)**2 * k_t * dX**(-2 * I_t)))
            dm_ddX = gamma((dNu - 1)/2)/(gamma(dNu/2) * np.sqrt((dNu - 2) * np.pi)) * (1 + 1/dX**2)
            ds_ddX = (2 * dX - 2/(dX**3) - 2 * m * dm_ddX)/(2 * s)
            dz_t_ddX = ds_ddX * e_t/np.sqrt(h_t) + dm_ddX
            dlog_p_ddX = (1/s * ds_ddX - 2/(dX + 1/dX) * (1 - 1/dX**2)/(dX + 1/dX) - (dNu + 1)/(2 * k_t) * (2 * z_t * dz_t_ddX/((dNu - 2) * dX**(-2 * I_t)) + 2 * I_t * z_t**2/((dNu - 2) * dX**(-2 * I_t + 1))))
            grad_matrix[t] = [dlog_p_dOmega,dlog_p_dAlpha,dlog_p_dBeta,dlog_p_dGamma,dlog_p_dNu,dlog_p_ddX]
    elif model == "EGARCH":
        dOmega, dAlpha, dBeta, dNu, dGamma, dX = vTheta
        dSkew = dX
        vH = compute_variance_egarch_skew(vR, vTheta)
        term = (4 * dSkew**2) / (dSkew**2 + 1 / dSkew**2)
        expected_abs_eta = term * np.sqrt((dNu - 2) / np.pi) * (gamma((1 + dNu) / 2) / gamma(dNu / 2))
        m = gamma((dNu - 1) / 2) / gamma(dNu / 2) * np.sqrt((dNu - 2) / np.pi) * (dX - 1/dX)
        s = np.sqrt((dX**2 + 1/dX**2 - 1) - m**2)
        for t in range(1, iT):
            e_t, h_t = vR[t], vH[t]
            z_t = s * e_t / np.sqrt(h_t) + m
            I_t = 1 if z_t >= 0 else -1
            eta_t_1 = vR[t-1] / np.sqrt(vH[t-1])
            log_h_t = np.log(h_t)
            k_t = 1 + z_t**2 / ((dNu - 2) * dX**(-2 * I_t))
            common_factor = -0.5/h_t * (1 - (dNu + 1)/((dNu - 2)) * z_t**2/k_t)
            dlog_p_dOmega = common_factor * h_t
            dlog_p_dAlpha = common_factor * h_t * (np.abs(eta_t_1) - expected_abs_eta)
            dlog_p_dBeta = common_factor * h_t * log_h_t
            dlog_p_dGamma = common_factor * h_t * eta_t_1
            dlog_p_dNu = 0.5 * (gamma((dNu + 1)/2) - gamma(dNu/2) - 1/(dNu - 2) - np.log(k_t) + (dNu + 1) * z_t**2 / ((dNu - 2)**2 * k_t * dX**(-2 * I_t)))
            dm_ddX = gamma((dNu - 1)/2)/(gamma(dNu/2) * np.sqrt((dNu - 2) * np.pi)) * (1 + 1/dX**2)
            ds_ddX = (2 * dX - 2/(dX**3) - 2 * m * dm_ddX)/(2 * s)
            dz_t_ddX = ds_ddX * e_t/np.sqrt(h_t) + dm_ddX
            dlog_p_ddX = (1/s * ds_ddX - 2/(dX + 1/dX) * (1 - 1/dX**2)/(dX + 1/dX) - (dNu + 1)/(2 * k_t) * (2 * z_t * dz_t_ddX/((dNu - 2) * dX**(-2 * I_t)) + 2 * I_t * z_t**2/((dNu - 2) * dX**(-2 * I_t + 1))))
            grad_matrix[t] = [dlog_p_dOmega,dlog_p_dAlpha,dlog_p_dBeta,dlog_p_dNu,dlog_p_dGamma,dlog_p_ddX]
    return grad_matrix[1:]

def compute_standard_errors_skew(vTheta, vR, model="GARCH"):
    grad_matrix = compute_gradient_skewt(vTheta, vR, model)
    G = grad_matrix.T
    mCovariance_OPG = inv(G @ G.T)
    return np.sqrt(np.diag(mCovariance_OPG))

def log_likelihood_skewed_t(vR, vH, dNu, dX):
    iT = len(vR)
    m = np.exp(gammaln((dNu - 1) / 2) - gammaln(dNu / 2)) * np.sqrt((dNu - 2) / np.pi) * (dX - 1 / dX)
    s = np.sqrt((dX**2 + 1 / dX**2 - 1) - m**2)
    I_t = np.where(s * vR / np.sqrt(vH) + m >= 0, 1, -1)
    vLogPdf = (gammaln((dNu + 1) / 2) - gammaln(dNu / 2) - 0.5 * np.log((dNu - 2) * np.pi * vH) + np.log(s) + np.log(2 / (dX + 1 / dX)) 
               - 0.5 * (dNu + 1) * np.log(1 + (s * (vR / np.sqrt(vH)) + m) ** 2 / ((dNu - 2) * dX ** (-2 * I_t))))
    return np.sum(vLogPdf)

def nll_garch_skewed_t(vTheta, vR):
    dNu, dX = vTheta[-2], vTheta[-1]
    vH = compute_variance_garch(vR, vTheta[:-2])
    return -log_likelihood_skewed_t(vR, vH, dNu, dX)

def nll_gjr_skewed_t(vTheta, vR):
    dNu, dX = vTheta[-2], vTheta[-1]
    vH = compute_variance_garch(vR, vTheta[:-2], gjr=True)
    return -log_likelihood_skewed_t(vR, vH, dNu, dX)

def nll_egarch_skewed_t(vTheta, vR):
    dNu, dX = vTheta[-3], vTheta[-1]
    vH = compute_variance_egarch_skew(vR, vTheta)
    return -log_likelihood_skewed_t(vR, vH, dNu, dX)

def main(file_path, model="GARCH"):
    vR = load_data(file_path)
    num_observations = len(vR)

    if model == "GARCH":
        vTheta_ini = [0.01, 0.2, 0.90, 6, 1.1]  # Omega, Alpha, Beta, Nu, Skewness
        vTheta_bnds = [(1e-6, 1), (1e-6, 1), (0, 0.999), (2.1, None), (0.2, 5)]
        nll_function = nll_garch_skewed_t
        param_names = ["Omega", "Alpha", "Beta", "Nu", "Skewness"]
        num_params = len(param_names)
    elif model == "GJR-GARCH":
        vTheta_ini = [0.0001, 0.2, 0.40, 0.2, 5.2, 0.1]  # Omega, Alpha, Beta, Gamma, Nu, Skewness
        vTheta_bnds = [(1e-6, 1), (1e-6, 1), (0, 0.999), (0, 1), (3, None), (1e-6, 5)]
        nll_function = nll_gjr_skewed_t
        param_names = ["Omega", "Alpha", "Beta", "Gamma", "Nu", "Skewness"]
        num_params = len(param_names)
    elif model == "EGARCH":
        vTheta_ini = [0.1, 0.2, 0.85, 8, 0.1, 1.5]  # Omega, Alpha, Beta, Nu, Gamma, Skewness
        vTheta_bnds = [(-0.1, 1), (1e-6, 1), (0, 0.999), (3.5, 10), (-1, 1), (0.5, 5)]
        nll_function = nll_egarch_skewed_t
        param_names = ["Omega", "Alpha", "Beta", "Nu", "Gamma", "Skewness"]
        num_params = len(param_names)
    else:
        raise ValueError("Unsupported model type!")

    vTheta_ML, log_likelihood = optimize_model(vR, nll_function, vTheta_ini, vTheta_bnds)
    standard_errors = compute_standard_errors_skew(vTheta_ML,vR, model)
    print("Skewed Student-t")

    print(f"\nModel: {model}")
    print("Log-likelihood: ", log_likelihood)
    info_criteria = aic_bic(log_likelihood, num_params, num_observations)
    print(f"AIC: {info_criteria['AIC']:.4f}")
    print(f"BIC: {info_criteria['BIC']:.4f}")
    print("Parameter Estimates:")
    print(f"{'Parameter':<10}{'Estimate':<15}{'Std. Error':<15}")
    for name, est, se in zip(param_names, vTheta_ML, standard_errors):
        print(f"{name:<10}{est:<15.6f}{se:<15.6f}")

if __name__ == "__main__":
    file_path = r"C:\Users\rwe20\OneDrive\Desktop\FECS\daily_return_in.csv"
    main(file_path, model="GARCH")
    main(file_path, model="GJR-GARCH")
    main(file_path, model="EGARCH")


# In[37]:


def compute_variance_egarch_normal(vR, vTheta):
    dOmega, dAlpha, dBeta, dGamma = vTheta
    iT = len(vR)
    vH = np.ones(iT) * np.var(vR)
    log_vH = np.log(vH)
    for t in range(1, iT):
        eta_t_1 = vR[t - 1] / np.sqrt(np.exp(log_vH[t - 1]))
        log_vH[t] = dOmega + dAlpha * (np.abs(eta_t_1) - np.sqrt(2 / np.pi)) + dGamma * eta_t_1 + dBeta * log_vH[t - 1]
        vH[t] = np.exp(log_vH[t])
    return vH

def compute_gradient_normal(vTheta, vR, model):
    iT = len(vR)
    grad_matrix = np.zeros((iT, len(vTheta)))
    if model == 'GARCH':
        dOmega, dAlpha, dBeta = vTheta
        vH = compute_variance_garch(vR, vTheta)
        for t in range(1, iT):
            e_t, h_t = vR[t], vH[t]
            common_factor = -0.5/h_t * (1 - e_t**2/h_t)
            dlog_p_dOmega = common_factor
            dlog_p_dAlpha = common_factor * vR[t-1]**2
            dlog_p_dBeta = common_factor * vH[t-1]
            grad_matrix[t] = [dlog_p_dOmega, dlog_p_dAlpha, dlog_p_dBeta]
    elif model == "GJR-GARCH":
        dOmega, dAlpha, dBeta, dGamma = vTheta
        vH = compute_variance_garch(vR, vTheta, gjr=True)
        for t in range(1, iT):
            e_t, h_t = vR[t], vH[t]
            common_factor = -0.5/h_t * (1 - e_t**2/h_t)
            dlog_p_dOmega = common_factor
            dlog_p_dAlpha = common_factor * vR[t-1]**2
            dlog_p_dBeta = common_factor * vH[t-1]
            dlog_p_dGamma = common_factor * (vR[t-1]**2 if vR[t-1] < 0 else 0)
            grad_matrix[t] = [dlog_p_dOmega, dlog_p_dAlpha, dlog_p_dBeta, dlog_p_dGamma]
    elif model == "EGARCH":
        dOmega, dAlpha, dBeta, dGamma = vTheta
        vH = compute_variance_egarch_normal(vR, vTheta)
        for t in range(1, iT):
            e_t, h_t = vR[t], vH[t]
            eta_t_1 = vR[t-1] / np.sqrt(vH[t-1])
            log_h_t = np.log(h_t)
            common_factor = -0.5/h_t * (1 - e_t**2/h_t)
            dlog_p_dOmega = common_factor * h_t
            dlog_p_dAlpha = common_factor * h_t * (np.abs(eta_t_1) - np.sqrt(2/np.pi))
            dlog_p_dBeta = common_factor * h_t * log_h_t
            dlog_p_dGamma = common_factor * h_t * eta_t_1
            grad_matrix[t] = [dlog_p_dOmega, dlog_p_dAlpha, dlog_p_dBeta, dlog_p_dGamma]
    return grad_matrix[1:]

def compute_standard_errors_normal(vTheta, vR, model):
    grad_matrix = compute_gradient_normal(vTheta, vR, model)
    G = grad_matrix.T
    mCovariance_OPG = inv(G @ G.T)
    return np.sqrt(np.diag(mCovariance_OPG))

def log_likelihood_normal(vR, vH):
    return -0.5 * np.sum(np.log(2 * np.pi * vH) + (vR ** 2) / vH)

def nll_garch_normal(vTheta, vR):
    vH = compute_variance_garch(vR, vTheta)
    return -log_likelihood_normal(vR, vH)

def nll_gjr_normal(vTheta, vR):
    vH = compute_variance_garch(vR, vTheta, gjr=True)
    return -log_likelihood_normal(vR, vH)

def nll_egarch_normal(vTheta, vR):
    vH = compute_variance_egarch_normal(vR, vTheta)
    return -log_likelihood_normal(vR, vH)

def main(file_path, model="GARCH"):
    vR = load_data(file_path)
    num_observations = len(vR)
    if model == "EGARCH":
        vTheta_ini = [0.1, 0.3, 0.90, 0.1]
        vTheta_bnds = [(-0.01, 1), (0, 1), (0, 1), (-1, 1)]
        nll_function = nll_egarch_normal
        param_names = ["Omega", "Alpha", "Beta", "Gamma"]
        num_params = len(param_names)
    elif model == "GARCH":
        vTheta_ini = [0.01, 0.1, 0.75]
        vTheta_bnds = [(1e-6, 0.1), (1e-6, 0.5), (0.5, 0.999)]
        nll_function = nll_garch_normal
        param_names = ["Omega", "Alpha", "Beta"]
        num_params = len(param_names)
    elif model == "GJR-GARCH":
        vTheta_ini = [0.01, 0.02, 0.40, 0.1]
        vTheta_bnds = [(0, 1), (1e-6, 1), (0.7, 0.999), (0, 1)]
        param_names = ["Omega", "Alpha", "Beta", "Gamma"]
        num_params = len(param_names)
        nll_function = nll_gjr_normal
    else:
        raise ValueError("Unsupported model type!")
    vTheta_ML, log_likelihood = optimize_model(vR, nll_function, vTheta_ini, vTheta_bnds)
    standard_errors = compute_standard_errors_normal(vTheta_ML, vR, model)
    print("Normal Distribution")
    print(f"\nModel: {model}")
    print("Log-likelihood: ", log_likelihood)
    info_criteria = aic_bic(log_likelihood, num_params, num_observations)
    print(f"AIC: {info_criteria['AIC']:.4f}")
    print(f"BIC: {info_criteria['BIC']:.4f}")
    print("Parameter Estimates:")
    print(f"{'Parameter':<10}{'Estimate':<15}{'Std. Error':<15}")
    for name, est, se in zip(param_names, vTheta_ML, standard_errors):
        print(f"{name:<10}{est:<15.6f}{se:<15.6f}")

if __name__ == "__main__":
    file_path = r"C:\Users\rwe20\OneDrive\Desktop\FECS\daily_return_in.csv"
    main(file_path, model="GARCH")
    main(file_path, model="GJR-GARCH")
    main(file_path, model="EGARCH")


# ## MCMC

# In[45]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, gamma
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import timeit

def aic_bic(log_likelihood, num_params, num_observations):
    aic = -2 * log_likelihood + 2 * num_params
    bic = -2 * log_likelihood + num_params * np.log(num_observations)
    return {'AIC': aic, 'BIC': bic}

def compute_variance_garch(vR, vTheta):
    dOmega, dAlpha, dBeta = vTheta[:3]
    iT = len(vR)
    vH = np.ones(iT) * np.var(vR)
    for t in range(1, iT):
        vH[t] = dOmega + dAlpha * vR[t - 1] ** 2 + dBeta * vH[t - 1]
    return vH

def log_likelihood_student_t(vR, vH, dNu):
    return np.sum(gammaln((dNu + 1) / 2) - gammaln(dNu / 2) - 0.5 * np.log((dNu - 2) * np.pi * vH) - 0.5 * (dNu + 1) * np.log(1 + (vR ** 2) / ((dNu - 2) * vH)))

def nll_garch_student_t(vTheta, vR):
    dNu = vTheta[-1]
    vH = compute_variance_garch(vR, vTheta)
    return -log_likelihood_student_t(vR, vH, dNu)

def compute_gradients(vTheta, vR, model="GARCH"):
    iT = len(vR)
    grad_matrix = np.zeros((iT, len(vTheta)))
    dOmega, dAlpha, dBeta, dNu = vTheta
    vH = compute_variance_garch(vR, vTheta[:3])
    for t in range(1, iT):
        e_t = vR[t]
        h_t = vH[t]
        dlog_p_dOmega = 0.5 * (1 - (1 + (e_t ** 2) / ((dNu - 2) * h_t)) ** (-1) * e_t ** 2 / ((dNu - 2) * h_t ** 2))
        dlog_p_dAlpha = dlog_p_dOmega * vR[t - 1] ** 2
        dlog_p_dBeta = dlog_p_dOmega * vH[t - 1]
        dlog_p_dNu = 0.5 * (1 - (1 + (e_t ** 2) / ((dNu - 2) * h_t)) ** (-1) * (e_t ** 2 / ((dNu - 2) ** 2 * h_t)) - 1 / (dNu - 2))
        grad_matrix[t] = [dlog_p_dOmega, dlog_p_dAlpha, dlog_p_dBeta, dlog_p_dNu]
    return grad_matrix[1:]

def bayesian_estimation(vTheta_ML, mCovariance, fMinusLogLikelihood, num_draws=5000, burn_in=200):
    iNumAcceptance = 0
    mTheta = np.zeros((num_draws, len(vTheta_ML)))
    mTheta[0] = vTheta_ML

    for i in range(1, num_draws):
        vTheta_candidate = multivariate_normal.rvs(mean=mTheta[i - 1], cov=mCovariance)
        
        if (vTheta_candidate[0] < 0 or vTheta_candidate[1] < 0 or vTheta_candidate[2] < 0 or
            vTheta_candidate[3] < 2.1 or vTheta_candidate[3] > 30):
            mTheta[i] = mTheta[i - 1]
        else:
            dAccProb = min(np.exp(-fMinusLogLikelihood(vTheta_candidate) + fMinusLogLikelihood(mTheta[i - 1])), 1)
            if np.random.uniform(0, 1) < dAccProb:
                mTheta[i] = vTheta_candidate
                iNumAcceptance += 1
            else:
                mTheta[i] = mTheta[i - 1]

    acceptance_rate = 100 * iNumAcceptance / num_draws
    posterior_mean = np.mean(mTheta[burn_in:], axis=0)
    posterior_std = np.std(mTheta[burn_in:], axis=0)

    return posterior_mean, posterior_std, acceptance_rate, mTheta[burn_in:]

def optimize_model(vR, nll_function, vTheta_ini, vTheta_bnds):
    result = minimize(nll_function, vTheta_ini, args=(vR,), method="L-BFGS-B", bounds=vTheta_bnds)
    return result.x, -result.fun

def main(file_path):
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    vR = data['daily_return'].values

    vTheta_ini = [0.001, 0.2, 0.90, 4.5]
    
    vTheta_bnds = [(1e-6, 1), (1e-6, 1), (1e-6, 0.999), (2.1, 150)]
    

    vTheta_ML, log_likelihood = optimize_model(vR, nll_garch_student_t, vTheta_ini, vTheta_bnds)

    grad_matrix = compute_gradients(vTheta_ML, vR, model="GARCH")
    G = grad_matrix.T

    mCovariance_OPG = inv(G @ G.T) 
    standard_errors = np.sqrt(np.diag(mCovariance_OPG))

    print("Maximum Likelihood Estimates:")
    for i, param in enumerate(["Omega", "Alpha", "Beta", "Nu"]):
        print(f"{param}: {vTheta_ML[i]:.6f} (SE: {standard_errors[i]:.6f})")

    print(f"Log-Likelihood: {log_likelihood:.6f}")
    info_criteria = aic_bic(log_likelihood, len(vTheta_ini), len(vR))
    print(f"AIC: {info_criteria['AIC']:.6f}, BIC: {info_criteria['BIC']:.6f}")
    scaled_covariance = mCovariance_OPG * 1e-2 

    posterior_mean, posterior_std, acceptance_rate, posterior_samples = bayesian_estimation(
    vTheta_ML, scaled_covariance, lambda theta: nll_garch_student_t(theta, vR)
        )
    print("Bayesian Estimation Results:")
    for i, param in enumerate(["Omega", "Alpha", "Beta", "Nu"]):
        print(f"{param}: {posterior_mean[i]:.6f} (Posterior Std: {posterior_std[i]:.6f})")

    print(f"Acceptance Rate: {acceptance_rate:.2f}%")

    for i, param in enumerate(["Omega", "Alpha", "Beta", "Nu"]):
        plt.hist(posterior_samples[:, i], bins=20, label=param)
        plt.title(f"Posterior Distribution for {param}")
        plt.xlabel(param)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    
    quantiles = [2.5, 97.5]  # 95% credible interval

    print("\nBayesian Estimation Results (with 95% Credible Intervals):")
    for i, param in enumerate(["Omega", "Alpha", "Beta", "Nu"]):
        left_quantile, right_quantile = np.percentile(posterior_samples[:, i], quantiles)
        print(f"{param}: {posterior_mean[i]:.6f} (Posterior Std: {posterior_std[i]:.6f})")
        print(f"  - 95% Credible Interval: [{left_quantile:.6f}, {right_quantile:.6f}]")
    param_names = ["Omega", "Alpha", "Beta", "Nu"]


# Define parameter names and colors
    param_names = ["Omega", "Alpha", "Beta", "Nu"]
    colors = ["blue", "red", "green", "purple"]  # Different colors for each trace

    plt.figure(figsize=(12, 6))  # Set figure size

# Plot all traces on the same graph
    for i in range(4):
        plt.plot(posterior_samples[:, i], label=param_names[i], alpha=0.7, color=colors[i])

# Add title, labels, and legend
    plt.title("Trace Plots for All Parameters")
    plt.xlabel("Iterations")
    plt.ylabel("Parameter Values")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    file_path = r"C:\\Users\\rwe20\\OneDrive\\Desktop\\FECS\\daily_return_in.csv"
    main(file_path)


# In[3]:


def realized_garch_log(params, returns, realized_measure):
    omega, beta, gamma, xi, phi, sigma_u, tau1, tau2 = params[:8]
    n = len(returns)
    log_h = np.zeros(n)
    log_h[0] = np.log(np.var(returns))
    for t in range(1, n):
        log_h[t] = omega + beta * log_h[t-1] + gamma * np.log(realized_measure[t-1])
        z_t = returns[t-1] / np.sqrt(np.exp(log_h[t-1]))
        # Compute u_t based on the updated definition
        u_t = np.log(realized_measure[t]) - xi - phi * log_h[t] - tau1*z_t - tau2*(z_t**2 -1)
    return log_h, u_t


def log_likelihood_realized_garch(params, returns, realized_measure, distribution="normal"):
    log_h, u_t = realized_garch_log(params, returns, realized_measure)
    h = np.exp(log_h)
    if distribution == "normal":
        sigma_u = params[5]
        ll_returns = -0.5 * np.sum(np.log(2 * np.pi * h) + (returns ** 2) / h)
        ll_measure = -0.5 * np.sum(np.log(2 * np.pi * sigma_u ** 2) + (u_t ** 2) / (sigma_u ** 2))
    elif distribution == "student-t":
        sigma_u = params[5]
        nu = params[-1]
        ll_returns = log_likelihood_student_t(returns, h, nu)
        ll_measure = ll_measure = np.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2)- 0.5 * np.log(np.pi * nu * sigma_u ** 2)
                                         - (nu + 1) / 2 * np.log(1 + (u_t ** 2) / (nu * sigma_u ** 2)))
    else:  # skewed-student-t
        sigma_u = params[5]
        nu = params[-2]
        x = params[-1]
        ll_returns = log_likelihood_skewed_t(returns, h, nu, x )
        m = np.sqrt(nu / (nu - 2)) * (x - 1 / x) / np.sqrt(2)
        s = np.sqrt(1 + (x ** 2 + 1 / x ** 2 - 1) / 2 - m ** 2)
        I = np.where(u_t >= -m / s, 1, -1)
        ll_measure = np.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2)- 0.5 * np.log(np.pi * nu * sigma_u ** 2) - 0.5 * np.log(s ** 2 * x ** (2 * I))
                           - (nu + 1) / 2 * np.log(1 + (s * u_t + m) ** 2 / (nu * sigma_u ** 2)))

    return -(ll_returns + ll_measure)

def optimize_realized_garch(returns, realized_measure, distribution="normal"):
    if distribution == "normal":
        initial_params = [0.01, 0.85, 0.1, 0.0, 1.0, 0.1, 0.0, 0.0]
        bounds = [(1e-8, 0.1), (0.7, 0.999), (-1, 1),  # GARCH parameters
                  (-5, 5), (0.01, 5), (1e-8, 2),       # Measurement equation parameters
                  (None, None), (0.1,1.5)]                    # Skewness parameters
    elif distribution == "student-t":
        initial_params = [0.01, 0.8, 0.1, 0.0, 1.0, 0.1, 0.0, 0.0, 10]
        bounds = [(1e-8, 1), (0, 0.999), (-1, 1),  # GARCH parameters
                  (-5, 5), (0.01, 5), (1e-8, 2),       # Measurement equation parameters
                  (None, None), (-2, 2), (2.6, 30)]         # Nu parameter
    elif distribution == "skewed-student-t":
        initial_params = [0.1, 0.8, 0.1, 0.0, 1.0, 0.1, 0.0, 0.0, 10, 1]
        bounds = [(1e-8, 1), (0.7, 0.999), (-1, 1),  # GARCH parameters
                  (-5, 5), (0.01, 5), (1e-8, 2),       # Measurement equation parameters
                  (None, None), (-1, 1), (2.6, 30), (0.1, 1.5)]  # Skewness parameters

    result = minimize(log_likelihood_realized_garch, initial_params, args=(returns, realized_measure, distribution), method="L-BFGS-B", bounds=bounds)
    hessian_inv = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else np.linalg.inv(result.hess_inv)
    standard_errors = np.sqrt(np.diag(hessian_inv))
    return result.x, -result.fun, standard_errors

def main_realized_garch(file_path, realized_measure_path):
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    realized_measure_data = pd.read_csv(realized_measure_path, parse_dates=True, index_col=0)

    returns = data['daily_return'].values
    returns = returns * 100
    realized_kernel = realized_measure_data['realized_kernel'].values 
    realized_volatility = returns **2

    for distribution in ["normal", "student-t", "skewed-student-t"]:
        params, log_likelihood, std_errors = optimize_realized_garch(returns, realized_kernel, distribution)
        num_params = len(params)
        num_obs = len(returns)
        info_criteria = aic_bic(log_likelihood, num_params, num_obs)
        
        # Updated parameter names list
        base_params = ["Omega", "Beta", "Gamma", "Xi", "Phi", "Sigma_u", "Tau1", "Tau2"]
        if distribution == "student-t":
            param_names = base_params + ["Nu"]
        elif distribution == "skewed-student-t":
            param_names = base_params + ["Nu", "X"]
        else:
            param_names = base_params

        print(f"\nRealized GARCH Model with Realized Kernel ({distribution.capitalize()} Distribution)")
        print(f"Log-Likelihood: {log_likelihood:.4f}")
        print(f"AIC: {info_criteria['AIC']:.4f}")
        print(f"BIC: {info_criteria['BIC']:.4f}")
        print("\nParameters and Standard Errors:")
        for name, value, std_error in zip(param_names, params, std_errors):
            print(f"{name}: {value:.6f} (SE: {std_error:.6f})")


if __name__ == "__main__":
    file_path = r"C:\\Users\\rwe20\\OneDrive\\Desktop\\FECS\\daily_return_in.csv"
    realized_measure_path = r"C:\\Users\\rwe20\\OneDrive\\Desktop\\FECS\\RK_in_0122.csv"
    main_realized_garch(file_path, realized_measure_path)


# In[ ]:


def optimize_realized_garch(returns, realized_measure, distribution="normal"):
    if distribution == "normal":
        initial_params = [0.01, 0.85, 0.1, 0.0, 1.0, 0.1, 0.0, 0.0]
        bounds = [(1e-8, 0.1), (0.7, 0.999), (-1, 1),  # GARCH parameters
                  (-5, 5), (0.01, 5), (1e-8, 2),       # Measurement equation parameters
                  (None, None), (None,None)]                    # Skewness parameters
    elif distribution == "student-t":
        initial_params = [0.01, 0.8, 0.1, 0.0, 1.0, 0.1, 0.0, 0.0, 10]
        bounds = [(1e-8, None), (0, 0.999), (-1, 1),  # GARCH parameters
                  (-5, 5), (0.01, 5), (1e-8, 2),       # Measurement equation parameters
                  (None, None), (None, None), (3, 30)]         # Nu parameter
    elif distribution == "skewed-student-t":
        initial_params = [0.1, 0.8, 0.1, 0.0, 1.0, 0.1, 0.0, 0.0, 10, 1.5]
        bounds = [(1e-8, None), (0.7, 0.999), (-1, 1),  # GARCH parameters
                  (-5, 5), (0.01, 5), (1e-8, 2),       # Measurement equation parameters
                  (-2, 2), (-2, 2), (2.1, 30), (0.1, 2.5)]  # Skewness parameters

    result = minimize(log_likelihood_realized_garch, initial_params, args=(returns, realized_measure, distribution), method="L-BFGS-B", bounds=bounds)
    hessian_inv = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else np.linalg.inv(result.hess_inv)
    standard_errors = np.sqrt(np.diag(hessian_inv))
    return result.x, -result.fun, standard_errors

def main_realized_garch(file_path, realized_measure_path):
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    realized_measure_data = pd.read_csv(realized_measure_path, parse_dates=True, index_col=0)

    returns = data['daily_return'].values * 100
    realized_volatility = realized_measure_data['realized_volatility']

    for distribution in ["normal", "student-t", "skewed-student-t"]:
        params, log_likelihood, std_errors = optimize_realized_garch(returns, realized_volatility, distribution)
        num_params = len(params)
        num_obs = len(returns)
        info_criteria = aic_bic(log_likelihood, num_params, num_obs)
        
        # Updated parameter names list
        base_params = ["Omega", "Beta", "Gamma", "Xi", "Phi", "Sigma_u", "Tau1", "Tau2"]
        if distribution == "student-t":
            param_names = base_params + ["Nu"]
        elif distribution == "skewed-student-t":
            param_names = base_params + ["Nu", "X"]
        else:
            param_names = base_params

        print(f"\nRealized GARCH Model with Realized Volatility({distribution.capitalize()} Distribution)")
        print(f"Log-Likelihood: {log_likelihood:.4f}")
        print(f"AIC: {info_criteria['AIC']:.4f}")
        print(f"BIC: {info_criteria['BIC']:.4f}")
        print("\nParameters and Standard Errors:")
        for name, value, std_error in zip(param_names, params, std_errors):
            print(f"{name}: {value:.6f} (SE: {std_error:.6f})")


if __name__ == "__main__":
    file_path = r"C:\\Users\\rwe20\\OneDrive\\Desktop\\FECS\\daily_return_in.csv"
    realized_measure_path = r"C:\\Users\\rwe20\\OneDrive\\Desktop\\FECS\\RK_in_0122.csv"
    main_realized_garch(file_path, realized_measure_path)


# In[ ]:





# In[ ]:





# In[ ]:




