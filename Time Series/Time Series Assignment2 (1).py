import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
def load_data():
    sv_data = pd.read_excel(r"C:\Users\rwe20\OneDrive\Desktop\Time Series\sv_log_returns.xlsx")
    y_t = sv_data.iloc[:,0].values
    k = -11.277092 # QML estimates from c
    sigma = np.exp((k+1.27)/2) # QML estimates from c
    phi = 0.991127 # QML estimates from c
    y_bar = (y_t - np.mean(y_t)) / sigma
    n = len(y_t)
    sigma_eta2 = 0.007047
    mu = np.mean(y_t)
    return y_t, y_bar, n, sigma, phi, sigma_eta2, mu, k
y_t, y_bar, n, sigma, phi, sigma_eta2, mu, k = load_data()
##############################################################################
# Question a
file_path = r"C:\Users\rwe20\OneDrive\Desktop\Time Series\sv_log_returns.xlsx"
df = pd.read_excel(file_path)
log_returns = df["GBPUSD"]

# (i) 
plt.figure(figsize=(10, 4))
plt.plot(y_t, color='black', linewidth=0.5)
plt.xlabel("Time")
plt.ylabel("Log Returns")
plt.title("(i) Daily Log Returns of Exchange Rates")
plt.show()

# (ii) 
log_squared_returns = np.log(log_returns**2)

plt.figure(figsize=(10, 4))
plt.scatter(range(len(log_squared_returns)), log_squared_returns, color='black', s=5)
plt.xlabel("Time")
plt.ylabel("log(y_t^2)")
plt.title("(ii) Log Squared Returns with Smoothed Estimate")
plt.show()

# (iii) 
smoothed_volatility = np.exp(pd.Series(log_squared_returns).ewm(span=50).mean() / 2)

plt.figure(figsize=(6, 2))
plt.plot(smoothed_volatility, color='black', linewidth=1)
plt.xlabel("Time")
plt.ylabel("Estimated Volatility")
plt.title("(iii) Smoothed Estimate of Volatility Measure exp(θ_t / 2)")
plt.show()

descriptive_stats = df.describe()
print(descriptive_stats)
###############################################################################
#b
mu = log_returns.mean()  
x_t = np.log((log_returns - mu) ** 2)  

def plot_b():
    plt.figure(figsize=(10, 4))
    plt.plot(x_t, color='black', linewidth=0.5)
    plt.xlabel("Time")
    plt.ylabel("x_t")
    plt.title("Transformed Returns x_t Time Series")
    plt.show()
    return

plot_b()

###############################################################################
#c
mu = df["GBPUSD"].mean()
df["x_t"] = np.log((df["GBPUSD"] - mu) ** 2)
x_t = df["x_t"].values
n = len(x_t)

def neg_log_likelihood(params):
    kappa, phi, sigma_eta_sq = params

    P_t = sigma_eta_sq / max(1e-6, (1 - phi**2))  
    a_t = 0 
    log_likelihood = 0

    for t in range(n):
        Ft = P_t + 4.93 
        v_t = x_t[t] - kappa - a_t  

        if Ft <= 0:  
            return np.inf
        
        log_likelihood += 0.5 * (np.log(2 * np.pi * Ft) + (v_t ** 2) / Ft)
        
        K_t = P_t / Ft  
        a_t = phi * a_t + K_t * v_t  
        P_t = phi**2 * P_t + sigma_eta_sq - K_t * P_t  

    return log_likelihood  

initial_params = [0, 0, 0]  
bounds = [(-np.inf, np.inf), (0, 1), (1e-6, np.inf)] 

result = minimize(neg_log_likelihood, initial_params, bounds=bounds, method="L-BFGS-B")

kappa_hat, phi_hat, sigma_eta_sq_hat = result.x
log_likelihood_hat = -result.fun 

param_results = pd.DataFrame({
    "Parameter": ["κ (kappa)", "φ (phi)", "σ²_η (sigma_eta_sq)", "Log-likelihood"],
    "Estimate": [kappa_hat, phi_hat, sigma_eta_sq_hat, log_likelihood_hat]
})

print(param_results)

###############################################################################
#d
import statsmodels.api as sm
# Step 1: Compute x_t
x_t = np.log((np.abs(y_t - mu)) ** 2)

# Step 2: Kalman filter estimation for α_t
mod_kalman = sm.tsa.statespace.SARIMAX(
    x_t - k,
    order=(1, 0, 0), 
    trend="n", 
    measurement_error=True,  
)
res_kalman = mod_kalman.fit(disp=False)

# Initialize Kalman filter variables
a_t = np.zeros(n + 1)
P_t = np.zeros(n + 1)
v_t = np.zeros(n)
F_t = np.zeros(n)
K_t = np.zeros(n)

a_t[0] = 0
P_t[0] = 1e-7

# Kalman filter loop
for i in range(n):
    v_t[i] = x_t[i] - (k + a_t[i])
    F_t[i] = P_t[i] + 4.93
    K_t[i] = P_t[i] / F_t[i]
    a_t[i + 1] = phi * a_t[i] + K_t[i] * v_t[i]
    P_t[i + 1] = phi**2 * (P_t[i] - K_t[i] * P_t[i]) + sigma_eta2

# Kalman smoother variables
r = np.zeros(n)
N = np.zeros(n)
a_smooth = np.zeros(n)
V_smooth = np.zeros(n)

a_smooth[-1] = a_t[-1]
V_smooth[-1] = P_t[-1]

# Kalman smoothing loop
for t in range(n - 2, -1, -1):
    L = phi - K_t[t] * phi  
    r[t] = v_t[t] / F_t[t] + L * r[t + 1]  
    N[t] = 1 / F_t[t] + L**2 * N[t + 1]  
    a_smooth[t] = a_t[t] + P_t[t] * r[t]
    V_smooth[t] = P_t[t] - P_t[t] ** 2 * N[t]

# Step 4: Compute θ_t = κ + α_t
theta_filtered = k + a_t[:-1]  # Remove last point to match x_t length
theta_smoothed = k + a_smooth

# Step 5: Plot Filtered and Smoothed α_t
plt.figure(figsize=(10, 6))
plt.plot(a_t[:-1], label="Filtered α_t from QML", color='blue', linewidth=2)
plt.plot(a_smooth, label="Smoothed α_t from QML", color='green', linewidth=2)
plt.xlabel("Time")
plt.ylabel("α_t")
plt.title("Filtered and Smoothed α_t from QML")
plt.legend()
plt.grid()
plt.show()

# Step 6: Plot Smoothed θ_t and Transformed Data x_t (Plot 2)
plt.figure(figsize=(10, 5))
plt.plot(theta_smoothed, label="Smoothed θ_t", color="blue", linewidth=2)
plt.plot(x_t, label="Transformed Data x_t", linestyle="None", marker=".", color="black")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Plot of Smoothed θ_t and Transformed Data x_t")
plt.legend()
plt.grid()
plt.show()


###############################################################################
# QML estimates
def kalman_filter(y, sigma_eps2, sigma_eta2):
    n = len(y)
    a_t = np.zeros(n+1)  # State estimates
    P_t = np.zeros(n+1)  # State variances
    v_t = np.zeros(n)  
    F_t = np.zeros(n)  
    K_t = np.zeros(n)
    a_t[0] = 0
    P_t[0] = 1e-7  
    
    for i in range(n):
        v_t[i] = y[i] - a_t[i] 
        F_t[i] = P_t[i] + sigma_eps2[i]
        K_t[i] = P_t[i] / F_t[i]
        a_t[i+1] = phi * a_t[i] + K_t[i] * v_t[i] #prediction
        P_t[i+1] = phi**2 * (P_t[i] - K_t[i] * P_t[i]) + sigma_eta2
    return a_t[1:], P_t[1:], v_t, F_t, K_t

def kalman_smoother(y, a_filt, P_filt, v, F, K, sigma_eps2, sigma_eta2):
    n = len(y)
    r = np.zeros(n)
    N = np.zeros(n)
    a_smooth = np.zeros(n)
    V_smooth = np.zeros(n)
    # Initialize at last point
    a_smooth[-1] = a_filt[-1]  
    V_smooth[-1] = P_filt[-1]
    for t in range(n-2, -1, -1):
        L = phi - K[t+1] * phi
        r[t] = v[t+1] / F[t+1] + L * r[t+1]
        N[t] = 1 / F[t+1] + L**2 * N[t+1]
        a_smooth[t] = a_filt[t] + P_filt[t] * r[t] 
        V_smooth[t] = P_filt[t] - P_filt[t]**2 * N[t]
    return a_smooth, V_smooth, r, N

def evaluate_z_A(y_bar, g=None, first_iteration=False):
    if first_iteration:
        A_t = np.full(n,2)
        z_t = 2 * np.log(np.abs(y_bar))
    else:
        A_t = 2 * np.exp(g) / (y_bar ** 2)
        z_t = g + 1 - np.exp(g) / (y_bar ** 2)
    return z_t, A_t


def mode_estimation(tol=1e-7, max_iter=1000):
    iteration = 0
    z_t, A_t = evaluate_z_A(y_bar, g=None, first_iteration=(iteration==0))
    a_t, P_t, v_t, F_t, K_t = kalman_filter(z_t, A_t, sigma_eta2)
    a_smooth, V_smooth, r, N = kalman_smoother(z_t, a_t, P_t, v_t, F_t, K_t, A_t, sigma_eta2)
    g = a_smooth # initial
    g_updates = []
    while iteration < max_iter:
        g_old = g.copy()
        z_t, A_t = evaluate_z_A(y_bar, g, first_iteration=False)
        a_t, P_t, v_t, F_t, K_t = kalman_filter(z_t, A_t, sigma_eta2)
        a_smooth, V_smooth, r, N = kalman_smoother(z_t, a_t, P_t, v_t, F_t, K_t, A_t, sigma_eta2)
        g  =  a_smooth 
        g_updates.append(g.copy())  

        max_diff = np.max(np.abs(g - g_old))
        print(f"Iteration {iteration}: max change in g = {max_diff}") 

        if max_diff <= tol:
            print(f"Converged in {iteration} iterations.")
            break

        iteration += 1

    if iteration == max_iter:
        print("Did not converge within max iterations.")

    return g, g_updates

g, g_updates = mode_estimation()

###############################################################################
def plot_e():
    plt.figure(figsize=(10, 6))
    plt.plot(a_smooth, label="Smoothed α_t from QML", color='gray', linewidth=2)
    plt.plot(g, label='Mode Estimation', color='blue')
    plt.xlabel("Time")
    plt.ylabel("Smoothed State")
    plt.legend()
    plt.grid()
    plt.show()
    
    return
plot_e()

###############################################################################
# Question F


def initialize_particles(M, a1, P1):
    return np.random.normal(a1, np.sqrt(P1), M)

def draw_particles(prev_particles, phi, sigma_eta2, M):
    return np.random.normal(loc=phi * prev_particles, scale=np.sqrt(sigma_eta2), size=M)

def compute_likelihood(y_t, particles, mu):
    log_likelihood = -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma) + particles) - ((y_t - mu) ** 2) / (2 * sigma**2 * np.exp(particles))
    likelihoods = np.exp(log_likelihood)
    likelihoods /= np.sum(likelihoods)
    return likelihoods

def normalize_weights(likelihoods):
    weights = likelihoods / np.sum(likelihoods)
    return weights

def resample_particles(particles, weights, M):
    indices = np.random.choice(np.arange(M), size=M, p=weights)
    return particles[indices]

def particle_filter(y_t_seq, M, sigma_eta2, sigma_eps2, phi, a1, P1):
    particles = initialize_particles(M, a1, P1)
    weights = np.ones(M) / M
    estimates = np.zeros(len(y_t_seq))
    
    for t in range(len(y_t_seq)):
        particles = draw_particles(particles, phi, sigma_eta2, M)
        likelihoods = compute_likelihood(y_t_seq[t], particles, sigma_eps2)
        weights = normalize_weights(likelihoods)
        estimates[t] = np.sum(particles * weights)
        particles = resample_particles(particles, weights, M)
    
    return estimates

# Parameters from QML estimates
a1 = 0
P1 = sigma_eta2 / (1 - phi**2)
M = 10000  # Number of particles

# Run particle filter
x_hat = particle_filter(y_t, M, sigma_eta2, sigma**2, phi, a1, P1)

# Plot results
def plot_particle(x_hat, a_t):
    plt.figure(figsize=(10, 5))
    plt.plot(x_hat, label="Bootstrap Filter", color='b')
    plt.plot(a_t, label="Filter from QML", color='g')
    plt.xlabel("Time Step")
    plt.ylabel("Estimated State")
    plt.title("Particle Filter Estimated States Over Time")
    plt.legend()
    plt.show()

plot_particle(x_hat, a_t)

###############################################################################
# ------------ PART (e) Mode Estimation for Smoothing Density ------------
print("\nPart (e): Mode estimation for smoothing density")
def sv_log_posterior(alpha, y, phi, sigma_eta, kappa):

    n = len(y)
    sigma = np.exp((kappa + 1.27) / 2)  # Convert kappa to sigma
    log_posterior = 0

    # Prior for first state (stationary distribution)
    stationary_var = sigma_eta ** 2 / (1 - phi ** 2)
    log_posterior += -0.5 * np.log(2 * np.pi * stationary_var) - 0.5 * (alpha[0] ** 2) / stationary_var

    # State transitions
    for t in range(1, n):
        trans_var = sigma_eta ** 2
        log_posterior += -0.5 * np.log(2 * np.pi * trans_var) - 0.5 * ((alpha[t] - phi * alpha[t - 1]) ** 2) / trans_var

    # Observation likelihood - using the correct formula from DK-book Section 10.6.5
    # log p(y_t|θ_t) = -1/2[log 2πσ² + θ_t + z_t² exp(-θ_t)]
    for t in range(n):
        z_t = (y[t] - mu) / sigma
        log_posterior += -0.5 * (np.log(2 * np.pi * sigma ** 2) + alpha[t] + z_t ** 2 * np.exp(-alpha[t]))

    return -log_posterior  # Return negative for minimization


# Convert parameters for the original SV model
sigma_eta_hat = np.sqrt(sigma_eta_sq_hat)

# Initial guess for alpha (using smoothed estimates from QML)
alpha_initial = a_smooth

# Optimization to find mode
print("\nRunning mode estimation (this may take a while)...")

try:
    mode_result = minimize(
        sv_log_posterior,
        alpha_initial,
        args=(log_returns, phi_hat, sigma_eta_hat, kappa_hat),
        method="L-BFGS-B",
        options={'maxiter': 50, 'ftol': 1e-5}
    )

    # Extract the mode estimate
    alpha_mode = mode_result.x
    print(f"Mode estimation completed: {mode_result.message}")

except Exception as e:
    print(f"Mode estimation failed with error: {e}")
    # Fallback: Use smoothed estimates as proxy for mode
    print("Using smoothed estimates as proxy for mode")
    alpha_mode = a_smooth

# Convert to volatility
vol_mode = alpha_mode  # Volatility from mode
vol_smoothed = a_smooth  # Volatility from QML smoothed

# Plot comparison of volatility estimates
plt.figure(figsize=(12, 6))
plt.plot(vol_mode, label="Mode Estimate", color="black", linewidth=1.5)
plt.plot(vol_smoothed, label="QML Smoothed alpha", color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.title("(e) Comparison of Mode and Smoothed QML Alpha")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_e_mode_smoothed.png", dpi=300)
plt.show()
