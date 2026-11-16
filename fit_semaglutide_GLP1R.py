# fit_semaglutide_GLP1R.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Task A (Adapt your previous codes for modeling semaglutide (fit_semaglutide.py, semaglutide_data.csv) to fit the Cavalitide binding data)

def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists"""
    return (Kd2*L1 + Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

# Parameters [nM]
Kd1 = 0.17       # GLP1 dissociation constant
L1 = 0.015       # GLP1 concentration (fasting)

# Load Cavalitide experimental data
data = np.loadtxt("cavalitide_data.csv", delimiter=",", skiprows=1)

L2_loaded = data[:, 0]
Fact_loaded = data[:, 1]

# Approach 1: Brute force optimization of Kd2
Kd2_candidates = np.logspace(-3, 2, 100)
SSE_brute = np.inf

for Kd2_try in Kd2_candidates:
    SSE = np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, Kd2_try))**2)

    if SSE < SSE_brute:
        SSE_brute = SSE
        Kd2_brute = Kd2_try

print(f"Brute-force Kd2 = {Kd2_brute:.3f} nM (SSE = {SSE_brute:.4f})")

# Approach 2: Using scipy curve_fit
popt, pcov = curve_fit(
    lambda L2, Kd2: Fact(L1, L2, Kd1, Kd2),
    L2_loaded,
    Fact_loaded,
    p0=[0.1],
    bounds=(0, np.inf)
)

Kd2_fit = popt[0]

SSE_fit = np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, Kd2_fit))**2)

print(f"Fitted Kd2 = {Kd2_fit:.3f} nM (SSE = {SSE_fit:.4f})")

# Task B (Show a plot of [Cavalitide] vs. Fraction active receptors (Fact) demonstrating a good fit)

# Generate concentrations for a smooth model curve
L2 = np.logspace(-3, 2, 500)

# Calculate predicted receptor activation using fitted Kd
Fact_fit = Fact(L1, L2, Kd1, Kd2_fit)

# Plot experimental data and fitted model
plt.figure()

plt.semilogx(L2, Fact_fit, lw=2,
             label=f"Fitted model (Kd = {Kd2_fit:.3f} nM)")

plt.semilogx(L2_loaded, Fact_loaded, "o",
             label="Experimental data")

plt.ylim(-0.1, 1.1)
plt.xlabel("[Cavalitide] (nM)")
plt.ylabel("Fraction active receptors (Fact)")
plt.title("GLP1R Activation by Cavalitide")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


# Task 1C: Compare Cavalitide and Semaglutide Kd values

# Cavalitide fitted Kd from Task 1A
Kd2_cavalitide = Kd2_fit

print(f"Cavalitide fitted Kd = {Kd2_cavalitide:.3f} nM")

# Load semaglutide experimental data
sema_data = np.loadtxt("semaglutide_data.csv", delimiter=",", skiprows=1)

L2_sema = sema_data[:, 0]
Fact_sema = sema_data[:, 1]

# Fit semaglutide Kd using the same model
popt_sema, pcov_sema = curve_fit(
    lambda L2, Kd2: Fact(L1, L2, Kd1, Kd2),
    L2_sema,
    Fact_sema,
    p0=[0.1],
    bounds=(0, np.inf)
)

Kd2_semaglutide = popt_sema[0]

print(f"Semaglutide fitted Kd = {Kd2_semaglutide:.3f} nM")

# Compare the two Kd values
print(f"Cavalitide Kd is {Kd2_semaglutide/Kd2_cavalitide:.2f} times lower than semaglutide.")
