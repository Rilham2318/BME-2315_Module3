# cavalitidePKPD.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Task 3 part 1

# Cavalitide dosing information from Task 2
Favailable = 0.01
ka = 0.006
D = 21
interval = 24
n_doses = 100

# Other drug parameters
ke = 0.0039
V = 10.8
MW = 4113.6
Ffree = 0.01

# Time between saved results
dt = 0.1

# GLP1 receptor parameters from Task 1
Kd1 = 0.17       # GLP1 Kd
Kd2 = 0.077      # Fitted Cavalitide Kd
L1 = 0.015       # GLP1 concentration

# Calculate drug absorption and elimination
def ODEfunc(t, y):
    Asc, Cc = y
    dAsc = -ka * Asc
    dCc = ka * Asc / V - ke * Cc
    return [dAsc, dCc]

# Simulate Cavalitide doses over time
def simulate(D, n_doses, interval, dt):

    t_out = []
    Cc_out = []

    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))

    for i in range(n_doses):

        y[0] += Favailable * D

        t0 = i * interval
        t1 = (i + 1) * interval
        t_eval = np.linspace(t0, t1, n_steps + 1)

        sol = solve_ivp(
            ODEfunc, (t0, t1), y,
            t_eval=t_eval,
            method="RK45"
        )

        if not sol.success:
            raise RuntimeError("PK simulation failed")

        t_out.append(sol.t)
        Cc_out.append(sol.y[1])

        y = sol.y[:, -1].copy()

    t = np.concatenate(t_out)
    Cc = np.concatenate(Cc_out)

    # Convert blood concentration to nM
    Cc_nM = Cc * 1e-3 / MW * 1e9

    return t, Cc_nM

# Calculate receptor activation
def Fact(L1, L2, Kd1, Kd2):
    return (Kd2 * L1 + Kd1 * L2) / (Kd1 * Kd2 + Kd2 * L1 + Kd1 * L2)

# Run the simulation
t, Cc = simulate(D, n_doses, interval, dt)

# Calculate free Cavalitide concentration
L2 = Ffree * Cc

# Calculate fraction of active receptors over time
Fact_values = Fact(L1, L2, Kd1, Kd2)

# Print receptor activation at day 100
print(f"Receptor activation at day 100: {Fact_values[-1]*100:.1f}%")

# Plot receptor activation over time
plt.plot(t / 24, Fact_values)

plt.xlabel("Time (days)")
plt.ylabel("Fraction active receptors (Fact)")
plt.title("Cavalitide GLP1 Receptor Activation Over 100 Days")
plt.ylim(0, 1)
plt.grid(alpha=0.3)

plt.show()



# Task 3 part 2: Optimize Cavalitide dosing

# Increase the daily dose to improve receptor activation
D_optimized = 29

# Run the simulation with the new dose
t_optimized, Cc_optimized = simulate(D_optimized, n_doses, interval, dt)

# Calculate free Cavalitide concentration
L2_optimized = Ffree * Cc_optimized

# Calculate receptor activation with the new dose
Fact_optimized = Fact(L1, L2_optimized, Kd1, Kd2)

# Print and compare receptor activation at day 100
print(f"Original receptor activation: {Fact_values[-1]*100:.1f}%")
print(f"Optimized receptor activation: {Fact_optimized[-1]*100:.1f}%")

# Plot both dosing strategies
plt.figure()

plt.plot(t / 24, Fact_values, label="Original dose (21 mg)")
plt.plot(t_optimized / 24, Fact_optimized, label="Optimized dose (29 mg)")

# Show the 90% target
plt.axhline(y=0.90, color="red", linestyle="--", label="90% target")

plt.xlabel("Time (days)")
plt.ylabel("Fraction active receptors (Fact)")
plt.title("Original vs. Optimized Cavalitide Dosing")
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)

plt.show()
