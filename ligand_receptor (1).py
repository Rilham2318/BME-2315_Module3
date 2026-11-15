# ligand_receptor.py

import numpy as np
import matplotlib.pyplot as plt

def Fact(L1, L2, Kd1, Kd2):
    """Fractional GLP1R activation by GLP-1 and semaglutide."""
    return (Kd2 * L1 + Kd1 * L2) / (
        Kd1 * Kd2 + Kd2 * L1 + Kd1 * L2
    )


Kd1 = 0.17  # GLP-1 dissociation constant [nM]
Kd2 = 0.38  # Semaglutide dissociation constant [nM]

L1s = {
    "fasting": 0.015,     # GLP-1 concentration [nM]
    "after meal": 0.040,  # GLP-1 concentration [nM]
}
L2 = np.logspace(-3, 2, 500)  # Semaglutide concentration [nM]

plt.figure()
for state, L1 in L1s.items():
    line, = plt.semilogx(
        L2, Fact(L1, L2, Kd1, Kd2), lw=2,
        label=f"GLP-1 {state} ({L1} nM)"
    )
    plt.axhline(
        Fact(L1, 0, Kd1, Kd2), ls=":", c=line.get_color(), lw=1
    )

plt.axvline(
    Kd2, ls=":", c="k", lw=1,
    label=f"Semaglutide Kd ({Kd2} nM)"
)
plt.axhline(0.5, ls="--", c="gray", lw=1)
plt.ylim(0, 1)
plt.xlabel("[Semaglutide] (nM, log scale)")
plt.ylabel("Fraction active")
plt.title("GLP1R activation by GLP-1 and semaglutide")
plt.legend(loc="upper left")
plt.show()