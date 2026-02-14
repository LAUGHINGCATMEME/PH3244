import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# data
x = np.array([200, 180, 150, 120, 100], dtype=float)
y = np.array([110.25, 93.44444444, 77.73361111, 64.8025, 50.41], dtype=float)
sigma = np.array([
    6.500961467,
    2.161532378,
    3.239450185,
    0.9295339334,
    0.4099186911
], dtype=float)

# ---- weighted mx fit ----
w = 1 / sigma**2

m = np.sum(w * x * y) / np.sum(w * x**2)
sigma_m = np.sqrt(1 / np.sum(w * x**2))

# fitted curve
xfit = np.linspace(np.min(x), np.max(x), 400)
yfit = m * xfit

# ---- chi square ----
residuals = y - m*x
chi2_val = np.sum((residuals/sigma)**2)
dof = len(x) - 1
reduced_chi2 = chi2_val / dof
p_value = 1 - chi2.cdf(chi2_val, dof)

print(f"slope m = {m:.6g}")
print(f"uncertainty in slope = {sigma_m:.2g}")
print(f"reduced chi^2 = {reduced_chi2:.4f}")
print(f"p-value = {p_value:.5f}")

# ---- plot ----
plt.errorbar(x, y, yerr=sigma, fmt='o', capsize=3, label='data')
plt.plot(xfit, yfit, label='Weighted fit')

plt.xlabel(r"Voltage (V)")
plt.ylabel(r"Distance$^2$ (cm$^2$)")
plt.legend()
plt.tight_layout()
plt.show()

