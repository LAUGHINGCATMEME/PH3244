# save_as_plots_and_fit_h_vs_B2.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

# -------------------------
# DATA (from your uploaded file). Source cited in chat.
# -------------------------
current_h = np.array([0,0.8,1,1.25,1.5,1.75,2,2.5,2.75,3,3.25,3.3])
mainscale = np.array([9.5,11.5,12.5,13.5,16.5,18.5,20.0,24,24,24.5,25,25])
circularscale = np.array([48,10,5,3,2,12,19,12,38,15,40,48])

# delta_h as in your file
delta_h = mainscale + 0.01*circularscale - 9.98 

current_A = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.1,2.5,3.0,3.5,3.55])
feild_str_gauss = np.array([423,729,1052,1400,1769,2150,2550,2970,3380,3790,4210,6340,8490,10230,11840,13260,13390]) 

# -------------------------
# UNCERTAINTIES (least counts -> standard uncertainties)
# -------------------------
current_lc = 0.01
sigma_I = current_lc / np.sqrt(12)               # σ on current (A)

# magnetic field least count rule (>2000 -> 10 else 1)
field_lc = np.where(feild_str_gauss > 2000, 10.0, 1.0)
sigma_B = field_lc / np.sqrt(12)                 # σ on B (Gauss)

# height uncertainty: mainscale reading + circular count converted to mm
h_lc = 0.01
sigma_h = np.sqrt((h_lc/np.sqrt(12))**2 + (0.01/np.sqrt(12))**2)

# x-errors arrays (same σ for all current points)
sx_A = np.full_like(current_A, sigma_I)
sx_h = np.full_like(current_h, sigma_I)

# -------------------------
# MODELS (linear mx + c)
# -------------------------
def linear_model(beta, x):
    return beta[0] * x + beta[1]

def odr_linear_fit(x, y, sx, sy, beta0):
    data = odr.RealData(x, y, sx=sx, sy=sy)
    model = odr.Model(linear_model)
    odr_obj = odr.ODR(data, model, beta0=beta0)
    out = odr_obj.run()
    return out

# initial guesses
beta0_field = [4000.0, 0.0]   # initial slope (Gauss/A), intercept
beta0_height = [0.0, 0.5]     # initial slope (mm/A), intercept

# perform fits (B vs I and h vs I)
fit_field = odr_linear_fit(current_A, feild_str_gauss, sx_A, sigma_B, beta0_field)
fit_height = odr_linear_fit(current_h, delta_h, sx_h, np.full_like(delta_h, sigma_h), beta0_height)

# extract results
m_field, c_field = fit_field.beta
dm_field, dc_field = fit_field.sd_beta

m_h, c_h = fit_height.beta
dm_h, dc_h = fit_height.sd_beta

print("=== Magnetic field fit (B = m I + c) ===")
print(f"m = {m_field:.6g} ± {dm_field:.6g}  (Gauss/A)")
print(f"c = {c_field:.6g} ± {dc_field:.6g}  (Gauss)")

print("\n=== Meniscus rise fit (Δh = m I + c) ===")
print(f"m = {m_h:.6g} ± {dm_h:.6g}  (mm/A)")
print(f"c = {c_h:.6g} ± {dc_h:.6g}  (mm)")

# -------------------------
# Save two separate plots (Field vs I) and (h vs I)
# -------------------------
# 1) Field vs Current
fig, ax = plt.subplots(figsize=(7,6))
xgrid = np.linspace(np.min(current_A)*0.95, np.max(current_A)*1.05, 300)
y_fit_field = linear_model(fit_field.beta, xgrid)

ax.errorbar(current_A, feild_str_gauss, xerr=sx_A, yerr=sigma_B,
             fmt='o', capsize=3, label='data', zorder=3)
ax.plot(xgrid, y_fit_field, '-', lw=1.8,
        label=f"Fit: B = ({m_field:.4g}±{dm_field:.1g})·I + ({c_field:.4g}±{dc_field:.1g})", zorder=4)
ax.set_xlabel("Current I (A)")
ax.set_ylabel("Magnetic field B (Gauss)")
ax.set_title("Magnetic field vs Current")
ax.legend(loc='upper left')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("field_vs_current.png", dpi=300)
plt.close(fig)  # close to free memory

# 2) Height vs Current
fig, ax = plt.subplots(figsize=(7,6))
xgrid2 = np.linspace(np.min(current_h)*0.95, np.max(current_h)*1.05, 300)
y_fit_h = linear_model(fit_height.beta, xgrid2)

ax.errorbar(current_h, delta_h, xerr=sx_h, yerr=np.full_like(delta_h, sigma_h),
             fmt='o', capsize=3, label='data', zorder=3)
ax.plot(xgrid2, y_fit_h, '-', lw=1.8,
        label=f"Fit: Δh = ({m_h:.4g}±{dm_h:.1g})·I + ({c_h:.4g}±{dc_h:.1g})", zorder=4)
ax.set_xlabel("Current I (A)")
ax.set_ylabel(r'$\Delta h$ (mm)')
ax.set_title("Meniscus rise vs Current")
ax.legend(loc='upper left')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("height_vs_current.png", dpi=300)
plt.close(fig)

# -------------------------
# Construct h vs B^2 using predicted B at the currents where h was measured
# Method: use B_pred = m_field * I_h + c_field (from the B vs I fit).
# Propagate uncertainty on B_pred using parameter uncertainties (approx, ignoring covariances).
# Then x = B_pred**2 with sigma_x = 2 * B_pred * sigma_B_pred
# -------------------------
# predicted B at current_h
B_pred_at_current_h = linear_model(fit_field.beta, current_h)

# approximate sigma on predicted B from fit parameter s.d. (neglecting covariance)
sigma_B_pred = np.sqrt((current_h * dm_field)**2 + (dc_field)**2)

# if any B_pred is zero, handle carefully (shouldn't happen here)
sigma_x = 2.0 * B_pred_at_current_h * sigma_B_pred  # σ(H^2) ≈ 2 B σ_B

# prepare arrays for fit: x = B^2, y = delta_h
x_B2 = B_pred_at_current_h**2
y_h = delta_h

# avoid zero uncertainties: force small floor if needed
sigma_x = np.where(sigma_x == 0, 1e-12, sigma_x)
sigma_y = np.full_like(y_h, sigma_h)

# ODR fit of h vs B^2
beta0_hB2 = [1e-6, 0.0]  # initial guess (slope will be small)
fit_h_vs_B2 = odr_linear_fit(x_B2, y_h, sigma_x, sigma_y, beta0_hB2)

m_hB2, c_hB2 = fit_h_vs_B2.beta
dm_hB2, dc_hB2 = fit_h_vs_B2.sd_beta

print("\n=== Fit: Δh = m * B^2 + c  (using B predicted from B(I) fit) ===")
print(f"m = {m_hB2:.6g} ± {dm_hB2:.6g}   (mm / Gauss^2)")
print(f"c = {c_hB2:.6g} ± {dc_hB2:.6g}   (mm)")

# -------------------------
# Plot h vs B^2 and save
# -------------------------
fig, ax = plt.subplots(figsize=(7,6))
# fine grid for plotting fit line
xgrid_B2 = np.linspace(np.min(x_B2)*0.95, np.max(x_B2)*1.05, 300)
y_fit_hB2 = linear_model(fit_h_vs_B2.beta, xgrid_B2)

ax.errorbar(x_B2, y_h, xerr=sigma_x, yerr=sigma_y,
            fmt='o', capsize=3, label='data (B predicted at I_h)', zorder=3)
ax.plot(xgrid_B2, y_fit_hB2, '-', lw=1.8,
        label=f"Fit: Δh = ({m_hB2:.4g}±{dm_hB2:.1g})·B^2 + ({c_hB2:.4g}±{dc_hB2:.1g})", zorder=4)
ax.set_xlabel(r'$B^2$ (Gauss$^2$)')
ax.set_ylabel(r'$\Delta h$ (mm)')
ax.set_title(r'Meniscus rise vs $B^2$ (B predicted at measured currents)')
ax.legend(loc='upper left')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("h_vs_B2.png", dpi=300)
plt.close(fig)

print("\nSaved images: field_vs_current.png, height_vs_current.png, h_vs_B2.png")

