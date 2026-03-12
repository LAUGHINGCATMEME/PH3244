# calculations_final.py
# Cleaned and extended Cornu-method processing
# Uses measurement data from the script (same structure).
#
# Outputs:
#  - df0, df98, df197, df296 with dn / dn' / dn'' and errors
#  - plot and fit for R0 (dn^2 vs n)
#  - for each mass 98,197,296: plot both f(dn',dn) and f(dn'',dn) with fits (R1,R2)
#  - detailed error-propagation prints for each mass
#  - final R0, R1_list, R2_list and mean ± stddev for R1 and R2
#
# NOTE: units are mm (lengths) throughout. least_count in mm.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------- Configuration / constants --------------
# length units: mm, mass units: g (weights), wavelength in mm
L = (30-6)/2*10
g = 9.81 * 1000.0    # acceleration * 1000 to convert g->? (left as your previous) ; not used directly here
w = 41
h = 3.7
lam = 589e-6         # wavelength in mm (589 nm -> mm)
LeastCount = 0.01    # least-count of circular scale in mm

# -------------- Measurement data (keep as in your file) --------------
# (This block came from your uploaded file.) See filecite below. :contentReference[oaicite:1]{index=1}
data = {
    0: {
        "left": {
            "main": [13.0, 13.0 ,13.0 ,12.5, 12.5, 12.5,12.5],
            "circular": [19, 9, 0, 43, 35, 29, 24]
        },
        "right": {
            "main": [13.5, 13.5, 14.0, 14.0, 14.0, 14.0, 14.0],
            "circular": [29, 42, 0, 8, 14, 20, 26]
        }
    },

    98: {
        "left": {
            "main": [13.5, 13.5, 13.5, 13.0, 13.0, 13.0],
            "circular": [18, 7, 0, 43, 38, 32],
        },
        "right": {
            "main": [14.0, 14.0, 14.0, 14.5, 14.5, 14.5],
            "circular": [31, 40, 49, 6, 12, 17],
        },
        "top": {
            "main": [11.0, 10.5, 10.5, 10.5, 10.5, 10.5],
            "circular": [6, 37, 29, 21, 15, 9],
        },
        "bottom": {
            "main": [11.5, 11.5, 11.5, 11.5, 11.5, 12.0],
            "circular": [14, 25, 33, 39, 46, 4],
        },
    },


    197: {
        "left": {
            "main": [13.5, 13.5, 13.0, 13.0, 13.0, 13.0],
            "circular": [13, 3, 44, 37, 32, 27],
        },
        "right": {
            "main": [14.0, 14.0, 14.0, 14.0, 14.5, 14.5],
            "circular": [21, 31, 39, 46, 2, 7],
        },
        "top": {
            "main": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "circular": [40, 28, 20, 13, 7, 1],
        },
        "bottom": {
            "main": [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            "circular": [6, 18, 25, 33, 40, 46],
        },
    },


    296: {
        "left": {
            "main": [12.5, 12.5, 12.5, 12.5, 12.5, 12.5],
            "circular": [43, 32, 23, 16, 11, 6],
        },
        "right": {
            "main": [13.0, 13.5, 13.5, 13.5, 13.5, 13.5],
            "circular": [46, 7, 16, 22, 28, 33],
        },
        "top": {
            "main": [10.0, 10.0, 10.0, 10.0, 10.0, 9.5],
            "circular": [40, 28, 19, 12, 6, 44],
        },
        "bottom": {
            "main": [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            "circular": [6, 17, 25, 34, 40, 46],
        },
    },
}

# -------------- Helper functions --------------
def combine_scale(main, circ):
    """Return main + 0.01*circular (both lists -> numpy array), in mm."""
    return np.array(main, dtype=float) + 0.01 * np.array(circ, dtype=float)

def build_dataframes(data_dict):
    """Produce df0, df98, df197, df296 as requested (shapes: 7x2 and 6x4)."""
    dfs = {}
    for mass, sides in data_dict.items():
        if mass == 0:
            df = pd.DataFrame({
                "left":  combine_scale(sides["left"]["main"],  sides["left"]["circular"]),
                "right": combine_scale(sides["right"]["main"], sides["right"]["circular"]),
            })
        else:
            df = pd.DataFrame({
                "left":   combine_scale(sides["left"]["main"],   sides["left"]["circular"]),
                "right":  combine_scale(sides["right"]["main"],  sides["right"]["circular"]),
                "top":    combine_scale(sides["top"]["main"],    sides["top"]["circular"]),
                "bottom": combine_scale(sides["bottom"]["main"], sides["bottom"]["circular"]),
            })
        dfs[mass] = df
    return dfs

def f_formula(x, y):
    """f(x,y) = x^2 y^2 / (y^2 - x^2). Use exact sign of denominator (we only use slope magnitude later)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = (y**2 - x**2)
    out = np.full_like(x, np.nan, dtype=float)
    mask = (denom != 0)
    out[mask] = (x[mask]**2 * y[mask]**2) / denom[mask]
    return out

def f_error(x, sx, y, sy):
    """
    Propagated error for f = x^2 y^2/(y^2 - x^2)
    sigma_f = (2 x^2 y^2) / (x^2 - y^2)^2 * sqrt( y^4 sx^2 + x^4 sy^2 )
    Note the formula uses x^2 - y^2 in denom squared; use absolute as needed when reporting.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    denom = (x**2 - y**2)              # note sign choice consistent with analytic derivation
    out = np.full_like(x, np.nan, dtype=float)
    mask = denom != 0
    if np.any(mask):
        num_pref = 2.0 * x[mask]**2 * y[mask]**2
        denom_sq = denom[mask]**2
        radicand = (y[mask]**4) * (sx[mask]**2) + (x[mask]**4) * (sy[mask]**2)
        radicand = np.clip(radicand, 0.0, None)
        out[mask] = (num_pref / denom_sq) * np.sqrt(radicand)
    return out

def safe_weighted_linear_fit(x, y, yerr, yerr_min=0, w_max=0):
    """
    Robust wrapper for a weighted linear fit.
    - x, y, yerr: 1D numpy arrays.
    - yerr_min: minimum allowed error to avoid infinite weights.
    - w_max: cap for the returned weights to avoid overflow inside LAPACK.
    Returns (slope, intercept, slope_err) or (nan, nan, nan) if not enough good points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    # filter finite rows
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    if np.count_nonzero(finite_mask) < 2:
        return np.nan, np.nan, np.nan

    x = x[finite_mask]
    y = y[finite_mask]
    yerr = yerr[finite_mask]

    # enforce minimum error (floor) and cap weights
    yerr_clipped = np.maximum(yerr, yerr_min)
    w = 1.0 / yerr_clipped
    w = np.minimum(w, w_max)

    # if weights are nearly constant (no meaningful weighting), fall back to unweighted
    if np.allclose(w, w.mean(), rtol=1e-3, atol=0.0):
        # ordinary least squares via np.polyfit
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope, intercept = coeffs
        slope_err = np.sqrt(cov[0,0])
        return slope, intercept, slope_err

    # attempt weighted polyfit; guard against exceptions
    try:
        coeffs, cov = np.polyfit(x, y, 1, w=w, cov=True)
        slope, intercept = coeffs
        slope_err = np.sqrt(cov[0,0])
        return slope, intercept, slope_err
    except Exception as e:
        # fallback: unweighted fit
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope, intercept = coeffs
        slope_err = np.sqrt(cov[0,0])
        return slope, intercept, slope_err

# -------------- Build dataframes and add dn columns --------------
dfs = build_dataframes(data)
df0   = dfs[0].copy()
df98  = dfs[98].copy()
df197 = dfs[197].copy()
df296 = dfs[296].copy()

# error scalar (mm)
dn_err_val = LeastCount / np.sqrt(6.0)

# m=0: dn = right - left
df0["dn"] = df0["right"] - df0["left"]
df0["dn_err"] = dn_err_val

# nonzero: dn' = right-left; dn'' = bottom-top and assign errors
for df in (df98, df197, df296):
    df["dn'"] = df["right"] - df["left"]
    df["dn'_err"] = dn_err_val
    df["dn''"] = df["bottom"] - df["top"]
    df["dn''_err"] = dn_err_val

# print shapes for verification
print("Shapes: df0", df0.shape, "df98", df98.shape, "df197", df197.shape, "df296", df296.shape)

# -------------- Compute R0 from df0 (dn^2 vs n) --------------
# dn^2 and propagated error:
df0["dn2"] = df0["dn"]**2
# delta(dn^2) approx sqrt( (2*dn*sd_dn)^2 ) = 2*|dn|*sd_dn
df0["dn2_err"] = 2.0 * np.abs(df0["dn"]) * df0["dn_err"]

n0 = np.arange(1, len(df0)+1)
y0 = df0["dn2"].to_numpy()
y0err = df0["dn2_err"].to_numpy()

# weighted fit
slope0, intercept0, slope0_err = safe_weighted_linear_fit(n0, y0, y0err)
R0 = slope0 / (4.0 * lam)
R0_err = slope0_err / (4.0 * lam)

# plot R0 fit
plt.figure(figsize=(7,5))
plt.errorbar(n0, y0, yerr=y0err, fmt='o', capsize=3, label=r'$d_n^2$ (data)')
xfit = np.linspace(n0.min(), n0.max(), 200)
yfit = slope0 * xfit + intercept0
plt.plot(xfit, yfit, '-', label=f'weighted fit (slope={slope0:.3e}±{slope0_err:.3e})')
plt.xlabel(r'$n$')
plt.ylabel(r'$d_n^2$ (mm$^2$)')
plt.title(r'$d_n^2$ vs $n$ and fit for $R_0$')
plt.grid(True)
plt.legend()
plt.gca().text(0.02, 0.95, rf'$R_0={R0:.3e}\pm{R0_err:.3e}\ \mathrm{{mm}}$', transform=plt.gca().transAxes, va='top')
plt.tight_layout()
plt.show()

print("\n===== R0 fit results =====")
print(f"slope = {slope0:.6g} ± {slope0_err:.6g}  (units mm^2 per fringe)")
print(f"R0 = {R0:.6g} ± {R0_err:.6g} mm")

# -------------- Compute f and errors, then R1/R2 for each nonzero mass --------------
# we will collect R1_list and R2_list for mean/std later
R1_list = []
R2_list = []
R1_err_list = []
R2_err_list = []

# We'll use df0's dn and dn_err rows aligned to the first 6 rows for df98/197/296
dn_ref = df0["dn"].to_numpy()
dn_ref_err = df0["dn_err"].to_numpy()

def plot_series_separately(n, y, yerr, fit_mask, slope, intercept, slope_err,
                           label, mass, removed_indices, filename=None):
    """
    Plot one series (data + errorbars), mark removed points, draw fit line over fit_mask range,
    annotate with slope and R, and optionally save to filename.
    - n: integer index array (1..N)
    - y, yerr: arrays
    - fit_mask: boolean array (True -> used in fit)
    - slope/intercept/slope_err: from fit (may be nan)
    - label: string for plot title/legend, e.g. "f(dn',dn)"
    - mass: numeric mass for plot title
    - removed_indices: list/array of 1-based indices removed from fit
    - filename: if provided, save figure to this path (e.g. 'mass98_f1.png')
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.6,4.2))
    # data + errorbars
    plt.errorbar(n, y, yerr=yerr, fmt='o', capsize=3, label=f'{label} data')
    # hollow markers for removed points (if any)
    if len(removed_indices)>0:
        idx0 = np.array(removed_indices)  # already 1-based from earlier code
        plt.scatter(idx0, y[idx0-1], facecolors='none', edgecolors='C1', s=80,
                    label='removed (for fit)')
    # plot fit line only over fit_mask range (if slope finite)
    if np.isfinite(slope):
        xfit = np.linspace(n[fit_mask].min(), n[fit_mask].max(), 200)
        yfit = slope * xfit + intercept
        plt.plot(xfit, yfit, '-', label=f'fit slope={slope:.3e}±{slope_err:.3e}')
        # annotate with R = slope/(4L)
        R_val = slope / (4.0 * L)
        R_err = slope_err / (4.0 * L) if np.isfinite(slope_err) else np.nan
        plt.gca().text(0.02, 0.92, rf'$R={R_val:.3e}\pm{R_err:.3e}\ \mathrm{{mm}}$',
                       transform=plt.gca().transAxes, va='top')
    # labels, grid, legend
    plt.xlabel(r'$n$')
    plt.ylabel(r'$f$ (mm$^2$)')
    plt.title(f'mass = {mass} g — {label}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()




def process_mass(mass, df, drop_first_for_fit=False):
    """
    Compute f and errors, print error-propagation tables, do combined plot of f1 and f2 with fits,
    return (R1, R1_err, R2, R2_err).
    """
    n = np.arange(1, len(df)+1)
    # x = dn' ; sx = dn'_err
    x1 = df["dn'"].to_numpy(dtype=float)
    sx1 = df["dn'_err"].to_numpy(dtype=float)
    # x2 = dn'' ; sx2 = dn''_err
    x2 = df["dn''"].to_numpy(dtype=float)
    sx2 = df["dn''_err"].to_numpy(dtype=float)
    # y = dn (from df0 reference, trimmed to same length)
    y = dn_ref[:len(df)]
    sy = dn_ref_err[:len(df)]

    # compute f values and errors
    f1 = f_formula(x1, y)         # f(dn', dn)
    f1_err = f_error(x1, sx1, y, sy)
    f2 = -f_formula(x2, y)         # f(dn'', dn)
    f2_err = f_error(x2, sx2, y, sy)

    # Print tidy table for error propagation (one table per f)
    # This helps with later manual propagation steps
    print(f"\n--- mass = {mass} g : detailed error-propagation table (f(dn',dn)) ---")
    print(" n |   x=dn'   | sx |   y=dn   | sy | denom (y^2 - x^2) | radicand | f | f_err")
    for i in range(len(n)):
        denom = (y[i]**2 - x1[i]**2) if (not np.isnan(x1[i]) and not np.isnan(y[i])) else np.nan
        rad = (y[i]**4) * (sx1[i]**2) + (x1[i]**4) * (sy[i]**2)
        print(f"{n[i]:2d} | {x1[i]:8.4f} | {sx1[i]:6.4f} | {y[i]:8.4f} | {sy[i]:6.4f} | {denom:12.4e} | {rad:9.4e} | {f1[i]:9.4e} | {f1_err[i]:9.4e}")

    print(f"\n--- mass = {mass} g : detailed error-propagation table (f(dn'',dn)) ---")
    print(" n |   x=dn''  | sx |   y=dn   | sy | denom (y^2 - x^2) | radicand | f | f_err")
    for i in range(len(n)):
        denom = (y[i]**2 - x2[i]**2) if (not np.isnan(x2[i]) and not np.isnan(y[i])) else np.nan
        rad = (y[i]**4) * (sx2[i]**2) + (x2[i]**4) * (sy[i]**2)
        print(f"{n[i]:2d} | {x2[i]:8.4f} | {sx2[i]:6.4f} | {y[i]:8.4f} | {sy[i]:6.4f} | {denom:12.4e} | {rad:9.4e} | {f2[i]:9.4e} | {f2_err[i]:9.4e}")

    # Prepare masks for fit: optionally drop first data point for mass==98 as requested
    if drop_first_for_fit:
        fit_mask = np.ones(len(n), dtype=bool)
        fit_mask[0] = False
        removed_indices = np.where(~fit_mask)[0] + 1
    else:
        fit_mask = np.ones(len(n), dtype=bool)
        removed_indices = []

    # For each dataset (f1,f2) do weighted linear fit and obtain R
    # remove rows with invalid numbers
    good1 = np.isfinite(f1) & np.isfinite(f1_err)
    good2 = np.isfinite(f2) & np.isfinite(f2_err)
    
    fit_mask_comb1 = fit_mask & good1
    fit_mask_comb2 = fit_mask & good2
    
    # ---- Fit for R1 ----
    if np.count_nonzero(fit_mask_comb1) >= 2:
        slope1, intercept1, slope1_err = safe_weighted_linear_fit(
            n[fit_mask_comb1], f1[fit_mask_comb1], f1_err[fit_mask_comb1]
        )
        R1 = slope1 / (4.0 * lam)
        R1_err = slope1_err / (4.0 * lam)
    else:
        slope1 = intercept1 = slope1_err = R1 = R1_err = np.nan
    
    
    # ---- Fit for R2 ----
    if np.count_nonzero(fit_mask_comb2) >= 2:
        slope2, intercept2, slope2_err = safe_weighted_linear_fit(
            n[fit_mask_comb2], f2[fit_mask_comb2], f2_err[fit_mask_comb2]
        )
        R2 = slope2 / (4.0 * lam)
        R2_err = slope2_err / (4.0 * lam)
    else:
        slope2 = intercept2 = slope2_err = R2 = R2_err = np.nan
    # Plot both series and fits together (same axes)
    plt.figure(figsize=(7.2,4.4))
    plt.errorbar(n, f1, yerr=f1_err, fmt='o', capsize=3, label=r"$f(dn',dn)$ data")
    plt.errorbar(n, f2, yerr=f2_err, fmt='s', capsize=3, label=r"$f(dn'',dn)$ data")
    # Mark removed points (if any) with hollow marker (plot them again with different style)
    if len(removed_indices) > 0:
        plt.scatter(removed_indices, f1[~fit_mask], facecolors='none', edgecolors='C0', s=80, label='removed (for fit)')
        plt.scatter(removed_indices, f2[~fit_mask], facecolors='none', edgecolors='C1', s=80)

    # plot fit lines (over fit_mask range)
    xfit = np.linspace(n[fit_mask].min(), n[fit_mask].max(), 200)
    yfit1 = slope1 * xfit + intercept1
    yfit2 = slope2 * xfit + intercept2
    plt.plot(xfit, yfit1, '-', label=f'fit f1 slope={slope1:.3e}±{slope1_err:.3e}')
    plt.plot(xfit, yfit2, '--', label=f'fit f2 slope={slope2:.3e}±{slope2_err:.3e}')

    plt.xlabel(r'$n$')
    plt.ylabel(r'$f$ (mm$^2$)')
    plt.title(f"mass = {mass} g : fits for R1 and R2 (combined)")
    plt.grid(True)
    plt.legend()
    plt.gca().text(0.02, 0.95, rf"$R_1={R1:.3e}\pm{R1_err:.3e}$" + "\n" + rf"$R_2={R2:.3e}\pm{R2_err:.3e}$",
                   transform=plt.gca().transAxes, va='top')
    plt.tight_layout()
    plt.show()

    # Print numeric summary
    print(f"\nMass {mass} g summary:")
    print(f"  f1 fit: slope = {slope1:.6g} ± {slope1_err:.6g} -> R1 = {R1:.6g} ± {R1_err:.6g} (mm)")
    print(f"  f2 fit: slope = {slope2:.6g} ± {slope2_err:.6g} -> R2 = {R2:.6g} ± {R2_err:.6g} (mm)")
    if len(removed_indices) > 0:
        print(f"  (note: first data point index {removed_indices.tolist()} was removed from fits)")

    return R1, R1_err, R2, R2_err

# Process each mass. For mass=98 remove first point while fitting (per your request).
for mass, df in ((98, df98), (197, df197), (296, df296)):
    drop_first = (mass == 98)
    R1, R1_err, R2, R2_err = process_mass(mass, df, drop_first_for_fit=drop_first)
    R1_list.append(R1)
    R1_err_list.append(R1_err)
    R2_list.append(R2)
    R2_err_list.append(R2_err)

# Compute overall statistics (mean and sample standard deviation) for R1 and R2
R1_arr = np.array(R1_list, dtype=float)
R2_arr = np.array(R2_list, dtype=float)
# ignore nan entries if any
R1_valid = R1_arr[~np.isnan(R1_arr)]
R2_valid = R2_arr[~np.isnan(R2_arr)]

R1_mean = R1_valid.mean() if R1_valid.size>0 else np.nan
R1_std  = R1_valid.std(ddof=1) if R1_valid.size>1 else np.nan
R2_mean = R2_valid.mean() if R2_valid.size>0 else np.nan
R2_std  = R2_valid.std(ddof=1) if R2_valid.size>1 else np.nan

print("\n\n========== Final Results ==========")
print(f"R0 = {R0:.6g} ± {R0_err:.6g} mm")
for i, mass in enumerate([98,197,296]):
    print(f"mass {mass} g: R1 = {R1_list[i]:.6g} ± {R1_err_list[i]:.6g} mm ; R2 = {R2_list[i]:.6g} ± {R2_err_list[i]:.6g} mm")
print(f"\nR1 mean (over masses) = {R1_mean:.6g} mm ; std.dev = {R1_std:.6g} mm")
print(f"R2 mean (over masses) = {R2_mean:.6g} mm ; std.dev = {R2_std:.6g} mm")

# ---------- Young's modulus (skip 98 g) ----------

valid_masses = [197,296]

Y_vals = []
Y_errs = []

for i,m in enumerate([98,197,296]):

    if m not in valid_masses:
        continue

    R1 = R1_list[i]
    R1_err = R1_err_list[i]

    Y = (12 * m * g * L * R1) / (w * h**3)

    dY_dR1 = (12 * m * g * L) / (w * h**3)
    Y_err = abs(dY_dR1) * R1_err

    Y_vals.append(Y)
    Y_errs.append(Y_err)

    print(f"\nMass = {m} g")
    print(f"R1 = {R1:.5e} ± {R1_err:.5e}")
    print(f"Young's modulus Y = {Y:.5e} ± {Y_err:.5e}")


# ---------- Poisson ratio ----------

nu_vals = []
nu_errs = []

for i,m in enumerate([98,197,296]):

    if m not in valid_masses:
        continue

    R1 = R1_list[i]
    R2 = R2_list[i]
    R1_err = R1_err_list[i]
    R2_err = R2_err_list[i]

    nu = R1 / R2

    nu_err = np.sqrt(
        (R1_err / R2)**2 +
        ((R1 * R2_err) / (R2**2))**2
    )

    nu_vals.append(nu)
    nu_errs.append(nu_err)

    print(f"\nMass = {m} g")
    print(f"Poisson ratio ν = {nu:.5f} ± {nu_err:.5f}")


# ---------- Final results ----------

Y_mean = np.mean(Y_vals)
Y_std = np.std(Y_vals, ddof=1)

nu_mean = np.mean(nu_vals)
nu_std = np.std(nu_vals, ddof=1)

print("\n========== FINAL RESULTS ==========")
print(f"Young's modulus Y = ({Y_mean:.5e} ± {Y_std:.5e})")
print(f"Poisson ratio ν = {nu_mean:.5f} ± {nu_std:.5f}")
