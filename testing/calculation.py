import numpy as np
import labtools
labtools.import_data()     
labtools.unit_check()      
labtools.unit_converter()  
labtools.make_parsed_data()

#calculate distance form lhs rhs
def D_func(rhs, lhs):
    return rhs - lhs

def D_jac(rhs, lhs):
    rhs = np.asarray(rhs, dtype=float)
    lhs = np.asarray(lhs, dtype=float)
    return np.vstack((np.ones_like(rhs), -np.ones_like(lhs))).T

j =  'V = 100 V, cross hair at diameter with leveling'

# I const 4 plots 
for OBS_NAME in [
'I = 1.3 Amp, cross hair at diameter with leveling',
'I = 1 Amp, cross hair at chord no leveling',
'I = 1 Amp, cross hair at diameter no leveling',
'I = 1 Amp, cross hair at diameter with leveling']:
    
    # create a new row for the D (diameter)
    labtools.propagate_function(
        OBS_NAME,
        input_cols=["rhs", "lhs"],
        func=D_func,
        new_col="D",
    #    jacobian=D_jac,
        overwrite=True
    )
    # create d^2
    labtools.propagate_square(OBS_NAME, "D", out_col="D_sq", overwrite=True)
    
    
    fit_result = labtools.fit_odr(
        OBS_NAME,
        x_col="Voltage",
        y_col="D_sq",
        x_err_col="Voltage_err",
        y_err_col="D_sq_err"
    )
    
    plot_path = labtools.plot_with_fit(
        OBS_NAME,
        x_col="Voltage",
        y_col="D_sq",
        x_err_col="Voltage_err",
        y_err_col="D_sq_err",
        fit_result=fit_result,
        title=rf"Voltage (V) vs Distance$^2$ (m$^2$) | {OBS_NAME.split(',')[0]}"
    )
    
    
    
    print("\nODR fit summary (Voltage vs D^2):")
    for i, (p, pe) in enumerate(zip(fit_result["params"], fit_result["param_err"])):
        print(f"  p[{i}] = {p:.6g} ± {pe:.6g}")
    print(f"  reduced chi2 = {fit_result.get('res_var'):.6g}")
    print(f"\nSaved figure: {plot_path}")
#!/usr/bin/env python3
# calculation.py  (modified)
import numpy as np
import os
import labtools

# --- config ---
OBS_NAME = "I = 1A combined"   # basename of uploaded CSV (without .csv)
GROUP_BY = "Voltage"           # column to collapse on
Y_COL = "D"              # dependent variable to fit
PARSED_DIR = "parsed_data"
IMAGE_DIR = "images"

# --- 1) ensure pipeline ---
labtools.import_data()
labtools.unit_check()
labtools.unit_converter()
labtools.make_parsed_data()

# --- 2) collapse repeated measurements by Voltage, combine uncertainties ---
# We explicitly request processing of both Voltage and Current so the function
# computes combined errors for Current. (group_by is Voltage; we will compute Voltage_err below)
combined = labtools.collapse_repeats_and_combine_errors(
    OBS_NAME,
    group_by=GROUP_BY,
    cols=[GROUP_BY, Y_COL],        # include GROUP_BY and Y column to ensure both are present
    random_method="sem",           # standard error of the mean for random component
    reduce_inst=True,              # reduce instrument uncertainty by sqrt(n) for the mean
    save=True,
    out_name=f"{labtools._sanitize_name(OBS_NAME)}__collapsed_by_{labtools._sanitize_name(GROUP_BY)}.csv",
    parsed_dir=PARSED_DIR
)

# combined is a DataFrame with columns such as:
#   Voltage, Current, Current_err, n
# (depending on your collapse function's ordering). Inspect to confirm.
print("\nCollapsed data (first 10 rows):")
print(combined.head(10))

# --- 3) compute Voltage_err per row from instrument least-count & distribution token ---
# We'll compute instrument sigma per point and (optionally) reduce it by sqrt(n).
# If OBS_LEAST_COUNTS or OBS_DISTRIBUTION missing for Voltage, fallback to 0.
vol_errs = []
if OBS_NAME in labtools.OBS_LEAST_COUNTS and GROUP_BY in labtools.OBS_LEAST_COUNTS[OBS_NAME]:
    lc_vol = float(labtools.OBS_LEAST_COUNTS[OBS_NAME][GROUP_BY])
    # get distribution token
    tok = ""
    if "OBS_DISTRIBUTION" in globals() or "OBS_DISTRIBUTION" in labtools.__dict__:
        # OBS_DISTRIBUTION may be global in labtools
        dist_map = getattr(labtools, "OBS_DISTRIBUTION", {})
        tok = dist_map.get(OBS_NAME, {}).get(GROUP_BY, "") if isinstance(dist_map.get(OBS_NAME, {}), dict) else ""
    # compute base instrument sigma
    if tok == "V":
        base_sigma = lc_vol / np.sqrt(24.0)
    else:
        base_sigma = lc_vol / np.sqrt(12.0)
    # now per-row reduce by sqrt(n)
    for idx, row in combined.iterrows():
        n = int(row.get("n", 1))
        sigma_inst_used = base_sigma / np.sqrt(n) if (n > 1) else base_sigma
        vol_errs.append(sigma_inst_used)
else:
    # fallback: no least-count information — set zero (ODR will treat x errors as zero)
    vol_errs = [0.0] * len(combined)

# attach Voltage_err into combined DataFrame
combined[f"{GROUP_BY}_err"] = vol_errs

# Save combined augmented DataFrame (optional redundant save)
out_comb_name = f"{labtools._sanitize_name(OBS_NAME)}__combined_for_fit.csv"
os.makedirs(PARSED_DIR, exist_ok=True)
outpath = os.path.join(os.getcwd(), PARSED_DIR, out_comb_name)
combined.to_csv(outpath, index=False, float_format="%.12g")
print(f"\nSaved augmented combined data to: {outpath}")

# --- 4) make this combined dataset available to labtools fitter ---
combined_obs_name = f"{OBS_NAME} (collapsed)"
labtools.OBSERVATIONS[combined_obs_name] = combined.rename(columns={f"{Y_COL}_mean": Y_COL}) if f"{Y_COL}_mean" in combined.columns else combined.copy()
# Ensure error columns are named as expected by fit_odr (e.g., Current_err, Voltage_err)
# If collapse function used "Current_err" already, we're fine; if not, try to create it.
if f"{Y_COL}_err" not in labtools.OBSERVATIONS[combined_obs_name].columns:
    # try common alternatives
    if f"{Y_COL}_sem" in combined.columns:
        labtools.OBSERVATIONS[combined_obs_name][f"{Y_COL}_err"] = combined[f"{Y_COL}_sem"].values
    elif f"{Y_COL}_std" in combined.columns:
        labtools.OBSERVATIONS[combined_obs_name][f"{Y_COL}_err"] = combined[f"{Y_COL}_std"].values
    else:
        # last-resort: zeros
        labtools.OBSERVATIONS[combined_obs_name][f"{Y_COL}_err"] = np.zeros(len(combined))

# Set Voltage_err if not already present
if f"{GROUP_BY}_err" not in labtools.OBSERVATIONS[combined_obs_name].columns:
    labtools.OBSERVATIONS[combined_obs_name][f"{GROUP_BY}_err"] = combined[f"{GROUP_BY}_err"].values

# Provide minimal units metadata for plotting/fit annotations (optional, adjust if needed)
labtools.OBS_UNITS[combined_obs_name] = labtools.OBS_UNITS.get(OBS_NAME, {})
labtools.OBS_LEAST_COUNTS[combined_obs_name] = labtools.OBS_LEAST_COUNTS.get(OBS_NAME, {})

# --- 5) Fit using ODR (accounts for x & y errors) ---
fit_result = labtools.fit_odr(
    combined_obs_name,
    x_col=GROUP_BY,
    y_col=Y_COL,
    x_err_col=f"{GROUP_BY}_err",
    y_err_col=f"{Y_COL}_err"
)

print("\nODR fit summary (Combined collapsed dataset):")
for i, (p, pe) in enumerate(zip(fit_result["params"], fit_result["param_err"])):
    print(f"  p[{i}] = {p:.6g} ± {pe:.6g}")
print(f"  reduced chi2 = {fit_result.get('res_var'):.6g}")

# --- 6) Plot and save fit figure ---
plot_path = labtools.plot_with_fit(
    combined_obs_name,
    x_col=GROUP_BY,
    y_col=Y_COL,
    x_err_col=f"{GROUP_BY}_err",
    y_err_col=f"{Y_COL}_err",
    fit_result=fit_result,
    title=rf"{GROUP_BY} (V) vs {Y_COL} (A) — combined & collapsed | {OBS_NAME}"
)
print(f"\nSaved figure: {plot_path}")

