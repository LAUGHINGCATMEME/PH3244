# labtools.py
"""
labtools.py — utilities for lab experiment data ingestion, unit conversion,
error propagation, ODR fitting and plotting.

Core pipeline (use in this order):
  import_data() -> unit_check() -> unit_converter() -> make_parsed_data()

New capabilities:
  - Generic error-propagation helpers (propagate_function + convenience wrappers)
  - ODR fitting (fit_odr) and plotting (plot_with_fit)
  - Images directory creation (ensure_image_dir)
Notes:
  - Parsed CSVs saved to parsed_data/<OBS_NAME>_SI_PARSED.csv (float_format='%.12g')
  - Least-counts are converted to standard uncertainty using sigma = lc / sqrt(12)
"""

from typing import Dict, Tuple, Callable, Sequence, Optional
import os
import re
import pandas as pd
import math
import inspect
import numpy as np
import matplotlib.pyplot as plt

# Try to import scipy.odr lazily when fit_odr is called (we check at runtime)

# ---- Module-level storage ----
OBSERVATIONS: Dict[str, pd.DataFrame] = {}     # observation_name -> DataFrame (current state)
OBS_UNITS: Dict[str, Dict[str, str]] = {}      # observation_name -> {colname: original_unit}
OBS_LEAST_COUNTS: Dict[str, Dict[str, float]] = {}  # observation_name -> {colname: least_count_in_original_unit}

# ---- Unit dictionary (same approach as before) ----
UNIT_CONVERSIONS: Dict[str, Tuple[float, str, Optional[str]]] = {
    # Length
    "mm":   (1e-3,    "m", None),
    "cm":   (1e-2,    "m", None),
    "m":    (1.0,     "m", None),
    "km":   (1000.0,  "m", None),
    "inch": (0.0254,  "m", None),
    "in":   (0.0254,  "m", None),
    "ft":   (0.3048,  "m", None),
    # Mass
    "mg":   (1e-6,    "kg", None),
    "g":    (1e-3,    "kg", None),
    "kg":   (1.0,     "kg", None),
    # Time
    "s":    (1.0,     "s", None),
    "sec":  (1.0,     "s", None),
    "ms":   (1e-3,    "s", None),
    "us":   (1e-6,    "s", None),
    # Current & charge
    "A":    (1.0,     "A", None),
    "mA":   (1e-3,    "A", None),
    "uA":   (1e-6,    "A", None),
    "C":    (1.0,     "C", None),
    # Voltage & resistance
    "V":    (1.0,     "V", None),
    "mV":   (1e-3,    "V", None),
    "ohm":  (1.0,     "ohm", None),
    "Ω":    (1.0,     "ohm", None),
    "kΩ":   (1e3,     "ohm", None),
    # Capacitance / inductance
    "F":    (1.0,     "F", None),
    "uF":   (1e-6,    "F", None),
    "H":    (1.0,     "H", None),
    "mH":   (1e-3,    "H", None),
    # Energy / units
    "J":    (1.0,     "J", None),
    "eV":   (1.602176634e-19, "J", None),
    "W":    (1.0,     "W", None),
    "Hz":   (1.0,     "Hz", None),
    # Temperature
    "K":    (1.0,     "K", None),
    "C":    (1.0,     "K", "C2K"),  # Celsius needs +273.15
    "°C":   (1.0,     "K", "C2K"),
    # Magnetic
    "T":    (1.0,     "T", None),
    "mT":   (1e-3,    "T", None),
    "G":    (1e-4,    "T", None),
    # A few more
    "mol":  (1.0,     "mol", None),
    "g/cm^3": (1000.0, "kg/m^3", None),
    "kV":   (1e3,     "V", None),
    "MΩ":   (1e6,     "ohm", None),
}

# small synonyms
UNIT_CONVERSIONS.setdefault("um", (1e-6, "m", None))
UNIT_CONVERSIONS.setdefault("μm", (1e-6, "m", None))
UNIT_CONVERSIONS.setdefault("degC", (1.0, "K", "C2K"))

# ---- Helpers ----
HEADER_RE = re.compile(r'^\s*(?P<name>.*?)\s*\(\s*(?P<unit>.+?)\s*\)\s*$')

def _sanitize_name(name: str) -> str:
    s = re.sub(r'\s+', '_', name.strip())
    s = re.sub(r'[^\w_]', '_', s)
    if re.match(r'^\d', s):
        s = '_' + s
    return s

def _parse_header_cell(cell: str):
    m = HEADER_RE.match(str(cell))
    if not m:
        raise ValueError(f"Header cell '{cell}' is not of the form 'Name (unit)'.")
    return m.group('name').strip(), m.group('unit').strip()

def _get_conversion_for_unit(unit: str):
    if unit is None:
        return None
    u = unit.strip().replace('°', '').replace('μ', 'u')
    if u in UNIT_CONVERSIONS:
        return UNIT_CONVERSIONS[u]
    if u.lower() in UNIT_CONVERSIONS:
        return UNIT_CONVERSIONS[u.lower()]
    # basic prefix handling
    m = re.match(r'^(?P<prefix>k|M|m|c|u|n|p)(?P<base>.+)$', u)
    if m:
        pf = {"k":1e3, "M":1e6, "m":1e-3, "c":1e-2, "u":1e-6, "n":1e-9, "p":1e-12}.get(m.group('prefix'))
        base = m.group('base')
        if pf is not None:
            if base in UNIT_CONVERSIONS:
                bf, bsi, bflag = UNIT_CONVERSIONS[base]
                return (pf * bf, bsi, bflag)
            if base.lower() in UNIT_CONVERSIONS:
                bf, bsi, bflag = UNIT_CONVERSIONS[base.lower()]
                return (pf * bf, bsi, bflag)
    return None

# ---- Pipeline: import, unit check, convert, parse ----
def import_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Read CSV files from `data_dir`, validate format, and import into module stores.

    New behaviour:
      - Detects an optional third-row distribution flag (tokens like 'V' or 'R').
        If present it will be stored in OBS_DISTRIBUTION[obs_name].
      - If no distribution row is present, OBS_DISTRIBUTION[obs_name] will contain
        empty-string entries for each column (meaning "default" behaviour).
    Strict CSV expectations (still):
      - Row 0: header cells of the form "Name (unit)".
      - Row 1: least-count numeric row (one entry per header column).
      - Optional Row 2: distribution tokens (e.g., V or R). If present, data start at row 3.
      - Remaining rows: numeric datapoints (one row per measurement).
    """
    cwd = os.getcwd()
    target = os.path.join(cwd, data_dir)
    if not os.path.isdir(target):
        raise ValueError(f"Data directory not found: {target}")

    csv_files = [f for f in os.listdir(target) if f.lower().endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {target}")

    # Clear previous
    OBSERVATIONS.clear()
    OBS_UNITS.clear()
    OBS_LEAST_COUNTS.clear()
    # OBS_DISTRIBUTION is expected to exist module-level; initialize/clear it here
    try:
        OBS_DISTRIBUTION.clear()
    except NameError:
        # If OBS_DISTRIBUTION not defined, create it
        globals()["OBS_DISTRIBUTION"] = {}
        OBS_DISTRIBUTION.clear()

    caller_globals = None
    try:
        frm = inspect.stack()[1].frame
        caller_globals = frm.f_globals
    except Exception:
        caller_globals = None

    for fname in csv_files:
        full = os.path.join(target, fname)
        obs_name = os.path.splitext(fname)[0]  # e.g., "For I = 1 Amp"
        sanitized = _sanitize_name(obs_name)

        # Read raw (no header) so we can inspect rows exactly
        raw = pd.read_csv(full, header=None, dtype=str, keep_default_na=False)
        if raw.shape[0] < 3:
            raise ValueError(f"CSV '{fname}' must have at least 3 rows: header, least counts, and >=1 datapoint.")

        header_row = raw.iloc[0].tolist()
        least_row = raw.iloc[1].tolist()

        # Detect optional distribution row (third CSV row)
        dist_row = None
        data_start_idx = 2  # default data starts at row index 2
        if raw.shape[0] >= 4:
            third = raw.iloc[2].tolist()
            # decide if third row is a distribution row by checking for alphabetic tokens
            alpha_count = sum(1 for c in third if re.search(r'[A-Za-z]', str(c)))
            # if at least one alphabetic token present (user-specified tokens like V/R), treat as distribution row
            if alpha_count >= 1:
                dist_row = third
                data_start_idx = 3
        data_rows = raw.iloc[data_start_idx:].copy()

        # Parse header -> (colnames, units)
        colnames = []
        units = {}
        for idx, cell in enumerate(header_row):
            cell_str = str(cell)
            try:
                varname, unit = _parse_header_cell(cell_str)
            except ValueError as e:
                raise ValueError(f"In file '{fname}', header column {idx+1}: {e}")
            if varname in colnames:
                raise ValueError(f"In file '{fname}', duplicate column name '{varname}'.")
            colnames.append(varname)
            units[varname] = unit

        # Parse least counts: must all be numeric and match column count
        if len(least_row) != len(colnames):
            raise ValueError(f"In file '{fname}', least count row has {len(least_row)} columns but header has {len(colnames)}.")
        least_counts = {}
        for i, val in enumerate(least_row):
            c = str(val).strip()
            try:
                f = float(c)
            except Exception:
                raise ValueError(f"In file '{fname}', least count for column '{colnames[i]}' is not numeric: '{c}'")
            least_counts[colnames[i]] = f

        # Parse distribution row if present and well-formed; otherwise default to ""
        dist_map = {}
        if dist_row and len(dist_row) == len(colnames):
            for i, tok in enumerate(dist_row):
                t = str(tok).strip()
                tnorm = t.upper() if t else ""
                # Accept only 'V' or 'R' (empty means default). Unknown tokens become "" (default).
                dist_map[colnames[i]] = tnorm if tnorm in ("V", "R") else ""
        else:
            for c in colnames:
                dist_map[c] = ""

        # Validate there is at least one data row after accounting for optional distribution row
        if data_rows.shape[0] < 1:
            raise ValueError(f"In file '{fname}', no data rows found after header/least-count{', distribution row' if dist_row else ''}.")

        # Ensure column counts match
        if data_rows.shape[1] != len(colnames):
            raise ValueError(f"In file '{fname}', data rows have {data_rows.shape[1]} columns but header has {len(colnames)}.")

        # Convert data rows to numeric DataFrame
        numeric_df = data_rows.copy()
        numeric_df.columns = colnames
        for col in colnames:
            try:
                numeric_df[col] = pd.to_numeric(numeric_df[col].astype(str).str.strip(), errors='raise')
            except Exception:
                raise ValueError(f"In file '{fname}', data column '{col}' contains non-numeric entries; can't import.")
        numeric_df.reset_index(drop=True, inplace=True)

        # Store into module-level structures
        OBSERVATIONS[obs_name] = numeric_df.copy()
        OBS_UNITS[obs_name] = units.copy()
        OBS_LEAST_COUNTS[obs_name] = least_counts.copy()
        OBS_DISTRIBUTION[obs_name] = dist_map.copy()

        # Expose as attribute on this module
        setattr(__import__(__name__), sanitized, OBSERVATIONS[obs_name])

        # Also, best-effort: set into caller globals with sanitized name
        if caller_globals is not None:
            try:
                caller_globals[sanitized] = OBSERVATIONS[obs_name]
            except Exception:
                pass

    return OBSERVATIONS


def unit_check():
    if not OBS_UNITS:
        raise ValueError("No observations imported. Run import_data() first.")
    errors = []
    for obs_name, unitmap in OBS_UNITS.items():
        for col, unit in unitmap.items():
            conv = _get_conversion_for_unit(unit)
            if conv is None:
                errors.append((obs_name, col, unit))
    if errors:
        lines = ["Unknown units detected:"]
        for obs, col, unit in errors:
            lines.append(f"  Observation '{obs}', column '{col}': unit '{unit}' not recognized.")
        lines.append("Add the missing unit(s) to labtools.UNIT_CONVERSIONS or fix the CSV header.")
        raise ValueError("\n".join(lines))
    return True

def unit_converter():
    if not OBSERVATIONS:
        raise ValueError("No observations imported. Run import_data() first.")
    unit_check()

    for obs_name, df in list(OBSERVATIONS.items()):
        units = OBS_UNITS[obs_name]
        lcounts = OBS_LEAST_COUNTS[obs_name]

        df_si = df.copy().astype(float)
        new_units = {}
        new_lcounts = {}

        for col in df_si.columns:
            orig_unit = units[col]
            conv = _get_conversion_for_unit(orig_unit)
            if conv is None:
                raise ValueError(f"Unit resolution failed for '{orig_unit}'.")
            factor, si_unit, special = conv
            if special == "C2K":
                df_si[col] = df_si[col] + 273.15
                new_lcounts[col] = float(lcounts[col])
                new_units[col] = si_unit
            else:
                df_si[col] = df_si[col].astype(float) * float(factor)
                new_lcounts[col] = float(lcounts[col]) * float(factor)
                new_units[col] = si_unit

        OBSERVATIONS[obs_name] = df_si
        OBS_UNITS[obs_name] = new_units
        OBS_LEAST_COUNTS[obs_name] = new_lcounts

        sanitized = _sanitize_name(obs_name)
        setattr(__import__(__name__), sanitized, OBSERVATIONS[obs_name])

    return OBSERVATIONS

# ---- Step 4: create parsed_data with _err columns (using lc/sqrt(12)) ----

def make_parsed_data(parsed_dir: str = "parsed_data", float_format: str = "%.12g"):
    """
    STEP 4 (updated):
    - Uses OBS_DISTRIBUTION to choose how least-counts convert to standard uncertainty:
        'V' => sigma = lc / sqrt(24)
        'R' or '' (default) => sigma = lc / sqrt(12)
    - If OBS_LEAST_COUNTS does not contain a column but the dataframe already has <col>_err,
      it will use that existing error column (assumed to be in SI).
    - Produces parsed_data/<OBS_NAME>_SI_PARSED.csv and updates OBSERVATIONS in-memory.
    """
    if not OBSERVATIONS:
        raise ValueError("No observations available. Run import_data(), unit_check(), unit_converter() first.")

    out_dir = os.path.join(os.getcwd(), parsed_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    # Ensure OBS_DISTRIBUTION exists as a dict
    dist_store = globals().get("OBS_DISTRIBUTION", {})
    for obs_name, df in list(OBSERVATIONS.items()):
        units = OBS_UNITS.get(obs_name)
        lcounts = OBS_LEAST_COUNTS.get(obs_name, {})
        dist_map = dist_store.get(obs_name, {})
        if units is None:
            raise ValueError(f"Missing units metadata for '{obs_name}'. Run prior steps in order.")

        cols = []
        data = {}
        for col in df.columns:
            cols.append(col)
            data[col] = df[col].values

            # Determine sigma for this column
            err_col = f"{col}_err"

            # Priority:
            # 1) If df already has err_col, use those values (assumed SI)
            # 2) Else if lcounts has column, compute sigma per distribution token
            # 3) Else error
            if err_col in df.columns:
                # use existing error column (already in OBSERVATIONS)
                sigma_arr = np.asarray(df[err_col], dtype=float)
            elif col in lcounts:
                lc = float(lcounts[col])
                token = dist_map.get(col, "").upper() if dist_map else ""
                if token == "V":
                    sigma = lc / math.sqrt(24.0)
                else:
                    # default and 'R'
                    sigma = lc / math.sqrt(12.0)
                sigma_arr = np.full(len(df), sigma, dtype=float)
            else:
                raise ValueError(f"No least-count or existing '{err_col}' for column '{col}' in observation '{obs_name}'.")

            # append error column data
            cols.append(err_col)
            data[err_col] = sigma_arr

        parsed_df = pd.DataFrame(data, columns=cols)

        safe_name = obs_name.replace(os.sep, "_")
        outname = os.path.join(out_dir, f"{safe_name}_SI_PARSED.csv")
        parsed_df.to_csv(outname, index=False, float_format=float_format)

        OBSERVATIONS[obs_name] = parsed_df
        sanitized = _sanitize_name(obs_name)
        setattr(__import__(__name__), sanitized, parsed_df)

        written.append(outname)

    return written


# ---- Image directory helper ----

def ensure_image_dir(image_dir: str = "images"):
    path = os.path.join(os.getcwd(), image_dir)
    os.makedirs(path, exist_ok=True)
    return path

# ---- Error propagation utilities ----

def _safe_step(x_val: float, err_val: float):
    """Choose a small step for numerical derivative: prefer err_val, fallback to scale-based."""
    if err_val is not None and err_val > 0:
        return err_val
    return max(abs(x_val) * 1e-6, 1e-8)

def combine_observations(obs_list: Sequence[str],
                         x_col: str,
                         y_col: str,
                         require_exact_x: bool = True,
                         atol: float = 1e-9,
                         save_name: Optional[str] = None):
    """
    Combine multiple observations (same experiment repeated) into a single dataset.

    - obs_list: list of observation keys (exact filenames without .csv)
    - x_col: independent variable column name to align on (e.g., "Voltage")
    - y_col: dependent variable name to combine (e.g., "D_sq")
    - require_exact_x: if True, x values must match exactly across datasets (within `atol`).
      if False, the function will match nearest x (not implemented here — I can add it).
    - Returns: combined_df with columns:
         x_col, y_mean, y_std (sample std across N), y_sem (std / sqrt(N)), n_points
    - Also writes parsed file parsed_data/<save_name or commonprefix>_combined.csv if save_name provided.
    """
    # Validate
    dfs = []
    for obs in obs_list:
        if obs not in OBSERVATIONS:
            raise KeyError(f"Observation '{obs}' not loaded.")
        dfs.append(OBSERVATIONS[obs].copy())

    # Extract x arrays and ensure matching
    x0 = np.asarray(dfs[0][x_col], dtype=float)
    N = len(dfs)
    for df in dfs[1:]:
        xi = np.asarray(df[x_col], dtype=float)
        if len(xi) != len(x0):
            raise ValueError("Datasets have different number of x points; combine requires same grid.")
        if require_exact_x:
            if not np.allclose(x0, xi, atol=atol, rtol=0):
                raise ValueError("x values differ between datasets; set require_exact_x=False to allow nearest matching.")
    # Stack y values: shape (N, n_points)
    ys = np.vstack([np.asarray(df[y_col], dtype=float) for df in dfs])  # (N, M)
    # compute stats along axis=0
    y_mean = np.mean(ys, axis=0)
    # sample standard deviation (ddof=1)
    if N > 1:
        y_std = np.std(ys, axis=0, ddof=1)
    else:
        y_std = np.zeros_like(y_mean)
    y_sem = y_std / np.sqrt(N)
    combined = pd.DataFrame({
        x_col: x0,
        f"{y_col}_mean": y_mean,
        f"{y_col}_std": y_std,
        f"{y_col}_sem": y_sem,
        "n": np.full(len(y_mean), N, dtype=int),
    })
    if save_name:
        outdir = os.path.join(os.getcwd(), "parsed_data")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, save_name)
        combined.to_csv(outpath, index=False, float_format="%.12g")
    return combined


def propagate_function(obs_name: str,
                       input_cols: Sequence[str],
                       func: Callable[..., np.ndarray],
                       new_col: str,
                       jacobian: Optional[Callable[..., np.ndarray]] = None,
                       overwrite: bool = False):
    """
    Generic error propagation.
    - obs_name: observation key in OBSERVATIONS
    - input_cols: list of column names used as inputs (order matters)
    - func: callable that accepts (x1_array, x2_array, ...) or row-wise scalars;
            it must return an array-like of same length (vectorized preferred)
            If `func` takes a single numpy array (n,) return (n,) array.
    - new_col: name for the resulting column (values)
    - jacobian: optional callable that returns partial derivatives dF/dxi as arrays.
                It must accept same args as func and return either:
                   * if single input: array (n,)
                   * if multiple inputs: array shape (n, n_inputs) or list of arrays [dF/dx1, dF/dx2, ...]
                If jacobian is None, numeric central-difference is used with step = input_err (stddev).
    - overwrite: if True, will overwrite existing columns.

    Adds new_col and new_col + "_err" to OBSERVATIONS[obs_name].
    Returns (parsed_df_of_observation) for chaining.
    """
    if obs_name not in OBSERVATIONS:
        raise KeyError(f"Observation '{obs_name}' not loaded.")

    df = OBSERVATIONS[obs_name].copy()
    # If we're in pre-parsed stage (no _err columns), allow using OBS_LEAST_COUNTS
    # Determine per-input standard uncertainties
    stds = []
    if all(col in df.columns for col in input_cols):
        # If parsed_data stage, try to find <col>_err columns
        for col in input_cols:
            err_col = f"{col}_err"
            if err_col in df.columns:
                stds.append(np.asarray(df[err_col], dtype=float))
            else:
                # fallback to module-level least counts if present
                lc_map = OBS_LEAST_COUNTS.get(obs_name)
                if lc_map and col in lc_map:
                    sigma = lc_map[col] / math.sqrt(12.0)
                    stds.append(np.full(len(df), sigma, dtype=float))
                else:
                    raise ValueError(f"No error metadata for input column '{col}' in observation '{obs_name}'.")
    else:
        raise ValueError(f"One or more input columns {input_cols} not present in observation '{obs_name}'.")

    # prepare arrays for func evaluation
    args = [np.asarray(df[c], dtype=float) for c in input_cols]
    # Evaluate values
    try:
        values = func(*args)
        values = np.asarray(values, dtype=float)
    except Exception:
        # Try element-wise evaluation (less efficient)
        vals = []
        for row in zip(*args):
            vals.append(func(*row))
        values = np.asarray(vals, dtype=float)

    # Compute partial derivatives
    n = len(values)
    if jacobian is not None:
        # user-provided jacobian
        J = jacobian(*args)
        J_arr = None
        if isinstance(J, list) or isinstance(J, tuple):
            # list of arrays
            J_arr = np.vstack([np.asarray(j, dtype=float) for j in J]).T  # (n, n_inputs)
        else:
            J_arr = np.asarray(J, dtype=float)
            if J_arr.ndim == 1:
                # single input
                J_arr = J_arr.reshape(n, 1)
        # shape check
        if J_arr.shape[0] != n:
            raise ValueError("Jacobian returned incompatible length.")
    else:
        # numeric central difference using stds as step
        J_arr = np.zeros((n, len(input_cols)), dtype=float)
        for i, col in enumerate(input_cols):
            x = args[i]
            step = np.array(stds[i], dtype=float)
            # avoid zero step
            step = np.where(step <= 0, np.maximum(np.abs(x) * 1e-6, 1e-8), step)
            # Compute f(x + step) and f(x - step)
            plus_args = [a.copy() for a in args]
            minus_args = [a.copy() for a in args]
            plus_args[i] = x + step
            minus_args[i] = x - step
            try:
                f_plus = func(*plus_args)
                f_minus = func(*minus_args)
                f_plus = np.asarray(f_plus, dtype=float)
                f_minus = np.asarray(f_minus, dtype=float)
            except Exception:
                # fallback to element-wise central diff
                f_plus = np.empty(n, dtype=float)
                f_minus = np.empty(n, dtype=float)
                for idx in range(n):
                    p_args = [a[idx] for a in plus_args]
                    m_args = [a[idx] for a in minus_args]
                    f_plus[idx] = func(*p_args)
                    f_minus[idx] = func(*m_args)
            # derivative approx
            deriv = (f_plus - f_minus) / (2.0 * step)
            J_arr[:, i] = deriv

    # propagate: sigma_f = sqrt( sum_i (dF/dxi * sigma_xi)^2 )
    sigma_sq = np.zeros(n, dtype=float)
    for i in range(len(input_cols)):
        sigma_sq += (J_arr[:, i] * np.asarray(stds[i], dtype=float))**2
    sigma_f = np.sqrt(sigma_sq)

    # Add columns to dataframe
    if (new_col in df.columns or f"{new_col}_err" in df.columns) and not overwrite:
        raise ValueError(f"Column {new_col} or {new_col}_err already exists. Set overwrite=True to replace.")

    df[new_col] = values
    df[f"{new_col}_err"] = sigma_f

    # persist and return
    OBSERVATIONS[obs_name] = df
    sanitized = _sanitize_name(obs_name)
    setattr(__import__(__name__), sanitized, df)
    return df

# Convenience wrappers (examples)
def propagate_square(obs_name: str, src_col: str, out_col: Optional[str] = None, overwrite: bool = False):
    out = out_col or f"{src_col}_sq"
    def f(x): return np.asarray(x)**2
    def jac(x):
        # derivative 2*x
        return 2.0 * np.asarray(x)
    return propagate_function(obs_name, [src_col], f, out, jacobian=jac, overwrite=overwrite)

def propagate_log(obs_name: str, src_col: str, out_col: Optional[str] = None, base: float = math.e, overwrite: bool = False):
    out = out_col or f"{src_col}_log"
    ln_base = math.log(base)
    def f(x): return np.log(np.asarray(x)) / ln_base
    def jac(x):
        return 1.0 / (np.asarray(x) * ln_base)
    return propagate_function(obs_name, [src_col], f, out, jacobian=jac, overwrite=overwrite)

def propagate_exp(obs_name: str, src_col: str, out_col: Optional[str] = None, overwrite: bool = False):
    out = out_col or f"{src_col}_exp"
    def f(x): return np.exp(np.asarray(x))
    def jac(x):
        return np.exp(np.asarray(x))
    return propagate_function(obs_name, [src_col], f, out, jacobian=jac, overwrite=overwrite)


def collapse_repeats_and_combine_errors(obs_name: str,
                                        group_by: str,
                                        cols: Optional[Sequence[str]] = None,
                                        random_method: str = "sem",
                                        reduce_inst: bool = True,
                                        save: bool = True,
                                        out_name: Optional[str] = None,
                                        parsed_dir: str = "parsed_data",
                                        float_format: str = "%.12g"):
    """
    Collapse repeated rows that share the same value in `group_by` column.
    For each group compute:
      - mean value for each column
      - random error (based on random_method)
      - instrumental uncertainty (from OBS_LEAST_COUNTS + OBS_DISTRIBUTION)
      - combined error = sqrt( sigma_inst_used^2 + sigma_random_used^2 )

    Parameters
    - obs_name: observation key in OBSERVATIONS
    - group_by: column name to group on (e.g., "Voltage")
    - cols: which columns to process (default: all numeric columns)
    - random_method: "sem" (standard error of mean, default), "std" (sample std), or "mad" (mean absolute deviation)
    - reduce_inst: if True, instrument uncertainty is divided by sqrt(n) when computing for the mean
    - save: if True, save combined DataFrame to parsed_dir/out_name (or generated name)
    - out_name: filename to save, e.g., "I=1Amp_collapsed.csv". If None, auto-generates.
    Returns (combined_df)
    """
    import numpy as np
    import pandas as pd
    import math
    if obs_name not in OBSERVATIONS:
        raise KeyError(f"Observation '{obs_name}' not loaded.")

    df = OBSERVATIONS[obs_name].copy()

    if group_by not in df.columns:
        raise KeyError(f"group_by column '{group_by}' not in observation '{obs_name}'.")

    # determine columns to process (exclude group_by if not requested)
    if cols is None:
        # take all numeric columns
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # ensure group_by included as first column in result
    if group_by in cols:
        cols_proc = [c for c in cols if c != group_by]
    else:
        cols_proc = list(cols)

    # Prepare containers
    grouped = df.groupby(group_by, sort=True)
    out_rows = []
    for key, grp in grouped:
        n = len(grp)
        row = {group_by: key, "n": n}
        for col in cols_proc:
            values = np.asarray(grp[col].astype(float))
            mean_val = float(np.mean(values))
            # compute random component
            if random_method == "sem":
                if n > 1:
                    s = float(np.std(values, ddof=1))
                    rand = s / math.sqrt(n)
                else:
                    # single measurement -> no scatter -> random=0
                    rand = 0.0
            elif random_method == "std":
                rand = float(np.std(values, ddof=1)) if n > 1 else 0.0
            elif random_method == "mad":
                rand = float(np.mean(np.abs(values - mean_val)))
            else:
                raise ValueError("random_method must be one of 'sem','std','mad'")

            # compute instrumental uncertainty from OBS_LEAST_COUNTS and OBS_DISTRIBUTION
            sigma_inst = 0.0
            if obs_name in OBS_LEAST_COUNTS and col in OBS_LEAST_COUNTS[obs_name]:
                lc = float(OBS_LEAST_COUNTS[obs_name][col])
                # distribution token (V or R or "")
                tok = ""
                if obs_name in globals().get("OBS_DISTRIBUTION", {}):
                    tok = OBS_DISTRIBUTION[obs_name].get(col, "") 
                # decide divisor
                if tok == "V":
                    sigma_inst = lc / math.sqrt(24.0)
                else:
                    # default / 'R'
                    sigma_inst = lc / math.sqrt(12.0)

                # If instrument uncertainty should be reduced for the mean, divide by sqrt(n)
                if reduce_inst and n > 1:
                    sigma_inst = sigma_inst / math.sqrt(n)

            # combined uncertainty
            sigma_comb = math.sqrt(sigma_inst**2 + (rand**2))

            # store
            row[col] = mean_val
            row[f"{col}_err"] = sigma_comb
            # also store components if useful for debugging
            row[f"{col}_rand"] = rand
            row[f"{col}_inst"] = sigma_inst

        out_rows.append(row)

    combined_df = pd.DataFrame(out_rows)
    # reorder columns: group_by, for each col: col, col_err (and optionally others)
    ordered = [group_by]
    for col in cols_proc:
        ordered.append(col)
        ordered.append(f"{col}_err")
    ordered = [c for c in ordered if c in combined_df.columns]
    # Keep n at end
    if "n" in combined_df.columns:
        ordered.append("n")
    combined_df = combined_df[ordered]

    # Save if requested
    if save:
        os.makedirs(os.path.join(os.getcwd(), parsed_dir), exist_ok=True)
        if out_name:
            fname = out_name
        else:
            safe = _sanitize_name(obs_name)
            fname = f"{safe}__collapsed_by_{_sanitize_name(group_by)}.csv"
        outpath = os.path.join(os.getcwd(), parsed_dir, fname)
        combined_df.to_csv(outpath, index=False, float_format=float_format)
        print(f"Saved collapsed data: {outpath}")

    return combined_df



# ---- ODR fitting + plotting ----

def fit_odr(obs_name: str,
            x_col: str,
            y_col: str,
            x_err_col: Optional[str] = None,
            y_err_col: Optional[str] = None,
            model_func: Optional[Callable] = None,
            beta0: Optional[Sequence[float]] = None,
            maxit: int = 200):
    """
    Perform ODR fit on a dataset.
    - model_func: callable f(B, x) where B is parameter vector and x is x-array (as required by scipy.odr)
      If None, defaults to linear: y = m*x + c
    - x_err_col, y_err_col: names of error columns (if None, zeros used)
    Returns the scipy.odr output object (odr_output).
    """
    try:
        from scipy.odr import ODR, Model, RealData
    except Exception as e:
        raise RuntimeError("scipy.odr is required for fit_odr. Install scipy.") from e

    if obs_name not in OBSERVATIONS:
        raise KeyError(f"Observation '{obs_name}' not loaded.")

    df = OBSERVATIONS[obs_name]
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError("x_col or y_col not present in observation.")

    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)
    sx = np.zeros_like(x)
    sy = np.zeros_like(y)
    if x_err_col:
        if x_err_col not in df.columns:
            raise KeyError(f"x error column '{x_err_col}' not found.")
        sx = np.asarray(df[x_err_col], dtype=float)
    if y_err_col:
        if y_err_col not in df.columns:
            raise KeyError(f"y error column '{y_err_col}' not found.")
        sy = np.asarray(df[y_err_col], dtype=float)

    if model_func is None:
        # linear default: y = m*x + c -> B=[m,c]
        def linear(B, x):
            return B[0]*x + B[1]
        model = Model(linear)
        beta0 = beta0 or [1.0, 0.0]
    else:
        model = Model(model_func)
        if beta0 is None:
            # try simple guess
            beta0 = np.polyfit(x, y, 1).tolist() if len(x) >= 2 else [1.0, 0.0]

    data = RealData(x, y, sx=sx, sy=sy)
    odr = ODR(data, model, beta0=beta0, maxit=maxit)
    out = odr.run()

    # prepare report
    params = out.beta
    param_err = out.sd_beta
    res = {
    "params": params,
    "param_err": param_err,
    "cov_beta": out.cov_beta,
    "sum_square": out.sum_square,
    "res_var": out.res_var,
    "info": out.info,
    "message": out.stopreason if hasattr(out, 'stopreason') else None,
    "odr_output": out,
    "model_func": model.fcn   # <<< IMPORTANT: store function
}


    # print succinct summary
    print("ODR fit results:")
    for i, (p, pe) in enumerate(zip(params, param_err)):
        print(f"  p[{i}] = {p:.6g} ± {pe:.6g}")
    print(f"  sum_square = {out.sum_square:.6g}")
    print(f"  reduced chi2 (res_var) = {out.res_var:.6g}")

    return res

def plot_with_fit(obs_name: str,
                  x_col: str,
                  y_col: str,
                  x_err_col: Optional[str],
                  y_err_col: Optional[str],
                  fit_result: dict,
                  title: Optional[str] = None,
                  image_dir: str = "images",
                  points_label: Optional[str] = None,
                  save_name: Optional[str] = None,
                  dpi: int = 300):
    """
    Plot data with errorbars and ODR fit curve.
    - fit_result: the dict returned by fit_odr(...)
    - saves PNG into image_dir and returns path.
    """
    if obs_name not in OBSERVATIONS:
        raise KeyError(f"Observation '{obs_name}' not loaded.")
    df = OBSERVATIONS[obs_name]

    # Extract data and uncertainties
    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)
    sx = np.zeros_like(x)
    sy = np.zeros_like(y)
    if x_err_col: sx = np.asarray(df[x_err_col], dtype=float)
    if y_err_col: sy = np.asarray(df[y_err_col], dtype=float)

    # Get model function and fitted parameters from fit_result
    model_func = fit_result.get("model_func")
    if model_func is None:
        raise ValueError("fit_result does not contain 'model_func'. Re-run fit_odr() which stores the model function in the returned dict.")

    params = fit_result.get("params")
    if params is None:
        raise ValueError("fit_result does not contain fitted parameters ('params').")

    # Build dense x-grid BEFORE calling the model function
    x_min = np.min(x)
    x_max = np.max(x)
    span = x_max - x_min
    if span == 0:
        # degenerate single-x case: make a small span
        span = abs(x_min) if x_min != 0 else 1.0
    xs = np.linspace(x_min - 0.05 * span, x_max + 0.05 * span, 500)

    # Evaluate fit curve: model_func must accept (params, x_array)
    try:
        ys_fit = np.asarray(model_func(params, xs), dtype=float)
    except TypeError:
        # try the alternate signature model_func(x, params) just in case
        try:
            ys_fit = np.asarray(model_func(xs, params), dtype=float)
        except Exception as e:
            raise RuntimeError("model_func(params, xs) failed — model function has unexpected signature.") from e
    except Exception as e:
        raise RuntimeError("Evaluation of model function failed.") from e

    # Plotting
    fig, ax = plt.subplots(figsize=(6,4))
    ax.errorbar(x, y, xerr=(sx if x_err_col else None), yerr=(sy if y_err_col else None),
                fmt='o', markersize=4, label=points_label or "data")
    ax.plot(xs, ys_fit, label="ODR fit")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title is None:
        title = f"{x_col} vs {y_col} — {obs_name}"
    ax.set_title(title)
    ax.legend(loc='best')

    # annotate parameters
    param_err = fit_result.get("param_err")
    text_lines = []
    if param_err is not None:
        for i, (p, pe) in enumerate(zip(params, param_err)):
            text_lines.append(f"p{i} = {p:.4g} ± {pe:.4g}")
    else:
        for i, p in enumerate(params):
            text_lines.append(f"p{i} = {p:.4g}")
    res_var = fit_result.get("res_var", None)
    if res_var is not None:
        text_lines.append(f"reduced χ² = {res_var:.4g}")
    ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=8, va='top', ha='left', bbox=dict(boxstyle='round', fc='w', alpha=0.8))

    # save
    imgdir = ensure_image_dir(image_dir)
    safe_name = save_name or f"{_sanitize_name(obs_name)}__{_sanitize_name(x_col)}_vs_{_sanitize_name(y_col)}.png"
    outpath = os.path.join(imgdir, safe_name)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved: {outpath}")
    return outpath


# ---- Utilities ----

def summary() -> str:
    lines = []
    if not OBSERVATIONS:
        return "No observations loaded."
    for obs, df in OBSERVATIONS.items():
        lines.append(f"Observation: '{obs}' — columns: {list(df.columns)} — rows: {len(df)}")
        if obs in OBS_UNITS:
            lines.append("  units: " + ", ".join(f"{k}: {v}" for k, v in OBS_UNITS[obs].items()))
        if obs in OBS_LEAST_COUNTS:
            lines.append("  least counts: " + ", ".join(f"{k}: {v}" for k, v in OBS_LEAST_COUNTS[obs].items()))
    return "\n".join(lines)

