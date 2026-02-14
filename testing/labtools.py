# labtools.py
"""
Comprehensive lab tools for CSV import, unit conversion, error propagation,
combining repeated runs, ODR fitting and plotting.

Pipeline overview:
  import_data() -> unit_check() -> unit_converter() -> make_parsed_data()
After that you can:
  - propagate_function / propagate_square / propagate_log / propagate_exp
  - combine_observations
  - fit_odr + plot_with_fit
  - plot (simple plotting)

CSV expectations (strict):
  Row 0: header like "Voltage (V), rhs (cm), lhs (cm)"
  Row 1: least-count numeric row (one column per header)
  Optional Row 2: distribution tokens (e.g., "V" or "R") — any non-numeric row is treated
                  as distribution flags; if present, data start at row 3.
  Remaining rows: numeric data rows.

Distribution rules used when making parsed data:
  - 'V' token for a column: sigma = lc / sqrt(24)
  - 'R' or missing token: sigma = lc / sqrt(12)
"""

from typing import Dict, Tuple, Callable, Sequence, Optional
import os
import re
import math
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- module-level storage ----
OBSERVATIONS: Dict[str, pd.DataFrame] = {}
OBS_UNITS: Dict[str, Dict[str, str]] = {}
OBS_LEAST_COUNTS: Dict[str, Dict[str, float]] = {}
OBS_DISTRIBUTION: Dict[str, Dict[str, str]] = {}  # 'V'/'R'/'' per column

# ---- unit conversion dictionary (unit -> (factor_to_SI, SI_unit_name, special_flag)) ----
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
    "min":  (60.0,    "s", None),
    "h":    (3600.0,  "s", None),
    # Electric
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
    "MΩ":   (1e6,     "ohm", None),
    # Capacitance / inductance
    "F":    (1.0,     "F", None),
    "uF":   (1e-6,    "F", None),
    "H":    (1.0,     "H", None),
    "mH":   (1e-3,    "H", None),
    # Energy / others
    "J":    (1.0,     "J", None),
    "eV":   (1.602176634e-19, "J", None),
    "W":    (1.0,     "W", None),
    "Hz":   (1.0,     "Hz", None),
    # Temperature
    "K":    (1.0,     "K", None),
    "C":    (1.0,     "K", "C2K"),
    "°C":   (1.0,     "K", "C2K"),
    "degC": (1.0,     "K", "C2K"),
    # Magnetic
    "T":    (1.0,     "T", None),
    "mT":   (1e-3,    "T", None),
    "G":    (1e-4,    "T", None),
    # amount
    "mol":  (1.0,     "mol", None),
    # density
    "g/cm^3": (1000.0, "kg/m^3", None),
    # volts
    "kV":   (1e3,     "V", None),
}

# small synonyms
UNIT_CONVERSIONS.setdefault("um", (1e-6, "m", None))
UNIT_CONVERSIONS.setdefault("μm", (1e-6, "m", None))

# regex for header parsing "Name (unit)"
HEADER_RE = re.compile(r'^\s*(?P<name>.*?)\s*\(\s*(?P<unit>.+?)\s*\)\s*$')

# ---- helpers ----
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

# ---- Primary functions ----

def import_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Import CSVs from data_dir.
    - Parses optional distribution row (third CSV row) if it is non-numeric.
    - Stores OBSERVATIONS, OBS_UNITS, OBS_LEAST_COUNTS, OBS_DISTRIBUTION.
    """
    cwd = os.getcwd()
    target = os.path.join(cwd, data_dir)
    if not os.path.isdir(target):
        raise ValueError(f"Data directory not found: {target}")

    csv_files = [f for f in os.listdir(target) if f.lower().endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {target}")

    OBSERVATIONS.clear()
    OBS_UNITS.clear()
    OBS_LEAST_COUNTS.clear()
    OBS_DISTRIBUTION.clear()

    caller_globals = None
    try:
        frm = inspect.stack()[1].frame
        caller_globals = frm.f_globals
    except Exception:
        caller_globals = None

    for fname in csv_files:
        full = os.path.join(target, fname)
        obs_name = os.path.splitext(fname)[0]
        sanitized = _sanitize_name(obs_name)

        raw = pd.read_csv(full, header=None, dtype=str, keep_default_na=False)
        if raw.shape[0] < 3:
            raise ValueError(f"CSV '{fname}' must have at least 3 rows: header, least counts, and >=1 datapoint.")

        header_row = raw.iloc[0].tolist()
        least_row = raw.iloc[1].tolist()

        # detect optional distribution row (third row) if it is non-numeric
        dist_row = None
        data_start_idx = 2
        if raw.shape[0] >= 4:
            third = raw.iloc[2].tolist()
            # treat as distribution row if any cell in third row is non-numeric
            is_numeric_row = True
            for cell in third:
                try:
                    _ = float(str(cell).strip())
                except Exception:
                    is_numeric_row = False
                    break
            if not is_numeric_row:
                dist_row = third
                data_start_idx = 3

        data_rows = raw.iloc[data_start_idx:].copy()

        colnames = []
        units = {}
        for idx, cell in enumerate(header_row):
            varname, unit = _parse_header_cell(cell)
            if varname in colnames:
                raise ValueError(f"In file '{fname}', duplicate column name '{varname}'.")
            colnames.append(varname)
            units[varname] = unit

        if len(least_row) != len(colnames):
            raise ValueError(f"In file '{fname}', least count row has {len(least_row)} columns but header has {len(colnames)}.")

        least_counts = {}
        for i, val in enumerate(least_row):
            c = str(val).strip()
            try:
                fval = float(c)
            except Exception:
                raise ValueError(f"In file '{fname}', least count for '{colnames[i]}' is not numeric: '{c}'")
            least_counts[colnames[i]] = fval

        # parse distribution row if present
        dist_map = {}
        if dist_row and len(dist_row) == len(colnames):
            for i, tok in enumerate(dist_row):
                t = str(tok).strip()
                tnorm = t.upper() if t else ""
                dist_map[colnames[i]] = tnorm if tnorm in ("V", "R") else ""
        else:
            for c in colnames:
                dist_map[c] = ""

        if data_rows.shape[0] < 1:
            raise ValueError(f"In file '{fname}', no data rows found after header/least-count{', distribution row' if dist_row else ''}.")

        if data_rows.shape[1] != len(colnames):
            raise ValueError(f"In file '{fname}', data rows have {data_rows.shape[1]} columns but header has {len(colnames)}.")

        numeric_df = data_rows.copy()
        numeric_df.columns = colnames
        for col in colnames:
            try:
                numeric_df[col] = pd.to_numeric(numeric_df[col].astype(str).str.strip(), errors='raise')
            except Exception:
                raise ValueError(f"In file '{fname}', data column '{col}' contains non-numeric entries; can't import.")
        numeric_df.reset_index(drop=True, inplace=True)

        OBSERVATIONS[obs_name] = numeric_df.copy()
        OBS_UNITS[obs_name] = units.copy()
        OBS_LEAST_COUNTS[obs_name] = least_counts.copy()
        OBS_DISTRIBUTION[obs_name] = dist_map.copy()

        setattr(__import__(__name__), sanitized, OBSERVATIONS[obs_name])
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

def make_parsed_data(parsed_dir: str = "parsed_data", float_format: str = "%.12g"):
    """
    Create parsed_data/<OBS_NAME>_SI_PARSED.csv with interleaved _err columns.
    Distribution-aware:
      - 'V' -> sigma = lc / sqrt(24)
      - 'R' or '' -> sigma = lc / sqrt(12)
    """
    if not OBSERVATIONS:
        raise ValueError("No observations available. Run import_data(), unit_check(), unit_converter() first.")

    out_dir = os.path.join(os.getcwd(), parsed_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for obs_name, df in list(OBSERVATIONS.items()):
        units = OBS_UNITS.get(obs_name)
        lcounts = OBS_LEAST_COUNTS.get(obs_name, {})
        dist_map = OBS_DISTRIBUTION.get(obs_name, {})
        if units is None:
            raise ValueError(f"Missing units metadata for '{obs_name}'. Run prior steps in order.")

        cols = []
        data = {}
        for col in df.columns:
            cols.append(col)
            data[col] = df[col].values

            err_col = f"{col}_err"
            # Use existing error if present
            if err_col in df.columns:
                sigma_arr = np.asarray(df[err_col], dtype=float)
            elif col in lcounts:
                lc = float(lcounts[col])
                token = dist_map.get(col, "").upper() if dist_map else ""
                if token == "V":
                    sigma = lc / math.sqrt(24.0)
                else:
                    sigma = lc / math.sqrt(12.0)
                sigma_arr = np.full(len(df), sigma, dtype=float)
            else:
                raise ValueError(f"No least-count or existing '{err_col}' for column '{col}' in observation '{obs_name}'.")

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

def ensure_image_dir(image_dir: str = "images"):
    path = os.path.join(os.getcwd(), image_dir)
    os.makedirs(path, exist_ok=True)
    return path

# ---- Error propagation utilities ----

def _safe_step(x_val: float, err_val: float):
    if err_val is not None and err_val > 0:
        return err_val
    return max(abs(x_val) * 1e-6, 1e-8)

def propagate_function(obs_name: str,
                       input_cols: Sequence[str],
                       func: Callable[..., np.ndarray],
                       new_col: str,
                       jacobian: Optional[Callable[..., np.ndarray]] = None,
                       overwrite: bool = False):
    """
    Generic error propagation that adds new_col and new_col_err to OBSERVATIONS[obs_name].
    If <col>_err exists in the DF it is used; otherwise uses OBS_LEAST_COUNTS + OBS_DISTRIBUTION to compute sigma.
    """
    if obs_name not in OBSERVATIONS:
        raise KeyError(f"Observation '{obs_name}' not loaded.")

    df = OBSERVATIONS[obs_name].copy()

    # determine per-input std arrays
    stds = []
    for col in input_cols:
        err_col = f"{col}_err"
        if err_col in df.columns:
            stds.append(np.asarray(df[err_col], dtype=float))
        else:
            # fallback to OBS_LEAST_COUNTS + OBS_DISTRIBUTION
            lc_map = OBS_LEAST_COUNTS.get(obs_name, {})
            dist_map = OBS_DISTRIBUTION.get(obs_name, {})
            if col in lc_map:
                lc = float(lc_map[col])
                token = dist_map.get(col, "").upper() if dist_map else ""
                if token == "V":
                    sigma = lc / math.sqrt(24.0)
                else:
                    sigma = lc / math.sqrt(12.0)
                stds.append(np.full(len(df), sigma, dtype=float))
            else:
                raise ValueError(f"No error metadata for input column '{col}' in observation '{obs_name}'.")

    # prepare input arrays
    args = [np.asarray(df[c], dtype=float) for c in input_cols]

    # compute values
    try:
        values = func(*args)
        values = np.asarray(values, dtype=float)
    except Exception:
        # fallback to element-wise
        values = np.array([func(*row) for row in zip(*args)], dtype=float)

    n = len(values)

    # compute jacobian
    if jacobian is not None:
        J = jacobian(*args)
        if isinstance(J, (list, tuple)):
            J_arr = np.vstack([np.asarray(j, dtype=float) for j in J]).T
        else:
            J_arr = np.asarray(J, dtype=float)
            if J_arr.ndim == 1:
                J_arr = J_arr.reshape(n, 1)
    else:
        J_arr = np.zeros((n, len(input_cols)), dtype=float)
        for i in range(len(input_cols)):
            x = args[i]
            step = np.array(stds[i], dtype=float)
            step = np.where(step <= 0, np.maximum(np.abs(x) * 1e-6, 1e-8), step)
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
                f_plus = np.empty(n, dtype=float)
                f_minus = np.empty(n, dtype=float)
                for idx in range(n):
                    p_args = [a[idx] for a in plus_args]
                    m_args = [a[idx] for a in minus_args]
                    f_plus[idx] = func(*p_args)
                    f_minus[idx] = func(*m_args)
            deriv = (f_plus - f_minus) / (2.0 * step)
            J_arr[:, i] = deriv

    sigma_sq = np.zeros(n, dtype=float)
    for i in range(len(input_cols)):
        sigma_sq += (J_arr[:, i] * np.asarray(stds[i], dtype=float))**2
    sigma_f = np.sqrt(sigma_sq)

    if (new_col in df.columns or f"{new_col}_err" in df.columns) and not overwrite:
        raise ValueError(f"Column {new_col} or {new_col}_err already exists. Set overwrite=True to replace.")

    df[new_col] = values
    df[f"{new_col}_err"] = sigma_f

    OBSERVATIONS[obs_name] = df
    sanitized = _sanitize_name(obs_name)
    setattr(__import__(__name__), sanitized, df)
    return df

# convenience wrappers
def propagate_square(obs_name: str, src_col: str, out_col: Optional[str] = None, overwrite: bool = False):
    out = out_col or f"{src_col}_sq"
    def f(x): return np.asarray(x)**2
    def jac(x): return 2.0 * np.asarray(x)
    return propagate_function(obs_name, [src_col], f, out, jacobian=jac, overwrite=overwrite)

def propagate_log(obs_name: str, src_col: str, out_col: Optional[str] = None, base: float = math.e, overwrite: bool = False):
    out = out_col or f"{src_col}_log"
    ln_base = math.log(base)
    def f(x): return np.log(np.asarray(x)) / ln_base
    def jac(x): return 1.0 / (np.asarray(x) * ln_base)
    return propagate_function(obs_name, [src_col], f, out, jacobian=jac, overwrite=overwrite)

def propagate_exp(obs_name: str, src_col: str, out_col: Optional[str] = None, overwrite: bool = False):
    out = out_col or f"{src_col}_exp"
    def f(x): return np.exp(np.asarray(x))
    def jac(x): return np.exp(np.asarray(x))
    return propagate_function(obs_name, [src_col], f, out, jacobian=jac, overwrite=overwrite)

# ---- Combining repeated runs ----

def combine_observations(obs_list: Sequence[str],
                         x_col: str,
                         y_col: str,
                         require_exact_x: bool = True,
                         atol: float = 1e-9,
                         save_name: Optional[str] = None) -> pd.DataFrame:
    """
    Combine repeated runs (obs_list) aligned on x_col.
    Returns a DataFrame with columns: x_col, y_mean, y_std (sample std), y_sem, n
    If save_name provided, saves to parsed_data/<save_name>.
    """
    dfs = []
    for obs in obs_list:
        if obs not in OBSERVATIONS:
            raise KeyError(f"Observation '{obs}' not loaded.")
        dfs.append(OBSERVATIONS[obs].copy())

    x0 = np.asarray(dfs[0][x_col], dtype=float)
    N = len(dfs)
    for df in dfs[1:]:
        xi = np.asarray(df[x_col], dtype=float)
        if len(xi) != len(x0):
            raise ValueError("Datasets have different number of x points; combine requires same grid.")
        if require_exact_x:
            if not np.allclose(x0, xi, atol=atol, rtol=0):
                raise ValueError("x values differ between datasets; set require_exact_x=False to allow nearest matching.")

    ys = np.vstack([np.asarray(df[y_col], dtype=float) for df in dfs])
    y_mean = np.mean(ys, axis=0)
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
    Perform ODR fit. Returns dict with params, param_err, cov_beta, res_var, odr_output, and model_func.
    model_func must be callable f(B, x) where B is parameter vector and x is array.
    If model_func is None, linear model is used.
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
        def linear(B, x):
            return B[0]*x + B[1]
        model = Model(linear)
        beta0 = beta0 or [1.0, 0.0]
    else:
        model = Model(model_func)
        if beta0 is None:
            try:
                beta0 = np.polyfit(x, y, 1).tolist()
            except Exception:
                beta0 = [1.0, 0.0]

    data = RealData(x, y, sx=sx, sy=sy)
    odr = ODR(data, model, beta0=beta0, maxit=maxit)
    out = odr.run()

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
        "model_func": model.fcn
    }

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
    Plot data with errorbars and ODR fit curve. fit_result must include 'model_func' and 'params'.
    """
    if obs_name not in OBSERVATIONS:
        raise KeyError(f"Observation '{obs_name}' not loaded.")
    df = OBSERVATIONS[obs_name]

    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)
    sx = np.zeros_like(x)
    sy = np.zeros_like(y)
    if x_err_col: sx = np.asarray(df[x_err_col], dtype=float)
    if y_err_col: sy = np.asarray(df[y_err_col], dtype=float)

    model_func = fit_result.get("model_func")
    if model_func is None:
        raise ValueError("fit_result does not contain 'model_func'. Re-run fit_odr().")

    params = fit_result.get("params")
    if params is None:
        raise ValueError("fit_result missing 'params'.")

    x_min = np.min(x)
    x_max = np.max(x)
    span = x_max - x_min
    if span == 0:
        span = abs(x_min) if x_min != 0 else 1.0
    xs = np.linspace(x_min - 0.05 * span, x_max + 0.05 * span, 500)

    try:
        ys_fit = np.asarray(model_func(params, xs), dtype=float)
    except TypeError:
        try:
            ys_fit = np.asarray(model_func(xs, params), dtype=float)
        except Exception as e:
            raise RuntimeError("model_func(params, xs) failed — model function has unexpected signature.") from e
    except Exception as e:
        raise RuntimeError("Evaluation of model function failed.") from e

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

    imgdir = ensure_image_dir(image_dir)
    safe_name = save_name or f"{_sanitize_name(obs_name)}__{_sanitize_name(x_col)}_vs_{_sanitize_name(y_col)}.png"
    outpath = os.path.join(imgdir, safe_name)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved: {outpath}")
    return outpath

# simple plot (no fit)
def plot(obs_name: str,
         x_col: str,
         y_col: str,
         x_err_col: Optional[str] = None,
         y_err_col: Optional[str] = None,
         title: Optional[str] = None,
         image_dir: str = "images",
         points_label: Optional[str] = None,
         save_name: Optional[str] = None,
         dpi: int = 300):
    if obs_name not in OBSERVATIONS:
        raise KeyError(f"Observation '{obs_name}' not loaded.")

    df = OBSERVATIONS[obs_name]
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"{x_col} or {y_col} not found in observation '{obs_name}'.")

    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)

    sx = None
    sy = None
    if x_err_col and x_err_col in df.columns:
        sx = np.asarray(df[x_err_col], dtype=float)
    if y_err_col and y_err_col in df.columns:
        sy = np.asarray(df[y_err_col], dtype=float)

    fig, ax = plt.subplots(figsize=(6,4))
    if sx is not None or sy is not None:
        ax.errorbar(x, y, xerr=sx, yerr=sy, fmt='o', markersize=4, label=points_label or "data")
    else:
        ax.plot(x, y, 'o', markersize=4, label=points_label or "data")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title is None:
        title = f"{x_col} vs {y_col} — {obs_name}"
    ax.set_title(title)
    ax.legend(loc='best')

    imgdir = ensure_image_dir(image_dir)
    safe_name = save_name or f"{_sanitize_name(obs_name)}__{_sanitize_name(x_col)}_vs_{_sanitize_name(y_col)}.png"
    outpath = os.path.join(imgdir, safe_name)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved: {outpath}")
    return outpath

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
        if obs in OBS_DISTRIBUTION:
            lines.append("  distribution tokens: " + ", ".join(f"{k}: {v}" for k, v in OBS_DISTRIBUTION[obs].items()))
    return "\n".join(lines)

