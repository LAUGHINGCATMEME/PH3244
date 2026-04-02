# calc_and_plot_with_fits.py
import numpy as np
import matplotlib.pyplot as plt

# --- DATA (copied from your calculations.py) ---
VGS_a = np.array([0.0, 0.197, 0.397, 0.604, 0.807, 1.0, 1.199])
ID_a  = np.array([7.17, 5.78, 4.41, 3.17, 2.14, 0.98, 0.64])
VGS_b = np.array([0.0, 0.202, 0.402, 0.684, 0.799, 0.999, 1.203])
ID_b  = np.array([8.58, 6.86, 5.23, 3.94, 2.76, 1.75, 0.74])

VDS_a = np.array([-1.05, -2.01, -3.55, -4.04, -5.08, -6.02, -7.03, -8.08, -9.02, -10.01, -11.01, -12.07, -13.04])
IGS_a = np.array([5.53, 7.15, 7.87, 8.27, 8.59, 8.79, 9.0, 9.16, 9.3, 9.42, 9.52, 9.59, 9.79])

VDS_b = np.array([-1.09, -1.96, -3.03, -3.97, -5.05, -6.0, -7.04, -8.05, -8.98, -10.02])
IGS_b = np.array([0.08, 0.11, 0.15, 0.17, 0.2, 0.22, 0.25, 0.27, 0.3, 0.33])

VDS_c = np.array([-1.0, -2.03, -3.14, -4.15, -5.03, -6.03, -7.12, -8.01, -9.54, -10.0, -11.1, -12.04])
IGS_c = np.array([1.01, 1.3, 1.51, 1.65, 1.76, 1.88, 2.0, 2.09, 2.19, 2.27, 2.37, 2.46])

# --- helper: linear fit with uncertainty using numpy.polyfit(cov=True) ---
def linfit_with_uncertainty(x, y):
    # returns (slope, slope_err, intercept, intercept_err, cov)
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    m, b = coeffs
    # covariance matrix is returned scaled by residual variance;
    # numpy.polyfit returns cov matrix if cov=True (or None)
    m_err = np.sqrt(cov[0,0])
    b_err = np.sqrt(cov[1,1])
    return m, m_err, b, b_err, cov

# --- Part 1: fit I vs VDS for VDS < -5 ---
vds_datasets = [
    ("VGS = 0.000 V", VDS_a, IGS_a),
    ("VGS = 1.558 V", VDS_b, IGS_b),
    ("VGS = 1.000 V", VDS_c, IGS_c),
]

part1_results = []
for label, x, y in vds_datasets:
    mask = x < -5.0
    xs = x[mask]
    ys = y[mask]
    if xs.size < 2:
        print(f"{label}: not enough points with VDS < -5 to fit.")
        part1_results.append((label, None))
        continue
    m, m_err, b, b_err, cov = linfit_with_uncertainty(xs, ys)
    part1_results.append((label, len(xs), m, m_err, b, b_err))
    print(f"{label} (VDS < -5, n={xs.size}): slope = {m:.5f} ± {m_err:.5f} mA/V, intercept = {b:.5f} ± {b_err:.5f}")

# --- Part 2: fit ID vs VGS (full range) ---
vgs_datasets = [
    ("VDS = -2.65 V", VGS_a, ID_a),
    ("VDS = -5.00 V", VGS_b, ID_b),
]
part2_results = []
for label, x, y in vgs_datasets:
    m, m_err, b, b_err, cov = linfit_with_uncertainty(x, y)
    part2_results.append((label, m, m_err, b, b_err))
    print(f"{label}: slope = {m:.5f} ± {m_err:.5f} mA/V, intercept = {b:.5f} ± {b_err:.5f}")

# --- Plotting: save images without fits and with fits/annotations ---
# OBS1: ID vs VGS (two datasets)
plt.figure(figsize=(6,4))
plt.plot(VGS_a, ID_a, 'o', label=r'$V_{DS}=-2.65\,$V', ls=':')
plt.plot(VGS_b, ID_b, 'o', label=r'$V_{DS}=-5.00\,$V', ls='-')
plt.xlabel(r'$V_{GS}$ (V)')
plt.ylabel(r'$I_D$ (mA)')
plt.legend()
plt.tight_layout()
plt.savefig('obs1_nofit.png', dpi=600)
plt.close()

# OBS1 with fits
plt.figure(figsize=(6,4))
plt.plot(VGS_a, ID_a, 'o', label=r'$V_{DS}=-2.65\,$V', ls=':')
plt.plot(VGS_b, ID_b, 'o', label=r'$V_{DS}=-5.00\,$V', ls='-')

# add fit lines
xfit = np.linspace(min(VGS_a.min(), VGS_b.min())-0.05,
                   max(VGS_a.max(), VGS_b.max())+0.05, 200)

for label, m, m_err, b, b_err in part2_results:
    yfit = m*xfit + b
    plt.plot(xfit, yfit, '-')

m1, m1err, b1, b1err = part2_results[0][1], part2_results[0][2], part2_results[0][3], part2_results[0][4]
m2, m2err, b2, b2err = part2_results[1][1], part2_results[1][2], part2_results[1][3], part2_results[1][4]
plt.plot(xfit, m1*xfit + b1, '-', alpha=0.8)
plt.plot(xfit, m2*xfit + b2, '-', alpha=0.8)
plt.text(0.02, 0.95, f"{vgs_datasets[0][0]}: m={m1:.4f}±{m1err:.4f} mA/V", transform=plt.gca().transAxes, va='top', fontsize=9)
plt.text(0.02, 0.88, f"{vgs_datasets[1][0]}: m={m2:.4f}±{m2err:.4f} mA/V", transform=plt.gca().transAxes, va='top', fontsize=9)
plt.xlabel(r'$V_{GS}$ (V)')
plt.ylabel(r'$I_D$ (mA)')
plt.legend()
plt.tight_layout()
plt.savefig('obs1_withfit.png', dpi=600)
plt.close()

# OBS2: I vs VDS (three datasets)
plt.figure(figsize=(6,4))
plt.plot(VDS_a, IGS_a, 'o', label=r'$V_{GS}=0.000\,$V', ls=':')
plt.plot(VDS_c, IGS_c, 'o', label=r'$V_{GS}=1.000\,$V', ls='--')
plt.plot(VDS_b, IGS_b, 'o', label=r'$V_{GS}=1.558\,$V', ls='-')
plt.xlim(-14, 1)
plt.xlabel(r'$V_{DS}$ (V)')
plt.ylabel(r'$I_D$ (mA)')
plt.legend()
plt.tight_layout()
plt.savefig('obs2_nofit.png', dpi=600)
plt.close()

# OBS2 with fits (only showing fit for full x-range and marking the subrange used VDS < -5)
plt.figure(figsize=(6,4))
plt.plot(VDS_a, IGS_a, 'o', label=r'$V_{GS}=0.000\,$V', ls=':')
plt.plot(VDS_c, IGS_c, 'o', label=r'$V_{GS}=1.000\,$V', ls='--')
plt.plot(VDS_b, IGS_b, 'o', label=r'$V_{GS}=1.558\,$V', ls='-')

# compute and draw fits for the VDS < -5 subsets and annotate slope ± err
for (label, x, y), res in zip(vds_datasets, part1_results):
    if res is None or res[1] is None:
        continue
    _, npts, m, m_err, b, b_err = res
    xfit = np.linspace(-14, -4.5, 200)
    plt.plot(xfit, m*xfit + b, '-', linewidth=1.2)
    # annotate each fit near the fitted region
    # choose an x position for the annotation near median of fitted points
    mask = x < -5.0
    if mask.sum() > 0:
        xm = x[mask].mean()
        ym = m*xm + b
        plt.text(xm, ym, f"m={m:.4f}±{m_err:.4f}", fontsize=8, ha='center', va='bottom')

plt.xlim(-14, 1)
plt.xlabel(r'$V_{DS}$ (V)')
plt.ylabel(r'$I_D$ (mA)')
plt.legend()
plt.tight_layout()
plt.savefig('obs2_withfit.png', dpi=600)
plt.close()

print("\nSaved images: obs1_nofit.png, obs1_withfit.png, obs2_nofit.png, obs2_withfit.png")
