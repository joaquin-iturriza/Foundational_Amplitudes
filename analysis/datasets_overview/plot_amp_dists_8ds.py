"""Log-amplitude distributions of the 8 base pretraining datasets, one color
per dataset (talk figure)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
DATASETS = [
    ("ee_aa_10-1000GeV_amplitudes",    r"$ee\to\gamma\gamma$"),
    ("ee_aaa_10-1000GeV_amplitudes",   r"$ee\to\gamma\gamma\gamma$"),
    ("ee_uu_91-1000GeV_amplitudes",    r"$ee\to u\bar u$"),
    ("ee_uug_91-1000GeV_amplitudes",   r"$ee\to u\bar u g$"),
    ("ee_uugg_91-1000GeV_amplitudes",  r"$ee\to u\bar u gg$"),
    ("ee_ttbar_346-1000GeV_amplitudes", r"$ee\to t\bar t$"),
    ("ee_WW_162-1000GeV_amplitudes",   r"$ee\to WW$"),
    ("ee_wwz_255-1000GeV_amplitudes",  r"$ee\to WWZ$"),
]
plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "legend.fontsize": 12})
fig, ax = plt.subplots(figsize=(8.2, 4.6))
for i, (name, label) in enumerate(DATASETS):
    amp = np.load(f"{ROOT}/data/{name}.npy")[:, -1]
    ax.hist(np.log10(np.abs(amp)), bins=120, histtype="step", lw=1.8,
            density=True, color=f"C{i}", label=label)
ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$")
ax.set_ylabel("density")
ax.set_xlim(-11.5, 2.5)
ax.set_yscale('log')
ax.legend(ncol=2, framealpha=0.9)
fig.tight_layout()
base = os.path.join(os.path.dirname(__file__), "amp_dists_8ds")
fig.savefig(base + ".png", dpi=200); fig.savefig(base + ".pdf")
print("saved", base)
