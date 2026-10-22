"""Print the arrays (shape, mean) of .npz files: python analysis/catalog_daniel/peek_npz.py f.npz [...]"""
import sys
import numpy as np
for f in sys.argv[1:]:
    z = np.load(f)
    print(f.split("/")[-1], {k: (z[k].shape, round(float(z[k].mean()), 4) if z[k].size else None) for k in z.files})
