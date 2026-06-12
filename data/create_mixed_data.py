import numpy as np
import glob

# Get all .npy files in a folder
file_list = ["ee_uu_13000GeV_amplitudes.npy", "pp_Zj_13000GeV_amplitudes.npy"]

# Load arrays
arrays = [np.load(f) for f in file_list]

# Stack along a new axis (e.g., axis=0)
stacked = np.concatenate(arrays, axis=0)

# Save result
np.save("ee_uu__pp_Zj.npy", stacked)