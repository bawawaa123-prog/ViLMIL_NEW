# Inspect BiomedCLIP Features

A simple utility script is provided to inspect HDF5 features extracted by BiomedCLIP.

Script path:
- `tools/inspect_biomedclip_feature.py`

Purpose:
- Print dataset keys, shape and dtype
- Print per-feature and per-patch statistics (means, stds)
- Print L2 norm distribution and top-N patches by norm
- (Optional) Generate PCA 2D scatter & norm histogram images
- (Optional) Save a sample of the features as `.npy`

Dependencies:
- Python packages: `h5py`, `numpy`
- Optional: `matplotlib`, `scikit-learn` if you want plotting

Install required packages (recommended):

```powershell
# Windows cmd / PowerShell
C:/Users/lenovo/anaconda3/Scripts/conda.exe run -p C:\Users\lenovo\anaconda3 --no-capture-output python -m pip install h5py numpy
C:/Users/lenovo/anaconda3/Scripts/conda.exe run -p C:\Users\lenovo\anaconda3 --no-capture-output python -m pip install matplotlib scikit-learn
```

Usage (example):

```powershell
# Basic inspect
python tools/inspect_biomedclip_feature.py --file D:/FenLei/ViLa-MIL-main/features_biomedclip_5x/2463643-B.h5

# Inspect and create plots + save first 1000 features as npy
python tools/inspect_biomedclip_feature.py --file D:/FenLei/ViLa-MIL-main/features_biomedclip_5x/2463643-B.h5 --plot --save-npy --sample-limit 1000
```

Notes:
- The script prefers dataset name `features` inside the HDF5 file but uses the first dataset if `features` is not present.
- The script looks for `coords` dataset (optional) and uses them for patch coordinate outputs if present.
- Saved PCA and histogram images will be created next to the HDF5 file and suffixed with `_pca_scatter.png` and `_norms_hist.png`.

If you'd like, I can: 
- Run the inspect script and show the output (requires you to install the dependencies), or
- Add an extended viewer (e.g., interactive plots with plotly or saving a CSV summary).