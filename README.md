# MPDV-toobox

A toolbox for processing and visualization of multipoint PDV data.

## Environment setup

This repository is now structured as an editable Python package with its own environment files.

### Option 1: Conda environment

```bash
conda env create -f environment.yml
conda activate mpdv-toolbox
```

### Option 2: Pip environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Development install

```bash
pip install -e .
```

### Included scientific stack

The environment includes the common scientific modules typically used for analysis and visualization:

- numpy
- scipy
- pandas
- matplotlib
- seaborn
- h5py
- scikit-image
- xarray
- jupyterlab
- ipykernel
- pytest

### Input data

Preprocessed PDV data files using openmsi-ALPSS. Velocity-time, Displacement-time, etc.