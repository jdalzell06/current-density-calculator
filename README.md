# Current Density Calculator

**Current Density Calculator** is a Python function to compute the 2D **current density** (jx, jy) and magnitude (j_norm) from a magnetic field map using FFT and NV projection. It includes internal edge-padding, Hanning window filtering, and returns k-space vectors along with the cropped magnetic field.

## Features
- Converts NV angles to Cartesian projection.
- Computes current density in k-space and inverse FFT.
- Hanning window filtering to reduce artifacts.
- Automatically crops results to original image size.
- Returns all essential arrays for analysis or plotting.

## Installation
Requires **numpy**. Install via pip if needed:

## Example Usage 

```python
import numpy as np
from current_density import CurrentDensityCalculator

B_field = np.random.rand(64,64) * 1e-4  # Tesla
calculator = CurrentDensityCalculator(B_field=B_field, scan_size_x=10e-6, scan_size_y=10e-6)
results = calculator.compute()

jx = results['jx']
jy = results['jy']
j_norm = results['j_norm']

```

## References
[1] C. J. McCluskey, J. Dalzell, A. Kumar, and J. M. Gregg, “Current Flow Mapping in Conducting Ferroelectric Domain Walls Using Scanning NV‐Magnetometry,” Advanced Electronic Materials, Jun. 2025, doi:10.1002/aelm.202500142.

[2] D. A. Broadway et al., “Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements,” Physical Review Applied, vol. 14, no. 2, p. 024076, Aug. 2020, doi:10.1103/PhysRevApplied.14.024076.

