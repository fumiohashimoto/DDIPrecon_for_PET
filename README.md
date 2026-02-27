# PET Image Reconstruction Using Deep Diffusion Image Prior (DDIP)

[Fumio Hashimoto](https://sites.google.com/view/fumio-hashimoto/) and [Kuang Gong](https://gong-lab.com/),

This repository provides code for diffusion-based PET image
reconstruction using the Deep Diffusion Image Prior (DDIP) framework.

Paper: F. Hashimoto and K. Gong, PET Image Reconstruction Using Deep
Diffusion Image Prior, IEEE Transactions on Medical Imaging, 2026.
https://ieeexplore.ieee.org/document/11369249

## Requirements

Create the environment using:

```
conda env create -f environment.yml
conda activate ddip
```

## Running Reconstruction

1.  Confirm the model checkpoint and simulation data exist:

```
checkpoints/model1100000.pt
data/PET/brainweb/negative_MRI_mmr/
```

2.  Run:

```
bash run_ddiprecon_2d.sh
```

3.  Output:

```
results/result.npy
```

## Notes

-   PET forward and back projection are implemented using parallelproj (https://parallelproj.readthedocs.io/).
-   Current implementation supports 2D experiments.

## Citation

If you use this code in academic work, please cite:

```
@article{Hashimoto2026DDIP,
  author   = {Hashimoto, Fumio and Gong, Kuang},
  journal  = {IEEE Transactions on Medical Imaging},
  title    = {PET Image Reconstruction Using Deep Diffusion Image Prior},
  year     = {2026},
  doi      = {10.1109/TMI.2026.3659792}
}
```

## Acknowledgements

- Base code reference: https://github.com/hyungjin-chung/DDIP3D
- PET projector implementation based on: https://parallelproj.readthedocs.io/
- Simulation data: https://brainweb.bic.mni.mcgill.ca/

## Contact

For any questions, please contact fumio.hashimo@ufl.edu.