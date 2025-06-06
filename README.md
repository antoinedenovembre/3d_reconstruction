# 3D reconstruction from a set of images

This project implements a basic 3D reconstruction pipeline using OpenCV and Python. It includes camera calibration and reconstruction from a set of two images (the reconstruction with N images is not functional yet).

## Where to look to understand the project and the code ?

To understand the project and the code, you can start by looking at the report [3D Reconstruction Report](report/cr.pdf) which provides an overview of the methods used and the results obtained.

## How to install dependencies

First, make sure you have Python 3.11.x installed, any later version will mess with open3d.

To install the required dependencies for this project, you can use the following command:

```bash
pip install -r requirements.txt
```

## How to use and run the code

This project is structured as follows:

```
├── data/
│   ├── calib/
│   ├── obj/
│   └── obj_two_images/
├── src/
│   ├── calib/
│   │   └── calib_opencv.py
│   │   └── calib_zhang.py
│   ├── reconstruction/
│   │   └── auto_reconstruction.py
│   │   └── manual_reconstruction.py
```

You will first need to calibrate your camera using the `calib_opencv.py` script. The calibration results will be saved in the `data/calib/` directory. The `calib_zhang.py` script is the implementation of Zhang's method but does not implement the optimization step.

Then, you can use the `auto_reconstruction.py` script to perform automatic reconstruction from a set of images. This script will read the calibration data and reconstruct the 3D model. Note that 3D reconstruction with N images is not functional, so the folder selected for automatic reconstruction is `data/obj_two_images/`, which contains only two images. The script will save the reconstructed 3D model in the `data/obj/` directory, as both points and faces.

The best results are obtained with the `manual_reconstruction.py` script, which allows you to manually select points in two images and reconstruct the 3D model. This script will also save the reconstructed 3D model in the `data/obj/` directory.
## How to use and run the code

This project is structured as follows:

```
├── data/
│   ├── calib/
│   ├── obj/
│   └── obj_two_images/
├── src/
│   ├── calib/
│   │   └── calib_opencv.py
│   │   └── calib_zhang.py
│   ├── reconstruction/
│   │   └── auto_reconstruction.py
│   │   └── manual_reconstruction.py
```

You will first need to calibrate your camera using the `calib_opencv.py` script. The calibration results will be saved in the `data/calib/` directory. The `calib_zhang.py` script is the implementation of Zhang's method but does not implement the optimization step.

Then, you can use the `auto_reconstruction.py` script to perform automatic reconstruction from a set of images. This script will read the calibration data and reconstruct the 3D model. Note that 3D reconstruction with N images is not functional, so the folder selected for automatic reconstruction is `data/obj_two_images/`, which contains only two images. The script will save the reconstructed 3D model in the `data/obj/` directory, as both points and faces.

The best results are obtained with the `manual_reconstruction.py` script, which allows you to manually select points in two images and reconstruct the 3D model. This script will also save the reconstructed 3D model in the `data/obj/` directory.