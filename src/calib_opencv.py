# ======================================= IMPORTS =======================================
import cv2
import numpy as np
import glob
from constants import CHESSBOARD_SIZE, CHESSBOARD_DIM

# ======================================= FUNCTIONS ======================================
def prepare_object_points(chessboard_size, square_size):
    """Prepare a single map of 3D object points for the chessboard (z=0)."""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def find_corners(images, chessboard_size, objp):
    """
    Iterate over all images, detect and refine chessboard corners.
    Returns objpoints, imgpoints, and image shape.
    """
    objpoints = []
    imgpoints = []
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not read image {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if not ret:
            print(f"No corners found in {fname}")
            continue

        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp)
        imgpoints.append(corners_subpix)

        cv2.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

    cv2.destroyAllWindows()
    return objpoints, imgpoints, img_shape

def calibrate_and_save(objpoints, imgpoints, img_shape, output_path):
    """Perform camera calibration and save results to .npz."""
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    print("Calibration reprojection error:", ret)
    if not ret:
        print("Calibration failed")
        exit(1)

    np.savez(output_path, K=K, dist_coeffs=dist_coeffs,
             rvecs=rvecs, tvecs=tvecs)
    print(f"Calibration complete. Data saved to {output_path}")

def main():
    image_pattern = 'data/calib/*.jpeg'
    output_file = 'output/calibration_data.npz'

    images = glob.glob(image_pattern)
    if not images:
        print(f"No images found in {image_pattern}")
        exit(1)

    objp = prepare_object_points(CHESSBOARD_SIZE, CHESSBOARD_DIM)
    objpoints, imgpoints, img_shape = find_corners(
        images, CHESSBOARD_SIZE, objp
    )
    calibrate_and_save(objpoints, imgpoints, img_shape, output_file)

# ======================================= MAIN ==========================================
if __name__ == "__main__":
    main()
