import cv2
import numpy as np
import glob

# Chessboard params
pattern_size = (10, 7) 
square_size = 15 # Size of a square in mm

# Prepare 3D points (z=0)
# objp will have size (pattern_size, 3), each line is (x, y, 0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in world coordinates
imgpoints = []  # 2D points in image coordinates

images = glob.glob('data/calib/*.jpeg') + glob.glob('data/calib/*.png')

if not images:
    print("Aucune image trouvée dans data/calib. Vérifie l'extension ou le chemin.")
    exit(1)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        corners_subpix = cv2.cornerSubPix(gray, corners, winSize=(11,11), zeroZone=(-1,-1),
                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners_subpix)

        cv2.drawChessboardCorners(img, pattern_size, corners_subpix, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)  # affiche chaque résultat 100 ms
    else:
        print(f"No corners in image {fname}")

cv2.destroyAllWindows()

# Calibration de la caméra
# image_size = (largeur, hauteur) en pixels
img_shape = gray.shape[::-1]  # (width, height)
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

if not ret:
    print("Calibration failed")
    exit(1)

# (Optionnel) Sauvegarder K et dist_coeffs dans un fichier npz
np.savez('calibration_data.npz', K=K, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
print("\nCalibration complete, results saved in calibration_data.npz")
