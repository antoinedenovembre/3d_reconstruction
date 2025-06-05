# ======================================= IMPORTS =======================================
import cv2
import numpy as np
import numpy as np
import os
import logging

from functions.obj import points_to_faces_to_file, points_to_file
from functions.image import resize_image
from functions.viz import plot_3D_points
from constants import *

# ======================================= GLOBALS =======================================
N = 1  # number of image pairs to process
NUM_POINTS = 52  # points per image (per pair)
CLICKED_POINTS_FILE = POINTS_SAVE_FOLDER + "clicked_points.npz"
clicked_pts = [[] for _ in range(2 * N)]
K = None  # camera intrinsic matrix
dist_coeffs = None  # distortion coefficients
images = []  # list to hold the images

# ======================================= LOGGER SETUP =======================================
class BlueInfoFormatter(logging.Formatter):
    BLUE = "\033[34m"
    RESET = "\033[0m"
    def format(self, record):
        formatted = super().format(record)
        if record.levelno == logging.INFO:
            return f"{self.BLUE}{formatted}{self.RESET}"
        return formatted

handler = logging.StreamHandler()
handler.setFormatter(
    BlueInfoFormatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
)
logging.basicConfig(level=logging.INFO, handlers=[handler])

# ======================================= FUNCTIONS =======================================
def on_mouse(event, x, y, flags, param):
    """ Mouse callback function to handle clicks on the image pair display. """
    global img_pair_display, current_click_image, current_pair

    if event == cv2.EVENT_LBUTTONDOWN:
        max_h, max_w = img_pair_display.shape[:2]
        half_w = max_w // 2

        left_idx = 2 * current_pair
        right_idx = left_idx + 1

        if current_click_image == 0 and x < half_w:
            if len(clicked_pts[left_idx]) < NUM_POINTS:
                clicked_pts[left_idx].append((x, y))
                cv2.circle(img_pair_display, (x, y), 5, (0, 255, 0), -1)
                print(f"Pair n°{current_pair} - Image {left_idx} : point {len(clicked_pts[left_idx])} = {(x, y)}")
                current_click_image = 1
        elif current_click_image == 1 and x >= half_w:
            if len(clicked_pts[right_idx]) < NUM_POINTS:
                clicked_pts[right_idx].append((x - half_w, y))
                cv2.circle(img_pair_display, (x, y), 5, (0, 0, 255), -1)
                print(f"Pair n°{current_pair} - Image {right_idx} : point {len(clicked_pts[right_idx])} = {(x - half_w, y)}")
                current_click_image = 0
        else:
            logging.info(f"Click on the {'left' if current_click_image == 0 else 'right'} side of the window.")

def process_pair(pair_idx):
    """ Process a pair of images to click points on and triangulate 3D points. """
    global img_pair_display, current_click_image, current_pair
    current_pair = pair_idx
    current_click_image = 0

    img_left = images[2 * pair_idx]
    img_right = images[2 * pair_idx + 1]

    # resizing
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    img_left_resized = resize_image(img_left, max_h, max_w)
    img_right_resized = resize_image(img_right, max_h, max_w)

    img_pair_display = np.hstack((img_left_resized, img_right_resized))

    left_idx = 2 * pair_idx
    right_idx = left_idx + 1

    if len(clicked_pts[left_idx]) < NUM_POINTS or len(clicked_pts[right_idx]) < NUM_POINTS:
        cv2.namedWindow("Image pair")
        cv2.setMouseCallback("Image pair", on_mouse)
        logging.info(f"\n--- Pair n°{pair_idx} : click {NUM_POINTS} alternative points in Image {left_idx} (left, green) then Image {right_idx} (right, red) ---")

        while True:
            cv2.imshow("Image pair", img_pair_display)
            key = cv2.waitKey(1) & 0xFF
            if len(clicked_pts[left_idx]) == NUM_POINTS and len(clicked_pts[right_idx]) == NUM_POINTS:
                logging.info(f"All points clicked for pair {pair_idx}")
                break
            if key == 27:
                logging.info("Interrupted by user.")
                exit(0)

        cv2.destroyAllWindows()

        # save clicks
        np.savez(CLICKED_POINTS_FILE, clicked_pts=np.array(clicked_pts, dtype=object))
        logging.info(f"Clicked points saved to {CLICKED_POINTS_FILE}")
    else:
        logging.info(f"Pair n°{pair_idx} already annotated. Automatic processing.")

    # convert to float32 and undistort
    pts1 = np.array(clicked_pts[left_idx], dtype=np.float32)
    pts2 = np.array(clicked_pts[right_idx], dtype=np.float32)
    pts1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist_coeffs, P=K).reshape(-1, 2)
    pts2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist_coeffs, P=K).reshape(-1, 2)

    # essential matrix and pose calculation
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, threshold=1.5)
    inliers = (mask.ravel() > 0)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

    # projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R_rel, t_rel))

    # triangulation
    pts4D_homo = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    pts3D = (pts4D_homo[:3] / pts4D_homo[3]).T

    # save 3D points to file
    points_to_file(POINTS_SAVE_FOLDER + f"3D_points_pair{pair_idx}.obj", pts3D)

    logging.info(f"\nPoints 3D triangulated for pair {pair_idx} (X,Y,Z) :")
    for i, p in enumerate(pts3D):
        logging.info(f"  Point {i+1}: {p}")

    return pts3D

def main():
    global clicked_pts, K, dist_coeffs, images

    # ----------------------- Calibration results -----------------------
    calib = np.load(CALIB_SAVE_FOLDER + "calibration_data.npz")
    K = calib["K"]
    dist_coeffs = calib["dist_coeffs"]

    # ----------------------- 3D Reconstruction -----------------------
    # image loading
    images = []
    for i in range(2 * N):
        img = cv2.imread(OBJECT_DATA_FOLDER + f"{i+1}.png")
        if img is None:
            raise RuntimeError(f"Could not load image {i+1}.png")
        images.append(img)

    # load or initialize clicked points
    if os.path.exists(CLICKED_POINTS_FILE):
        data = np.load(CLICKED_POINTS_FILE, allow_pickle=True)
        clicked_pts_loaded = data["clicked_pts"]
        clicked_pts = [list(pts) for pts in clicked_pts_loaded]
        logging.info("Loaded clicked points from file.")
    else:
        logging.warning("No clicked points file found. You will need to click the points.")

    # loop over pairs and save meshes
    all_pts3D = []
    for pair_idx in range(N):
        pts3D = process_pair(pair_idx)
        all_pts3D.append(pts3D)
        input_file = POINTS_SAVE_FOLDER + f'3D_points_pair{pair_idx}.obj'
        output_file = POINTS_SAVE_FOLDER + f'cube_with_faces{pair_idx}.obj'
        points_to_faces_to_file(input_file, output_file)
    all_pts3D = np.vstack(all_pts3D)

    plot_3D_points(all_pts3D)   

# ======================================= MAIN =======================================
if __name__ == "__main__":
    main()