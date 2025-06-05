# ======================================= IMPORTS =======================================
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from functions.image import extract_features, undistort_images
from functions.viz import plot_3D_points, view_camera_trajectory, view_mesh_file
from functions.obj import points_to_file, points_to_faces_to_file
from constants import *

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
def get_3d_points_from_images(images_obj, K, dist_coeffs=None):
    images_obj = undistort_images(images_obj, K, dist_coeffs)
    kp_list, desc_list = extract_features(images_obj)
    
    all_points_3D = []
    all_colors = []
    
    bf = cv2.BFMatcher()
    poses_rel = []
    
    # initial pose
    R_global = np.eye(3)
    t_global = np.zeros((3,1))
    
    for i in range(len(images_obj) - 1):
        desc1 = desc_list[i]
        desc2 = desc_list[i+1]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        img1 = images_obj[i]
        img2 = images_obj[i+1]
        
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        pts1, pts2 = [], []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        # essential matrix (threshold is distance from point to epipolar line in pixels)
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.5)
        inliers = mask.ravel() > 0
        
        # filter matches by RANSAC inliers
        ransac_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

        image1_points = []
        image2_points = []

        for match in ransac_matches:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            image1_points.append(pt1)
            image2_points.append(pt2)

        # convert image points to numpy arrays
        image1_points = np.array(image1_points, dtype=np.float32)
        image2_points = np.array(image2_points, dtype=np.float32)

        # retrieve relative pose (t_rel is unit‐norm, apply known baseline scale)
        _, R_rel, t_rel, _ = cv2.recoverPose(E, image1_points, image2_points, K)
        poses_rel.append((R_rel, t_rel))

        # chain global poses
        t_global = t_global + R_global @ t_rel
        R_global = R_global @ R_rel

        # global projection matrices (taking the first image as reference)
        # P1 is the projection matrix of the first camera (identity)
        # P2 is the projection matrix of the second camera (R_global, t_global)
        P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K @ np.hstack((R_global, t_global))
        
        # triangulation
        pts4D = cv2.triangulatePoints(P1, P2, image1_points.T, image2_points.T)
        pts3D = pts4D[:3] / pts4D[3]
        pts3D = pts3D.T

        # colors
        colors = []
        for pt in image1_points:
            x, y = int(pt[0]), int(pt[1])
            colors.append(img1[y, x])
        colors = np.array(colors)
        
        all_points_3D.append(pts3D)
        all_colors.append(colors)
    
    all_points_3D = np.vstack(all_points_3D)
    all_colors = np.vstack(all_colors)

    view_camera_trajectory(poses_rel)
    
    return all_points_3D, all_colors

def main():
    # ----------------------- Calibration results -----------------------
    calibration_data = np.load(CALIB_SAVE_FOLDER + 'calibration_data.npz')
    K = calibration_data['K']
    dist_coeffs = calibration_data['dist_coeffs']

    # ----------------------- 3D Reconstruction -----------------------

    # read images
    image_files = sorted(glob.glob(OBJECT_2_IMAGES_FOLDER + "*.png"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not image_files:
        raise FileNotFoundError("No images found in the specified folder.")
    if len(image_files) < 2:
        raise ValueError("At least two images are required for 3D reconstruction.")
    images = [cv2.imread(file) for file in image_files]

    # triangulate 3D points from images
    points3D, colors = get_3d_points_from_images(images, K, dist_coeffs)

    # plot the 3D points
    plot_3D_points(points3D, colors)

    # save the points to a file
    points_to_file(POINTS_SAVE_FOLDER + "3D_points_with_colors.obj", points3D, colors)

    # save the mesh from the points
    points_to_faces_to_file(POINTS_SAVE_FOLDER + "3D_points_with_colors.obj", MESH_SAVE_FOLDER + "3D_faces.obj")

    # display the reconstructed 3D mesh
    view_mesh_file(MESH_SAVE_FOLDER + "3D_faces.obj")

# ======================================= MAIN ==========================================
if __name__ == '__main__':
    main()
