import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_features(images):
    sift = cv2.SIFT_create()

    keypoints_list = []
    descriptors_list = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(desc)

    return keypoints_list, descriptors_list

def undistort_images(images, K, dist_coeffs):
    undistorted_images = []
    for img in images:
        undistorted_img = cv2.undistort(img, K, dist_coeffs)
        undistorted_images.append(undistorted_img)
    
    return undistorted_images

def reconstruct(images_obj, K, dist_coeffs=None):
    images_obj = undistort_images(images_obj, K, dist_coeffs)
    kp_list, desc_list = extract_features(images_obj)
    
    all_points_3D = []
    all_colors = []
    
    bf = cv2.BFMatcher()
    
    # Pose initiale
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
        
        # Matrice essentielle (threshold is distance from point to epipolar line in pixels)
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
        
        inliers = mask.ravel() > 0
        pts1 = pts1[inliers]
        pts2 = pts2[inliers]
        
        # Filter matches by RANSAC inliers
        ransac_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2,
            ransac_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        os.makedirs("output/matches", exist_ok=True)
        cv2.imwrite(f"output/matches/matches_{i}_{i+1}.png", img_matches)
        
        # Récupérer pose relative entre i et i+1
        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1, pts2, K)
        
        # Chaîner les poses globales
        R_global = R_global @ R_rel
        t_global = t_global + R_global @ t_rel
        
        # Matrices de projection globales (on prend la premiere image comme référence)
        # P1 est la matrice de projection de la première caméra (identité)
        # P2 est la matrice de projection de la deuxième caméra (R_global, t_global)
        P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K @ np.hstack((R_global, t_global))
        
        # Triangulation
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3D = pts4D[:3] / pts4D[3]
        pts3D = pts3D.T
        
        # Couleurs
        colors = []
        for pt in pts1:
            x, y = int(pt[0]), int(pt[1])
            colors.append(img1[y, x])
        colors = np.array(colors)
        
        all_points_3D.append(pts3D)
        all_colors.append(colors)
    
    all_points_3D = np.vstack(all_points_3D)
    all_colors = np.vstack(all_colors)
    
    return all_points_3D, all_colors

def save_obj_with_colors(filename, points3D, colors):
    with open(filename, 'w') as f:
        for p, c in zip(points3D, colors):
            # c est en BGR, on convertit en RGB
            r, g, b = c[2], c[1], c[0]
            f.write(f"v {p[0]} {p[1]} {p[2]} {r/255:.4f} {g/255:.4f} {b/255:.4f}\n")

def plot_3D_points(points3D, colors):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Si les couleurs sont en BGR (OpenCV), les convertir en RGB pour matplotlib
    colors_rgb = colors[:, ::-1] / 255.0  # Conversion BGR -> RGB et [0,1]

    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], 
            c=colors_rgb, s=1, marker='o')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstruction 3D")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data = 'data/calib/'
    save = 'output/'

    # ----------------------- Calibration -----------------------
    
    calibration_data = np.load('output/calibration_data.npz')
    K = calibration_data['K']
    dist_coeffs = calibration_data['dist_coeffs']

    # ----------------------- 3D Reconstruction -----------------------

    image_files = sorted(glob.glob("data/object/*.png"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    images_obj = [cv2.imread(file) for file in image_files]

    points3D, colors = reconstruct(images_obj, K, dist_coeffs)
    save_obj_with_colors("output/points/3D_points_with_colors.obj", points3D, colors)
    plot_3D_points(points3D, colors)
