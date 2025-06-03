# ======================================= IMPORTS =======================================
import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import open3d as o3d

# ======================================= FUNCTIONS ======================================
def extract_features(images):
    # Tweak SIFT parameters to get more features
    sift = cv2.SIFT_create(
        nfeatures=5000,          # max number of features
        contrastThreshold=0.01,  # lower = more features
        edgeThreshold=6,         # default=10, lower = more edge-like features
        sigma=1.6                # default smoothing
    )

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

def save_obj_with_faces(filename, points3D, faces):
    """
    points3D : liste de points [ [x, y, z], ... ]
    faces : liste de faces [ [i1, i2, i3], ... ] avec i1, i2, i3 étant les indices des points (0-based)
    """
    with open(filename, 'w') as f:
        for p in points3D:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def generate_faces_delaunay(points3D):
    points2D = np.array([[p[0], p[1]] for p in points3D])
    tri = Delaunay(points2D)
    return tri.simplices.tolist()

def reconstruct(images_obj, K, dist_coeffs=None):
    # images_obj = undistort_images(images_obj, K, dist_coeffs)
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
        first_match = ransac_matches[0] if ransac_matches else None

        # DEBUG: for every image pair, visualize the first match to ensure the correspondence of index
        if len(good_matches) > 0:
            print(f"Image pair {i} - {i+1}: First match queryIdx={first_match.queryIdx}, trainIdx={first_match.trainIdx}")
            img_matches = cv2.drawMatches(
                img1, kp1, img2, kp2,
                [first_match], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            os.makedirs("output/matches", exist_ok=True)
            cv2.imwrite(f"output/matches/matches_{i}_{i+1}.png", img_matches)
        
        if first_match is not None:
            pt1 = tuple(map(int, kp1[first_match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[first_match.trainIdx].pt))
            tmp1 = img1.copy()
            tmp2 = img2.copy()
            cv2.circle(tmp1, pt1, 5, (0, 255, 0), -1)
            cv2.circle(tmp2, pt2, 5, (0, 255, 0), -1)
            cv2.imwrite(f'output/matches/points_{i}_{i+1}.png', np.hstack((tmp1, tmp2)))
        # img_matches = cv2.drawMatches(
        #     img1, kp1, img2, kp2,
        #     ransac_matches, None,
        #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        # )
        # os.makedirs("output/matches", exist_ok=True)
        # cv2.imwrite(f"output/matches/matches_{i}_{i+1}.png", img_matches)

        image1_points = []
        image2_points = []

        for match in ransac_matches:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            image1_points.append(pt1)
            image2_points.append(pt2)
            
        tmp1 = img1.copy()
        tmp2 = img2.copy()
        # convert image points to numpy arrays
        image1_points = np.array(image1_points, dtype=np.float32)
        image2_points = np.array(image2_points, dtype=np.float32)
        # use numpy array indexing for coordinates
        cv2.circle(
            tmp1,
            (int(image1_points[0, 0]), int(image1_points[0, 1])),
            5, (0, 255, 0), -1
        )
        cv2.circle(
            tmp2,
            (int(image2_points[0, 0]), int(image2_points[0, 1])),
            5, (0, 255, 0), -1
        )
        cv2.imwrite(f'output/matches/points_verif_{i}_{i+1}.png', np.hstack((tmp1, tmp2)))

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

    # Initialisation des poses absolues
    camera_positions = [np.zeros((3, 1))]  # La première caméra est à l'origine
    R_abs = [np.eye(3)]  # La première rotation est identité

    for i, (R_rel, t_rel) in enumerate(poses_rel):
        # Position absolue précédente
        R_prev = R_abs[-1]
        pos_prev = camera_positions[-1]

        # Nouvelle position (attention : la translation est dans le repère précédent)
        pos_new = pos_prev + R_prev.T @ t_rel  # ou R_prev @ t_rel selon convention
        camera_positions.append(pos_new)

        # Nouvelle orientation
        R_new = R_rel @ R_prev
        R_abs.append(R_new)

    # Convertir en array pour plus de simplicité
    camera_positions = np.hstack(camera_positions)  # 3 x N

    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(camera_positions[0], camera_positions[1], camera_positions[2], marker='o')
    ax.set_title("Trajectoire caméra")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('equal')
    plt.show()
    
    return all_points_3D, all_colors

def save_obj_with_colors(filename, points3D, colors):
    with open(filename, 'w') as f:
        for p, c in zip(points3D, colors):
            # colors are in BGR format (OpenCV), convert to RGB
            r, g, b = c[2], c[1], c[0]
            f.write(f"v {p[0]} {p[1]} {p[2]} {r/255:.4f} {g/255:.4f} {b/255:.4f}\n")

def plot_3D_points(points3D, colors):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # if colors are in BGR (OpenCV), convert to RGB for matplotlib
    colors_rgb = colors[:, ::-1] / 255.0  # Convert BGR -> RGB and [0,1]

    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], 
            c=colors_rgb, s=1, marker='o')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstruction 3D")

    plt.tight_layout()
    plt.show()

def main():
    save = 'output/'

    # ----------------------- Calibration results -----------------------
    calibration_data = np.load(save + 'calibration_data.npz')
    K = calibration_data['K']
    dist_coeffs = calibration_data['dist_coeffs']

    # ----------------------- 3D Reconstruction -----------------------
    image_files = sorted(glob.glob("data/obj4/*.png"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    images_obj = [cv2.imread(file) for file in image_files]

    points3D, colors = reconstruct(images_obj, K, dist_coeffs)
    save_obj_with_colors("output/points/3D_points_with_colors.obj", points3D, colors)
    plot_3D_points(points3D, colors)

    faces = generate_faces_delaunay(points3D)
    save_obj_with_faces("output/points/3D_faces.obj", points3D, faces)

    # -- visualize in a window using Open3D --

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(points3D),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    # if you have per‐vertex colors:
    try:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, ::-1] / 255.0)
    except NameError:
        pass

    o3d.visualization.draw_geometries([mesh])

# ======================================= MAIN ==========================================
if __name__ == '__main__':
    main()
