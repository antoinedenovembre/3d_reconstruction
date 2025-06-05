# ======================================= IMPORTS =======================================
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import os
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

def read_obj_vertices(filename):
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
    return np.array(vertices)

def write_obj_with_faces(filename, vertices, faces):
    with open(filename, 'w') as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            # +1 car l’indexation .obj commence à 1
            file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

def reconstruct_faces_from_points(obj_input, obj_output):
    vertices = read_obj_vertices(obj_input)

    # ConvexHull génère les faces triangulées du nuage
    hull = ConvexHull(vertices)
    faces = hull.simplices  # indices des sommets formant chaque triangle

    write_obj_with_faces(obj_output, vertices, faces)
    print(f"Fichier avec faces écrit dans : {obj_output}")

def reconstruct(images_obj, K, dist_coeffs=None):
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
    save_obj_with_colors(save + "points/3D_points_with_colors.obj", points3D, colors)
    plot_3D_points(points3D, colors)

    reconstruct_faces_from_points(save + "points/3D_points_with_colors.obj", save + "points/3D_faces.obj")

    # display the reconstructed 3D mesh with open3d
    mesh = o3d.io.read_triangle_mesh(save + "points/3D_faces.obj")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# ======================================= MAIN ==========================================
if __name__ == '__main__':
    main()
