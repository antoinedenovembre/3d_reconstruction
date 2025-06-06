# ======================================= IMPORTS =======================================
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

# Options
ENABLE_VIEW = True # Set to True to view the reconstruction process

# ======================================= FUNCTIONS ======================================
def plot_3D_points_colors(points3D, colors):
    if not ENABLE_VIEW:
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # if colors are in BGR (OpenCV), convert to RGB for matplotlib
    colors_rgb = colors[:, ::-1] / 255.0  # Convert BGR -> RGB and [0,1]

    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], 
            c=colors_rgb, s=1, marker='o')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud with Colors")

    plt.tight_layout()
    plt.show()

def plot_3D_points(points3D):
    if not ENABLE_VIEW:
        return
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='orange', s=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D point cloud")
    plt.tight_layout()
    plt.show()

def view_camera_trajectory(poses_rel):
    if not ENABLE_VIEW:
        return

    # initialization of absolute camera positions and orientations
    camera_positions = [np.zeros((3, 1))]  # the first camera is at the origin
    R_abs = [np.eye(3)]  # the first rotation is identity

    for i, (R_rel, t_rel) in enumerate(poses_rel):
        # previous absolute position
        R_prev = R_abs[-1]
        pos_prev = camera_positions[-1]

        # new position (note: translation is in the previous frame's coordinate system)
        pos_new = pos_prev + R_prev.T @ t_rel  # or R_prev @ t_rel depending on convention
        camera_positions.append(pos_new)

        # new orientation
        R_new = R_rel @ R_prev
        R_abs.append(R_new)

    camera_positions = np.hstack(camera_positions)  # 3 x N

    # 3D plot of camera trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(camera_positions[0], camera_positions[1], camera_positions[2], marker='o')
    ax.set_title("Camera Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('equal')
    plt.show()

def view_mesh_file(filename):
    if not ENABLE_VIEW:
        return

    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)