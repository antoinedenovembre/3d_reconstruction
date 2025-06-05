# ======================================= IMPORTS =======================================
import matplotlib.pyplot as plt
import open3d as o3d

# ======================================= FUNCTIONS ======================================
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

def view_mesh_file(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)