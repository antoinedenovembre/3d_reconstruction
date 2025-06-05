# ======================================= IMPORTS =======================================
import numpy as np
from scipy.spatial import ConvexHull
import os
import logging

# ======================================= FUNCTIONS ======================================
def file_to_vertices(filename):
    """ Read vertices from an OBJ file. """
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
    return np.array(vertices)

def faces_to_file(filename, vertices, faces):
    """ Write vertices and faces to an OBJ file. """
    # mkdir if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            # +1 because indices in OBJ files start at 1
            file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    logging.info(f"Faces file written to {filename}")

def points_to_faces_to_file(obj_input, obj_output):
    """ Convert an OBJ file with vertices to a new OBJ file with faces (convex hull). """
    vertices = file_to_vertices(obj_input)

    hull = ConvexHull(vertices)
    faces = hull.simplices  # indices of the vertices forming each triangle

    faces_to_file(obj_output, vertices, faces)

def points_to_file_colors(filename, points3D, colors):
    """ Write 3D points with colors to an OBJ file. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for p, c in zip(points3D, colors):
            # colors are in BGR format (OpenCV), convert to RGB
            r, g, b = c[2], c[1], c[0]
            f.write(f"v {p[0]} {p[1]} {p[2]} {r/255:.4f} {g/255:.4f} {b/255:.4f}\n")
    logging.info(f"Points file written to {filename}")

def points_to_file(filename, points3D):
    """ Write 3D points to an OBJ file. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for p in points3D:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
    logging.info(f"Points file written to {filename}")