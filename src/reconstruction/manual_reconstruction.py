# ======================================= IMPORTS =======================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial import ConvexHull
import os

# ----------------------------------------
# 1) Paramètres / Calibration
# ----------------------------------------
calib = np.load("output/calibration_data.npz")
K = calib["K"]
dist_coeffs = calib["dist_coeffs"]

# ----------------------------------------
# Paramètres généraux
# ----------------------------------------
N = 1  # nombre de paires d'images
NUM_POINTS = 52  # points par image (par paire)
CLICKED_POINTS_FILE = "output/points/clicked_points.npz"

# ----------------------------------------
# Chargement des images
# ----------------------------------------
images = []
for i in range(2 * N):
    img = cv2.imread(f"data/obj4/{i+1}.png")
    if img is None:
        raise RuntimeError(f"Impossible de charger l'image {i+1}.png")
    images.append(img)

# ----------------------------------------
# Fonctions utiles
# ----------------------------------------

def save_obj(filename, points3D):
    with open(filename, 'w') as f:
        for p in points3D:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

def resize_img(img, target_h, target_w):
    return cv2.resize(img, (target_w, target_h))

# ----------------------------------------
# Chargement ou initialisation des points cliqués
# ----------------------------------------
clicked_pts = [[] for _ in range(2 * N)]
if os.path.exists(CLICKED_POINTS_FILE):
    data = np.load(CLICKED_POINTS_FILE, allow_pickle=True)
    clicked_pts_loaded = data["clicked_pts"]
    clicked_pts = [list(pts) for pts in clicked_pts_loaded]
    print("Points cliqués rechargés depuis le fichier.")
else:
    print("Aucun fichier de points trouvé. Vous devrez cliquer les points.")

# ----------------------------------------
# Gestion des clics
# ----------------------------------------
def on_mouse(event, x, y, flags, param):
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
                print(f"Paire {current_pair} - Image {left_idx} : point {len(clicked_pts[left_idx])} = {(x, y)}")
                current_click_image = 1
        elif current_click_image == 1 and x >= half_w:
            if len(clicked_pts[right_idx]) < NUM_POINTS:
                clicked_pts[right_idx].append((x - half_w, y))
                cv2.circle(img_pair_display, (x, y), 5, (0, 0, 255), -1)
                print(f"Paire {current_pair} - Image {right_idx} : point {len(clicked_pts[right_idx])} = {(x - half_w, y)}")
                current_click_image = 0
        else:
            print(f"Cliquez dans la {'gauche' if current_click_image == 0 else 'droite'} de la fenêtre.")

# ----------------------------------------
# Traitement d'une paire
# ----------------------------------------
def process_pair(pair_idx):
    global img_pair_display, current_click_image, current_pair
    current_pair = pair_idx
    current_click_image = 0

    img_left = images[2 * pair_idx]
    img_right = images[2 * pair_idx + 1]

    # Redimensionnement
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    img_left_resized = resize_img(img_left, max_h, max_w)
    img_right_resized = resize_img(img_right, max_h, max_w)

    img_pair_display = np.hstack((img_left_resized, img_right_resized))

    left_idx = 2 * pair_idx
    right_idx = left_idx + 1

    if len(clicked_pts[left_idx]) < NUM_POINTS or len(clicked_pts[right_idx]) < NUM_POINTS:
        cv2.namedWindow("Image pair")
        cv2.setMouseCallback("Image pair", on_mouse)
        print(f"\n--- Paire {pair_idx} : cliquez {NUM_POINTS} points alternatifs dans Image {left_idx} (gauche, vert) puis Image {right_idx} (droite, rouge) ---")

        while True:
            cv2.imshow("Image pair", img_pair_display)
            key = cv2.waitKey(1) & 0xFF
            if len(clicked_pts[left_idx]) == NUM_POINTS and len(clicked_pts[right_idx]) == NUM_POINTS:
                print(f"Tous les points cliqués pour la paire {pair_idx}")
                break
            if key == 27:
                print("Interrompu par l'utilisateur.")
                exit(0)

        cv2.destroyAllWindows()

        # Sauvegarde des clics
        np.savez(CLICKED_POINTS_FILE, clicked_pts=np.array(clicked_pts, dtype=object))
        print(f"Points cliqués sauvegardés dans {CLICKED_POINTS_FILE}")
    else:
        print(f"Paire {pair_idx} déjà annotée. Traitement automatique.")

    # Conversion en float32 et correction distorsion
    pts1 = np.array(clicked_pts[left_idx], dtype=np.float32)
    pts2 = np.array(clicked_pts[right_idx], dtype=np.float32)
    pts1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist_coeffs, P=K).reshape(-1, 2)
    pts2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist_coeffs, P=K).reshape(-1, 2)

    # Calcul essentiel et pose
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, threshold=1.5)
    inliers = (mask.ravel() > 0)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

    # Matrices de projection
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R_rel, t_rel))

    # Triangulation
    pts4D_homo = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    pts3D = (pts4D_homo[:3] / pts4D_homo[3]).T

    # Sauvegarde
    save_obj(f"output/points/3D_points_pair{pair_idx}.obj", pts3D)

    print(f"\nPoints 3D triangulés pour la paire {pair_idx} (X,Y,Z) :")
    for i, p in enumerate(pts3D):
        print(f"  Point {i+1}: {p}")

    return pts3D

# ----------------------------------------
# Boucle principale sur toutes les paires
# ----------------------------------------
all_pts3D = []

for pair_idx in range(N):
    pts3D = process_pair(pair_idx)
    all_pts3D.append(pts3D)

all_pts3D = np.vstack(all_pts3D)

# Affichage 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_pts3D[:, 0], all_pts3D[:, 1], all_pts3D[:, 2], c='orange', s=30)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Nuage de points 3D regroupé de toutes les {N} paires")
plt.tight_layout()
plt.show()

print("Traitement terminé pour toutes les paires.")


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

# Exemple d'utilisation
input_file = 'output/points/3D_points_pair0.obj'
output_file = 'output/points/cube_with_faces.obj'
reconstruct_faces_from_points(input_file, output_file)