import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1) Paramètres / Calibration
# ----------------------------------------
# On suppose que vous avez déjà sauvegardé K et dist_coeffs dans 'calibration_data.npz'
calib = np.load("output/calibration_data.npz")
K = calib["K"]
dist_coeffs = calib["dist_coeffs"]

# Combien de points (coins) on va cliquer par image
NUM_POINTS = 10  # par exemple les 8 coins d'un cube

# ----------------------------------------
# 2) Fonctions utilitaires pour la souris
# ----------------------------------------
clicked_pts1 = []
clicked_pts2 = []
current_image = 1  # on commencera par cliquer dans la fenêtre "Image 1"

def save_obj(filename, points3D):
    with open(filename, 'w') as f:
        for p in points3D:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

def on_mouse_img1(event, x, y, flags, param):
    global clicked_pts1, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts1) < NUM_POINTS:
            clicked_pts1.append((x, y))
            cv2.circle(img_display1, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow("Image 1", img_display1)
            if len(clicked_pts1) == NUM_POINTS:
                current_image = 2
                print("Vous avez cliqué tous les points dans Image 1. Passez à Image 2.")

def on_mouse_img2(event, x, y, flags, param):
    global clicked_pts2
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts2) < NUM_POINTS:
            clicked_pts2.append((x, y))
            cv2.circle(img_display2, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("Image 2", img_display2)
            if len(clicked_pts2) == NUM_POINTS:
                print("Vous avez cliqué tous les points dans Image 2. Appuyez sur une touche pour trianguler.")


# ----------------------------------------
# 3) Charger et afficher les deux images
# ----------------------------------------
img1 = cv2.imread("data/object/0.png")  # première vue
img2 = cv2.imread("data/object/1.png")  # deuxième vue

if img1 is None or img2 is None:
    raise RuntimeError("Impossible de charger les images. Vérifiez les chemins.")

# On les duplique en copies pour pouvoir dessiner les points sans écraser l'original
img_display1 = img1.copy()
img_display2 = img2.copy()

cv2.namedWindow("Image 1")
cv2.setMouseCallback("Image 1", on_mouse_img1)
cv2.namedWindow("Image 2")
cv2.setMouseCallback("Image 2", on_mouse_img2)

print(f"Cliquez sur {NUM_POINTS} coins du cube dans 'Image 1' (points verts).")
cv2.imshow("Image 1", img_display1)

# On attend que l'utilisateur clique tous les points sur la première image,
# puis qu'il ferme la fenêtre Image 1 pour passer à Image 2.
while True:
    key = cv2.waitKey(1)
    if current_image == 2:
        break
    # si on ferme la fenêtre manuellement, on sort
    if cv2.getWindowProperty("Image 1", cv2.WND_PROP_VISIBLE) < 1:
        exit(0)

print(f"Maintenant, cliquez sur les {NUM_POINTS} coins correspondants dans 'Image 2' (points rouges).")
cv2.imshow("Image 2", img_display2)

while True:
    key = cv2.waitKey(1)
    # Si l'utilisateur a fini de cliquer tous les points, il peut simplement appuyer sur n'importe quelle touche
    if len(clicked_pts2) == NUM_POINTS and key != -1:
        break
    # si on ferme la fenêtre manuellement, on sort
    if cv2.getWindowProperty("Image 2", cv2.WND_PROP_VISIBLE) < 1:
        exit(0)

cv2.destroyAllWindows()

# ----------------------------------------
# 4) Préparation des points pour triangulation
# ----------------------------------------
pts1 = np.array(clicked_pts1, dtype=np.float32)  # shape (NUM_POINTS, 2)
pts2 = np.array(clicked_pts2, dtype=np.float32)

# Correction de la distorsion des points d'image
pts1 = cv2.undistortPoints(pts1.reshape(-1,1,2), K, dist_coeffs, P=K).reshape(-1,2)
pts2 = cv2.undistortPoints(pts2.reshape(-1,1,2), K, dist_coeffs, P=K).reshape(-1,2)

# ----------------------------------------
# 5) Calcul de la matrice essentielle + recoverPose
# ----------------------------------------
E, mask = cv2.findEssentialMat(pts1, pts1, cameraMatrix=K, method=cv2.RANSAC, threshold=1.5)
print("Matrice essentielle E :\n", E)

# Filtrer par les inliers retournés par RANSAC (optionnel)
inliers = (mask.ravel() > 0)
pts1_in = pts1[inliers]
pts2_in = pts2[inliers]

print("Mask des inliers (0=outlier, 1=inlier) :", mask.ravel().astype(int))
print("Nombre d’inliers :", np.count_nonzero(mask))

_, R_rel_inv, t_rel_inv, _ = cv2.recoverPose(E, pts1_in, pts1_in, K)

print("Matrice relative R inv :\n", R_rel_inv)
print("Vecteur relatif t inv :\n", t_rel_inv.ravel())

_, R_rel, t_rel, _ = cv2.recoverPose(E, pts1_in, pts1_in, K)
R_rel_calc = np.linalg.inv(R_rel)
R_rel_calc2 = R_rel.T
t_rel_calc = R_rel_calc @ -t_rel

print("Matrice relative R calc :\n", R_rel_calc)
print("Matrice R calc Transposée :\n", R_rel_calc2)
print("Vecteur relatif t calc :\n", t_rel_calc.ravel())

# ----------------------------------------
# 6) Construction des matrices de projection P1, P2
# ----------------------------------------
P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))       # [I|0]
P2 = K @ np.hstack((R_rel, t_rel))                      # [R|t]

# On ne triangule que les points “inliers” pour avoir un nuage propre
# Attention : triangulatePoints attend pts sous forme (2, N)
pts4D_homo = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)  # shape (4, N_inliers)

# Passage en coordonnées 3D (on divise par w)
pts3D = (pts4D_homo[:3] / pts4D_homo[3]).T  # shape (N_inliers, 3)

# sauvegarde des points en obj
save_obj("output/points/3D_points_test.obj", pts3D)

print("\nPoints 3D triangulés (chaque ligne = X, Y, Z) :")
for i, P in enumerate(pts3D):
    print(f"  Point {i+1}: {P}")

# ----------------------------------------
# 7) (Optionnel) Affichage 3D avec matplotlib
# ----------------------------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3D[:,0], pts3D[:,1], pts3D[:,2], c='orange', s=30)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Points 3D triangulés")
plt.tight_layout()
plt.show()
