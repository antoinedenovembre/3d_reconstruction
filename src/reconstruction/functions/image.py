# ======================================= IMPORTS =======================================
import cv2

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

def resize_image(img, target_h, target_w):
    return cv2.resize(img, (target_w, target_h))