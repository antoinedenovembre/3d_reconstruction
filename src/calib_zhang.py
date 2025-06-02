import cv2
import glob
import numpy as np
import scipy.optimize as opt
import scipy

CHESSBOARD_SIZE = (10, 7)  # Number of inner corners per chessboard row and column
CHESSBOARD_DIM = 21.5 # Size of a chessboard square in cm

def homography(images, world_pts, save_folder):
    H_list = []
    img_pts = []

    for i, img in enumerate(images):
        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # call to opencv to find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        # if corners are found
        if ret: 
            # refine the corners (Nx1x2 to Nx2) with N number of corners
            corners = corners.reshape(-1, 2)

            # call to opencv to find homography (finds a transformation matrix between two planes, here world and image plane)
            H, _ = cv2.findHomography(world_pts, corners, cv2.RANSAC, 5.0)

            # append the homography matrix to the list
            H_list.append(H)

            # append the corners coordinates to the list
            img_pts.append(corners)

            # draw the corners on the image
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, True)
            img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
            cv2.imwrite(save_folder + '/corners/' + str(i) + '_corners.png', img)
        else:
            print(f"Chessboard corners not found in image {i}. Skipping this image.")

    # return the list of homography matrices and image points
    return H_list, img_pts

def v(i, j, H):
    # compute the vector v from the homography matrix H
    return np.array([
        H[0][i] * H[0][j],
        H[0][i] * H[1][j] + H[1][i] * H[0][j],
        H[1][i] * H[1][j],
        H[2][i] * H[0][j] + H[0][i] * H[2][j],
        H[2][i] * H[1][j] + H[1][i] * H[2][j],
        H[2][i] * H[2][j]
    ])

def compute_intrinsics(H_list):
    V = []

    # for each homography matrix
    for h in H_list:
        # compute the vector v (for current homography matrix)
        V.append(v(0, 1, h))
        V.append(v(0, 0, h) - v(1, 1, h))
    
    # create V from the n homography matrices, V * b = 0 (equation 8 in Zhang et al. paper)
    V = np.array(V)

    # retrieve conjugate-transpose of V
    _, _, vt = np.linalg.svd(V)

    # compute b from vt
    b = vt[-1][:]

    # compute B as lambda * K-t * K
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    # from B, we can compute the intrinsic parameters
    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2) # principal point (pixel coordinates)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11 # arbitrary scale-factor
    alpha_u = np.sqrt(lamda/B11) # focal length in x direction (pixels)
    alpha_v = np.sqrt(lamda*B11 / (B11*B22 - B12**2))
    gamma = -1*B12*(alpha_u**2)*alpha_v/lamda
    u0 = (gamma*v0/alpha_v) - (B13*(alpha_u**2)/lamda)

    # construct the intrinsic matrix K
    K = np.array([[alpha_u, gamma,   u0],
                  [0,       alpha_v, v0],
                  [0,       0,       1]])
    
    return K


def compute_extrinsics(K, H_list):
    # compute the inverse of the intrinsic matrix K
    K_inv = np.linalg.inv(K)
    R_list = []
    t_list = []

    # for each homography matrix
    for H in H_list:
        # retrieve H elements
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        # compute the scale factor lambda
        lamda = 1/scipy.linalg.norm((K_inv @ h1), 2)
        
        # compute the rotation and translation vectors
        r1 = lamda * (K_inv @ h1)
        r2 = lamda * (K_inv @ h2)
        r3 = np.cross(r1, r2)
        t = lamda * (K_inv @ h3)
        
        # append the rotation and translation vectors to the lists
        R_list.append(np.array([r1, r2, r3]))
        t_list.append(t)

    return R_list, t_list

if __name__ == '__main__':
    '''
    Steps :
    - Compute homography for each image
    - Compute intrinsic matrix K
    - Compute extrinsic matrix R and t
    '''

    data = 'data/calib/'
    save = 'output/'

    # get all images from the data folder
    images = [cv2.imread(file) for file in glob.glob(data + '*.jpeg')]

    if not images:
        raise ValueError("No images found in the specified directory.")

    # create world points for the chessboard
    world_pts_x, world_pts_y = np.meshgrid(range(CHESSBOARD_SIZE[0]), range(CHESSBOARD_SIZE[1]))

    # reshape and scale the world points
    world_pts = np.array(np.hstack((world_pts_x.reshape(CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 1), world_pts_y.reshape(CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 1))).astype(np.float32)*CHESSBOARD_DIM)

    # calculate homography for each image and get the image points
    H_list, img_pts = homography(images, world_pts, save)

    # compute the intrinsic matrix K
    K = compute_intrinsics(H_list)

    # compute the extrinsic matrices R and t
    R_list, t_list = compute_extrinsics(K, H_list)

    # save K matrix to a file
    np.savetxt(save + 'K.txt', K, fmt='%.6f')
    print("Saved intrinsic matrix K to output/K.txt")
