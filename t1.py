# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions.
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    # Dimensions of 2 images
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    h2 = img2.shape[0]
    w2 = img2.shape[1]

    # Function to resize large images but maintaining aspect ratio
    def resizing(image, w=None, h=None, inter=cv2.INTER_AREA):
        shape_tuple = None
        (h_old, w_old) = image.shape[:2]
        aspect = w / float(w_old)
        shape_tuple = (w, int(h_old * aspect))
        resized = cv2.resize(image, shape_tuple, interpolation=inter)
        return resized

    # I've commented out the image resizing code as i wasn't aware of the properties of test-set images, So i decided to play safe. But the utility image 'resizing' function can still be used.
    """
    # Resize image if it's larger than a threshold
    if h1 > 600 and w1 > 600:
        img1 = resizing(img1, w=600, h=600)
        h1 = img1.shape[0]
        w1 = img1.shape[1]

    # Resize image if it's larger than a threshold
    if h2 > 600 and w2 > 600:
        img2 = resizing(img2, w=600, h=600)
        h2 = img2.shape[0]
        w2 = img2.shape[1]
    """

    # Initialize SIFT variables for 2 images
    # Get an amount of SIFT Features. we can increase, deacrease or keep it empty depending upon image properties (i.e. Resolution, expected run time, etc)
    sift = cv2.SIFT_create(400)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Ratio test to get good matches for SIFT descriptors:
    #
    # Good points for image 1
    kp1m = []
    # Good points for image 2
    kp2m = []
    # Tuple for Good indexes in both images w.r.t kp1m & kp1m
    good = []
    norm = np.zeros((1, len(des1)))

    for ind_i, i in enumerate(des1):

        norm_row = np.array([])
        for ind_j, j in enumerate(des2):
            nrm = cv2.norm(i, j)
            norm_row = np.append(norm_row, nrm)

        # Get indexes for best 2 matches for w.r.t to i'th descriptor of 'des1'
        k1 = np.argsort(norm_row)[0]
        k2 = np.argsort(norm_row)[1]

        # Ratio test
        if norm_row[k1] < 0.7 * norm_row[k2]:
            good.append([ind_i, k1])
            kp1m.append(kp1[ind_i])
            kp2m.append(kp2[k1])

    # Return if Bad match
    if len(kp1m) < 8:
        print("Bad Match")
        return 0

    src = []
    dst = []
    for i in good:
        src.append(i[0])
        dst.append(i[1])

    arg1 = np.float32([kp1[m]
                       .pt for m in src]).reshape(-1, 1, 2)

    arg2 = np.float32([kp2[m]
                       .pt for m in dst]).reshape(-1, 1, 2)

    # Finding Homography for one image w.r.t to other image
    matrix, mask = cv2.findHomography(arg1, arg2, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Function to warp image
    def warping(img1, img2, H):

        # Dimensions of 2 images
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        list_of_points_1 = np.float32(
            [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        temp_points = np.float32(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        # Get perspective Transform for the images w.r.t to their corner
        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

        list_of_points = np.concatenate(
            (list_of_points_1, list_of_points_2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]

        H_translation = np.array([[1, 0, translation_dist[0]], [
                                 0, 1, translation_dist[1]], [0, 0, 1]])

        # Warping image
        warpedImage = cv2.warpPerspective(
            img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
        warpedImage[translation_dist[1]:rows1+translation_dist[1],
                    translation_dist[0]:cols1+translation_dist[0]] = img1

        return warpedImage

    # calling warping function to get final image
    final_image = warping(img2, img1, matrix)

    # Writing final image
    cv2.imwrite(savepath, final_image)
    return final_image


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
