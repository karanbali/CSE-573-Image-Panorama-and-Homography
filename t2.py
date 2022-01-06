# 1. Only add your code inside the function (including newly improted packages).
#  You can design a new function and call the new function in the given functions.
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


# For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
def stitch(imgmark, N=4, savepath=''):
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    def stitch_background(img1, img2, flag=1):

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
        # Get an amount SIFT Features. we can increase, deacrease or keep it empty depending upon image properties (i.e. Resolution, expected run time, etc)
        sift = cv2.SIFT_create(800)
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

        # if flag=1 (i.e. just want to know whether 2 images are good/bad match)
        if flag == 1:
            # Return if Bad match
            if len(kp1m) < 25:
                return 0
            # Return good match
            else:
                return 1

        # Return if Bad match
        if len(kp1m) < 25:
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
        matrix, mask = cv2.findHomography(
            arg1, arg2, cv2.RANSAC, 5.0)
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

        return final_image

    # Initializing overlap array to return good/bad matches between all images
    overlap = np.zeros((N, N), dtype=int)

    # Loop through all images twice to get overlap array (with flag=1, i.e. just want to know whether 2 images are good/bad match)
    for i in range(N):
        for j in range(N):
            if i == j:
                # if mathching between same image
                overlap[i, j] = 1
                continue
            # get overlap array '[i,j]' element (with flag=1, i.e. just want to know whether 2 images are good/bad match)
            overlap[i, j] = stitch_background(imgs[i], imgs[j], flag=1)

    #overlap = [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]
    #overlap = np.array(overlap)
    # Get the image with maximum matches
    max_i = np.argmax(np.sum(overlap, axis=0))
    # We'll consider that images as the main 'anchor image'
    anchor_img = imgs[max_i]

    j_list = []
    j_list.append(max_i)
    # loop through overlap array
    for i in range(N):
        for j in range(N):

            if np.any(np.isin(j_list, j)) == True:
                continue

            if i == j:
                continue
            # If found a match then compute a new stitched image using the anchor image....then replace the anchor image by newly computed stitched image
            if overlap[i, j] == 1:
                # compute a new stitched image using the anchor image & j'th image with flag=0 (i.e. want to get a new warped image)
                anchor_img = stitch_background(anchor_img, imgs[j], flag=0)
                j_list.append(j)

    # Writing final image
    cv2.imwrite(savepath, anchor_img)

    # return overlap array
    return overlap


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', N=4, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
