import numpy as np
import g2o
import cv2
import glob

from bundle_adjustment import MonoBA

# def makeNoisyPoints(Xs, xs, )

class Frame(object):
    def __init__(self):
        pass

class Mappoint(object):
    def __init__(self):
        pass

class Measurement(object):
    def __init__(self):
        pass

class CovisibilityGraph(object):
    def __init__(self):
        pass

def main():
    img_resize = 0.25
    f_init = 500
    cx_init = -1
    cy_init = -1
    Z_init = 2
    Z_limit = 100
    ba_loss_width = 9
    min_inlier_num = 200
    ba_inlier_num = 200
    show_match = False

    fdetector = cv2.BRISK_create()
    img_keypoint = []
    img_set = []
    img_descriptor = []

    files = sorted(glob.glob("../bin/data/relief/*.jpg"))
    for filename in files:
        image = cv2.imread(filename)
        if img_resize != 1:
            width = int(image.shape[1] * img_resize)
            height = int(image.shape[0] * img_resize)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        keypoint, descriptor = fdetector.detectAndCompute(image, None)
        img_set.append(image)
        img_keypoint.append(keypoint)
        img_descriptor.append(descriptor)
        
    if len(img_set) < 2:
        return
    if cx_init < 0:
        cx_init = int(img_set[0].shape[1]/2)
    if cy_init < 0:
        cy_init = int(img_set[0].shape[0]/2)
    print(cx_init, cy_init)

    fmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    match_pair = []
    match_inlier = []
    for i in range(len(img_set)):
        for j in range(i + 1, len(img_set)):
            matches = fmatcher.match(img_descriptor[i], img_descriptor[j])
            inlier = []
            src = []
            dst = []
            for itr in matches:
                src.append(img_keypoint[i][itr.queryIdx].pt)
                dst.append(img_keypoint[j][itr.trainIdx].pt)
            src = np.asarray(src)
            dst = np.asarray(dst)
            F, inlier_mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC)
            for k in range(len(inlier_mask)):
                if inlier_mask[k]:
                    inlier.append(matches[k])
            print("3DV Tutorial: Image %d - %d are matched (%d / %d).\n"
                % (i, j, len(inlier), len(inlier_mask)))

            if len(inlier) < min_inlier_num:
                continue
            print("3DV Tutorial: Image %d - %d are selected.\n" % (i, j))
            match_pair.append((i, j))
            match_inlier.append(inlier)
            if show_match:
                match_image = cv2.drawMatches(
                    img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], matches, None, None, None, inlier_mask)
                cv2.imshow("3DV Tutorial: Structure-from-Motion", match_image)
                cv2.waitKey()
    if len(match_pair) < 1:
        return

    ba = MonoBA()
    ba.set_camera(float(f_init), np.array([cx_init, cy_init]).astype(float))



if __name__ == "__main__":
    main()