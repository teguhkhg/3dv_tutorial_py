import cv2
import numpy as np

def main():
    image1 = cv2.imread("../bin/data/hill01.jpg")
    image2 = cv2.imread("../bin/data/hill02.jpg")

    fdetector = cv2.ORB_create()
    fmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    keypoint1, descriptor1 = fdetector.detectAndCompute(image1, None)
    keypoint2, descriptor2 = fdetector.detectAndCompute(image2, None)
    match = fmatcher.match(descriptor1, descriptor2)

    points1 = []
    points2 = []
    for m in match:
        points1.append(keypoint1[m.queryIdx].pt)
        points2.append(keypoint2[m.trainIdx].pt)
    
    H, inlier_mask = cv2.findHomography(
        np.asarray(points2, dtype=np.float32), np.asarray(points1, dtype=np.float32), cv2.RANSAC)
    merged = cv2.warpPerspective(image2, H, (image1.shape[1]*2, image1.shape[0]))
    merged[:, :image1.shape[1]] = image1

    matched = cv2.drawMatches(image1, keypoint1, image2, keypoint2, match, None,
        (0, 0, 255), (0, 127, 0), inlier_mask)
    original = np.hstack((image1, image2))
    matched = np.vstack((original, matched))
    merged = np.vstack((matched, merged))
    cv2.imshow("3DV Tutorial: Image Stitching", merged)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()