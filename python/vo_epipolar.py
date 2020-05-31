import cv2
import numpy as np
import glob

def main(use_5pt=True):
    folder = sorted(glob.glob("../bin/data/KITTI_07_L/*.png"))

    f = 707.0912
    cx = 601.8873
    cy = 183.1104
    min_inlier_num = 100
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    gray_prev = cv2.imread(folder[0])
    if gray_prev is None:
        return
    if gray_prev.shape[2] > 1:
        gray_prev = cv2.cvtColor(gray_prev, cv2.COLOR_BGR2GRAY)

    camera_pose = np.eye((4))
    with open("output/vo_epipolar.xyz", "w") as traj:
        for filename in folder:
            image = cv2.imread(filename)
            if image is None:
                continue
            if image.shape[2] > 1:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            point_prev = cv2.goodFeaturesToTrack(gray_prev, 2000, 0.01, 10)
            point, m_status, err = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray, point_prev, None)
            gray_prev = gray

            if use_5pt:
                E, inlier_mask = cv2.findEssentialMat(point_prev, point, 
                    focal=f, pp=(cx, cy), method=cv2.RANSAC, prob=0.99, threshold=1)
            else:
                F, inlier_mask = cv2.findFundamentalMat(point_prev, point, 
                    cv2.FM_RANSAC, 1, 0.99)
                E = K.T.dot(F).dot(K)

            _, R, t, inlier_mask = cv2.recoverPose(E, point_prev, point, K)
            if np.sum(inlier_mask) > min_inlier_num:
                T = np.eye((4))
                T[:3, :3] = R
                T[:3, 3:] = t
                camera_pose = camera_pose.dot(np.linalg.inv(T))

            for i in range(len(inlier_mask)):
                if inlier_mask[i][0]:
                    cv2.line(image, (point_prev[i, 0, 0], point_prev[i, 0, 1]), (point[i, 0, 0], point[i, 0, 1]), (0, 0, 255))
                else:
                    cv2.line(image, (point_prev[i, 0, 0], point_prev[i, 0, 1]), (point[i, 0, 0], point[i, 0, 1]), (0, 127, 0))

            info = "Inliers: %d (%d%%),  XYZ: [%.3f, %.3f, %.3f]" % (
                len(inlier_mask), 100 * len(inlier_mask) / len(point), camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3])
            cv2.putText(image, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            traj.write("%.6f %.6f %.6f\n" % (camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3]))
            cv2.imshow("3DV Tutorial: Visual Odometry (Epipolar", image)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

if __name__ == "__main__":
    main()