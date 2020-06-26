import cv2
import numpy as np
import glob
import g2o

from bundle_adjustment import MonoBA

def main(input_num=5, f=1000, cx=320, cy=240):
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    xs = []
    for i in range(input_num):
        pts = []
        with open("../bin/data/image_formation%d.xyz"%i, "r") as lines:
            for line in lines:
                x, y, z = line.split()
                pts.append((float(x), float(y)))
        xs.append(pts)
    xs = np.asarray(xs)

    F, inliers = cv2.findFundamentalMat(xs[0], xs[1], cv2.FM_8POINT)
    E = np.dot(K.T, F).dot(K)
    _, R, t, inliers = cv2.recoverPose(E, xs[0], xs[1], K)

    Rt = np.hstack((R, t))
    P0 = K.dot(np.eye(3, 4))
    P1 = K.dot(Rt)
    X = cv2.triangulatePoints(P0, P1, xs[0].T, xs[1].T).T
    X = X[:, :3]/X[:, 3:]

    ba = MonoBA()
    ba.set_camera(float(f), np.array([cx, cy]).astype(float))
    
    edge_id = 0
    for i in range(X.shape[0]):
        ba.add_point(i, X[i])

    for i in range(input_num):
        xss = xs[i]
        if i == 0:
            pose = g2o.SE3Quat(np.identity(3), [0, 0, 0])
            ba.add_pose(i, pose)
        else:
            if i > 1:
                _, rvec, t = cv2.solvePnP(X, xss, K, None)
                R, _ = cv2.Rodrigues(rvec)

            pose = g2o.SE3Quat(R, t[:, 0])
            ba.add_pose(i, pose)

        for j in range(xss.shape[0]):
            ba.add_edge(edge_id, j, i, xss[j])
            edge_id += 1
    
        ba.optimize()

    with open("output/bundle_adjustment_inc(point).xyz", "w") as fpts:
        for i in range(X.shape[0]):
            p = ba.get_point(i)
            # p = X[i]
            fpts.write("%f %f %f\n" % (p[0], p[1], p[2]))
            

if __name__ == "__main__":
    main()