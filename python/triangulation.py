import cv2
import numpy as np

def main():
    f = 1000
    cx = 320
    cy = 240
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    points0 = []
    with open("output/image_formation0.xyz") as infile:
        for line in infile:
            x, y, z = line.split()
            points0.append([float(x), float(y)])
    points0 = np.asarray(points0)

    points1 = []
    with open("output/image_formation1.xyz") as infile:
        for line in infile:
            x, y, z = line.split()
            points1.append([float(x), float(y)])
    points1 = np.asarray(points1)
    
    if points0.shape != points1.shape:
        return

    F, _ = cv2.findFundamentalMat(points0, points1, cv2.FM_8POINT)
    E = K.T.dot(F).dot(K)
    _, R, t, _ = cv2.recoverPose(E, points0, points1, K)
    
    P0 = K.dot(np.eye(3, 4))
    Rt = np.hstack((R, t))
    P1 = K.dot(Rt)

    X = cv2.triangulatePoints(P0, P1, np.expand_dims(points0, axis=1), np.expand_dims(points1, axis=1))
    X = X/X[3, :]

    with open("output/triangulation.xyz", "w") as outfile:
        for p in X[:3, :].T:
            outfile.write("%f %f %f\n" % (p[0], p[1], p[2]))

if __name__ == "__main__":
    main()