import cv2
import numpy as np

from collections import namedtuple
from math import cos, sin

def Rx(rx):
    return np.array([[1, 0, 0],
                     [0, cos(rx), -sin(rx)],
                     [0, sin(rx), cos(rx)]])

def Ry(ry):
    return np.array([[cos(ry), 0, sin(ry)],
                     [0, 1, 0],
                     [-sin(ry), 0, cos(ry)]])

def Rz(rz):
    return np.array([[cos(rz), -sin(rz), 0],
                     [sin(rz), cos(rz), 0],
                     [0, 0, 1]])

def main(f=1000, cx=320, cy=240, noise_std=1, img_res=(480, 640)):
    cam_pos = np.array([[0, 0, 0],
                        [-2, -2 ,0],
                        [2, 2, 0],
                        [-2, 2, 0],
                        [2, -2, 0]])
    cam_ori = np.array([[0, 0, 0],
                        [-np.pi/12.0, np.pi/12.0, 0],
                        [np.pi/12.0, -np.pi/12.0, 0],
                        [np.pi/12.0, np.pi/12.0, 0],
                        [-np.pi/12.0, -np.pi/12.0, 0]])
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    X = []
    with open("../bin/data/box.xyz", "r") as f:
        for line in f:
            if len(line) > 3:
                x, y, z = line.split()
                X.append([float(x), float(y), float(z), float(1)])
    X = np.asarray(X).T

    for i in range(cam_pos.shape[0]):
        Rc = Rz(cam_ori[i][2]).dot(Ry(cam_ori[i][1])).dot(Rx(cam_ori[i][0]))
        tc = np.array([cam_pos[i]]).T
        Rt = np.hstack((Rc, tc))
        P = K.dot(Rt)

        x = P.dot(X)
        x /= x[2, :]

        noise = np.random.normal(0, noise_std, (x.shape[1]))
        x[:2, :] += noise

        image = np.zeros(img_res)
        for p in x[:2, :].T:
            if 0 <= p[1] <= img_res[0] and 0 <= p[0] <= img_res[1]:
                cv2.circle(image, (int(p[0]), int(p[1])), 2, 255, -1)
        cv2.imshow("3DV_Tutorial: Image Formation %d" % i, image)

        with open("output/image_formation%d.xyz" % i, "w") as f:
            for p in x[:2, :].T:
                f.write("%f %f 1\n" % (p[0], p[1]))

    cv2.waitKey(0) 

if __name__ == "__main__":
    main()