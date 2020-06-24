import cv2
import numpy as np
import glob
import g2o

from bundle_adjustment import MonoBA

def main(input_num=5, f=1000, cx=320, cy=240):
    xs = []
    for i in range(input_num):
        pts = []
        with open("../bin/data/image_formation%d.xyz"%i, "r") as lines:
            for line in lines:
                x, y, z = line.split()
                pts.append((float(x), float(y)))
        xs.append(pts)
    xs = np.asarray(xs)
    
    X = np.zeros((xs.shape[1], 3))
    X[:, 2] = 5.5

    ba = MonoBA()
    ba.set_camera(float(f), np.array([cx, cy]).astype(float))
    
    edge_id = 0
    for i in range(X.shape[0]):
        ba.add_point(i, X[i])

    for i in range(input_num):
        pose = g2o.SE3Quat(np.identity(3), [0, 0, 0])
        ba.add_pose(i, pose)

        xss = xs[i]
        for j in range(xss.shape[0]):
            ba.add_edge(edge_id, j, i, xss[j])
            edge_id += 1
    
    ba.optimize()

    with open("output/bundle_adjustment_global(point).xyz", "w") as fpts:
        for i in range(X.shape[0]):
            p = ba.get_point(i)
            # p = X[i]
            fpts.write("%f %f %f\n" % (p[0], p[1], p[2]))
        

if __name__ == "__main__":
    main()