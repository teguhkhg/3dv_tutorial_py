import cv2
import numpy as np
import glob

from bundle_adjustment import MonoBA

def main(input_num=5, focal=1000, cx=320, cy=240):
    xs = []
    for i in range(input_num):
        pts = []
        with open("../bin/data/image_formation%d.xyz"%i, "r") as f:
            for line in f:
                x, y, z = line.split()
                pts.append((float(x), float(y)))
        xs.append(pts)
    xs = np.asarray(xs)
    print(xs.shape)

    ba = MonoBA()
    ba.set_camera(float(focal), np.array([cx, cy]).astype(float))
        

if __name__ == "__main__":
    main()