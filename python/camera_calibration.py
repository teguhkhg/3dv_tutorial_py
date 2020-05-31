import cv2
import numpy as np

from math import sqrt

def main(select_images = True):
    board_pattern = (10, 7)
    board_cellsize = 0.025
    video = cv2.VideoCapture("../bin/data/chessboard.avi")

    images = []

    while True:
        ret, image = video.read()
        if not ret:
            break

        if select_images:
            cv2.imshow("3DV Tutorial: Camera Calibration", image)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
            elif key == ord('p'):
                display = image.copy()

                complete, pts = cv2.findChessboardCorners(image, board_pattern, None)
                display = cv2.drawChessboardCorners(display, board_pattern, pts, complete)
                cv2.imshow("3DV Tutorial: Camera Calibration", display)

                key = cv2.waitKey() & 0xff
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    images.append(image)
        else:
            images.append(image)
    
    video.release()
    if len(images) == 0:
        return

    objp = np.zeros((board_pattern[0]*board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp *= board_cellsize

    img_points = []
    obj_points = []
    for image in images:
        complete, pts = cv2.findChessboardCorners(image, board_pattern, None)
        if complete:
            img_points.append(pts)
            obj_points.append(objp)

    h, w = images[0].shape[:2]
    ret, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)

    rms = 0
    for i in range(len(obj_points)):
        pts, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist_coeff)
        error = cv2.norm(img_points[i], pts, cv2.NORM_L2)/len(img_points)
        rms += error**2
    rms = sqrt(rms)

    with open("output/camera_calibration.txt", "w") as f:
        f.write("## Camera Calibration Results\n")
        f.write("* The number of applied images = %d\n" % len(img_points))
        f.write("* RMS error = %f\n" % rms)
        f.write("* Camera matrix (K) = \n   %s\n    %s\n    %s\n" % (K[0], K[1], K[2]))
        f.write("* Distortion coefficient (k1, k2, p1, p2, k3, ...) = \n    %s" % dist_coeff)

if __name__ == "__main__":
    main()